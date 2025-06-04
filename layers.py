import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from mamba_ssm import Mamba


class MambaEncoder(nn.Module):
    def __init__(self, d_model=64, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.expander = nn.Linear(4, d_model)
        self.encoder = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, reads):
        reads = self.expander(reads)
        encoded = self.encoder(reads)
        mean_pooled = reduce(encoded, "b l d_model -> b d_model", "mean")
        max_pooled = reduce(encoded, "b l d_model -> b d_model", "max")
        last_pooled = encoded[:, -1, :]
        return rearrange(
            [mean_pooled, last_pooled], "type b d_model -> b (type d_model)"
        )


class MambaDecoder(nn.Module):
    # The decoder takes as input the encoded sequence in the correct order
    # and tries to predict a one-hot vector of the index of the next chunk
    def __init__(self, d_model=64, d_hidden=32, d_state=64, expand=2):
        super().__init__()
        # d_conv set to 1 to prevent Mamba from cheating by looking at future tokens
        self.expander = nn.Linear(4, d_model)
        self.decoder = Mamba(d_model=d_model, d_state=d_state, d_conv=1, expand=expand)
        self.query_project = nn.Linear(d_model * 2, d_hidden)
        self.key_project = nn.Linear(d_model * 2, d_hidden)

    def forward(self, encoded_reads, decoded_reads):
        decoded_reads = self.expander(decoded_reads)
        decoded = self.decoder(decoded_reads)
        mean_pooled = reduce(decoded, "b l d_model -> b d_model", "mean")
        max_pooled = reduce(decoded, "b l d_model -> b d_model", "max")
        last_pooled = decoded[:, -1, :]
        decoded = rearrange(
            [mean_pooled, last_pooled], "type b d_model -> b (type d_model)"
        )
        queries = self.query_project(decoded)
        keys = self.key_project(encoded_reads)
        attention_scores = torch.einsum("nd,md->nm", queries, keys)
        # breakpoint()
        # attention_scores = nn.Softmax(dim=-1)(attention_scores)
        return attention_scores


class MambaEncoderDecoder(nn.Module):
    def __init__(
        self, d_model=64, num_heads=4, d_state=64, d_conv=4, expand=2, num_mamba_stack=3
    ):
        super().__init__()
        self.expander = nn.Linear(4, d_model)
        mamba_layers = []
        for _ in range(num_mamba_stack):
            mamba_layers += [
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
                nn.LayerNorm(d_model),
                nn.ReLU(),
            ]
        self.reads_encoder = nn.Sequential(*mamba_layers)
        self.reads_decoder = Mamba(
            d_model=d_model * 2, d_state=d_state, d_conv=1, expand=expand
        )
        # d_conv set to 1 to prevent Mamba from cheating by looking at future tokens
        self.query_project = nn.Linear(d_model * 2, d_model)
        self.key_project = nn.Linear(d_model * 2, d_model)
        assert d_model % num_heads == 0, "d_model should be divisible by num_heads."
        self.num_heads = num_heads
        self.ReLU = torch.nn.ReLU()
        self.mix_head_info_1 = nn.Linear(num_heads + 1, num_heads + 1)
        self.mix_head_info_2 = nn.Linear(num_heads + 1, num_heads + 1)

    def forward(self, reads, sequenced_reads_indices, overlap_lens):
        reads = self.expander(reads)
        encoded = self.reads_encoder(reads)
        mean_pooled = reduce(encoded, "b l d_model -> b d_model", "mean")
        max_pooled = reduce(encoded, "b l d_model -> b d_model", "max")
        last_pooled = encoded[:, -1, :]
        encoded = rearrange(
            [mean_pooled, last_pooled], "type b d_model -> b (type d_model)"
        )

        sequenced_reads = encoded[sequenced_reads_indices]
        sequenced_reads = self.reads_decoder(sequenced_reads.unsqueeze(0)).squeeze(0)
        queries = self.query_project(sequenced_reads)
        queries = rearrange(queries, "q (h d) -> h q d", h=self.num_heads)
        keys = self.key_project(encoded)
        keys = rearrange(keys, "k (h d) -> h k d", h=self.num_heads)
        attention_scores = torch.einsum("hqd,hkd->qkh", queries, keys)
        # Undo just the row permutation on overlap_lens to match the sequenced_reads
        overlap_info = overlap_lens[sequenced_reads_indices]
        overlap_info = repeat(overlap_info, "q k -> q k 1")
        # Add overlap information as another head
        attention_scores = torch.cat([attention_scores, overlap_info], dim=-1)
        attention_scores = self.ReLU(self.mix_head_info_1(attention_scores))
        attention_scores = self.mix_head_info_2(attention_scores)
        attention_scores = reduce(attention_scores, "q k h -> q k", "mean")
        return attention_scores


class TransformerEncoderDecoder(nn.Module):
    def __init__(
        self, d_model=64, num_heads=4, d_state=64, d_conv=4, expand=2, num_mamba_stack=3
    ):
        super().__init__()
        self.expander = nn.Linear(4, d_model)
        mamba_layers = [
            nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=num_heads, batch_first=True
                ),
                num_layers=num_mamba_stack,
                norm=nn.LayerNorm(d_model),
            )
        ]
        self.reads_encoder = nn.Sequential(*mamba_layers)
        # self.reads_decoder = Mamba(
        #     d_model=d_model * 2, d_state=d_state, d_conv=1, expand=expand
        # )
        self.reads_decoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model * 2, nhead=num_heads, batch_first=True
            ),
            num_layers=2,
            norm=nn.LayerNorm(d_model * 2),
        )
        # d_conv set to 1 to prevent Mamba from cheating by looking at future tokens
        self.query_project = nn.Linear(d_model * 2, d_model)
        self.key_project = nn.Linear(d_model * 2, d_model)
        assert d_model % num_heads == 0, "d_model should be divisible by num_heads."
        self.num_heads = num_heads
        self.ReLU = torch.nn.ReLU()
        self.mix_head_info_1 = nn.Linear(num_heads + 1, num_heads + 1)
        self.mix_head_info_2 = nn.Linear(num_heads + 1, num_heads + 1)

    def forward(self, reads, sequenced_reads_indices, overlap_lens):
        reads = self.expander(reads)
        encoded = self.reads_encoder(reads)
        mean_pooled = reduce(encoded, "b l d_model -> b d_model", "mean")
        max_pooled = reduce(encoded, "b l d_model -> b d_model", "max")
        last_pooled = encoded[:, -1, :]
        encoded = rearrange(
            [mean_pooled, last_pooled], "type b d_model -> b (type d_model)"
        )

        sequenced_reads = encoded[sequenced_reads_indices]
        sequenced_reads = self.reads_decoder(sequenced_reads.unsqueeze(0)).squeeze(0)
        queries = self.query_project(sequenced_reads)
        queries = rearrange(queries, "q (h d) -> h q d", h=self.num_heads)
        keys = self.key_project(encoded)
        keys = rearrange(keys, "k (h d) -> h k d", h=self.num_heads)
        attention_scores = torch.einsum("hqd,hkd->qkh", queries, keys)
        # Undo just the row permutation on overlap_lens to match the sequenced_reads
        overlap_info = overlap_lens[sequenced_reads_indices]
        overlap_info = repeat(overlap_info, "q k -> q k 1")
        # Add overlap information as another head
        attention_scores = torch.cat([attention_scores, overlap_info], dim=-1)
        attention_scores = self.ReLU(self.mix_head_info_1(attention_scores))
        attention_scores = self.mix_head_info_2(attention_scores)
        attention_scores = reduce(attention_scores, "q k h -> q k", "mean")
        return attention_scores
