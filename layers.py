import torch
import torch.nn as nn
from einops import rearrange, reduce
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
    def __init__(self, d_model=64, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.expander = nn.Linear(4, d_model)
        self.reads_encoder = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        # d_conv set to 1 to prevent Mamba from cheating by looking at future tokens
        self.query_project = nn.Linear(d_model * 2, d_model)
        self.key_project = nn.Linear(d_model * 2, d_model)

    def forward(self, reads, sequenced_reads_indices):
        reads = self.expander(reads)
        encoded = self.reads_encoder(reads)
        mean_pooled = reduce(encoded, "b l d_model -> b d_model", "mean")
        max_pooled = reduce(encoded, "b l d_model -> b d_model", "max")
        last_pooled = encoded[:, -1, :]
        encoded = rearrange(
            [mean_pooled, last_pooled], "type b d_model -> b (type d_model)"
        )

        sequenced_reads = encoded[sequenced_reads_indices]
        queries = self.query_project(sequenced_reads)
        keys = self.key_project(encoded)
        attention_scores = torch.einsum("nd,md->nm", queries, keys)
        return attention_scores
