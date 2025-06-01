import torch
from torch.utils.data import DataLoader

from dataset import *
from lightning_modules import FusedTrainingModule, TrainingModule

data = GenomeDataset(Genome.CHM13, 18, read_overlap_len=None)
sample = data[1]
sample["reads"] = sample["reads"].to("cuda")
sample["target"] = sample["target"].to("cuda")
sample["overlap_lens"] = sample["overlap_lens"].to("cuda")


model = FusedTrainingModule.load_from_checkpoint(
    # "lightning_logs/version_19/checkpoints/epoch=49-step=50000.ckpt",
    # "tb_logs/real_run_shuffled/version_0/checkpoints/epoch=99-step=100000.ckpt",
    # "tb_logs/fused_multi_head_3_mamba/version_1/checkpoints/epoch=99-step=100000.ckpt",
    # "tb_logs/random_read_starts/version_3/checkpoints/last.ckpt"
    # "tb_logs/overlap_info_normalized_max/version_0/checkpoints/last.ckpt"
    "tb_logs/overlap_info_mlp/version_0/checkpoints/last.ckpt"
).to("cuda")

result = model.forward(sample)
print(torch.argmax(result, dim=-1))
print(sample["target"][1:])
loss_fn = torch.nn.CrossEntropyLoss()
print(loss_fn(result, sample["target"][1:]))

acc_mask = (torch.argmax(result, dim=-1) == sample["target"][1:])
counts = {i.item(): v.item() for i, v in zip(*torch.argmax(result, dim=-1).unique(return_counts=True))}
lovro = [counts.get(i, 0) for i in torch.argmax(result, dim=-1)[acc_mask].tolist()]
lovro = torch.tensor(lovro)
print((lovro > 1).sum() / len(lovro))
breakpoint()
