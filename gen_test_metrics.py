import glob
import statistics

import torch
from torch.utils.data import DataLoader

from dataset import *
from lightning_modules import FusedTrainingModule, TrainingModule

data = GenomeDataset(Genome.CHM13, 21, read_overlap_len=None)
loss_fn = torch.nn.CrossEntropyLoss()

losses = []
accuracies = []

transformer_runs = ["mm21ar3y", "enhty7hy", "j3wq9sew", "bakerhse", "l4bly10n"]
mamba_runs = ["hxb2vrmr", "0qxwngia", "4aly6anh", "vnmmsuqu", "aai1emla"]

for run in mamba_runs:
    model = FusedTrainingModule.load_from_checkpoint(
        glob.glob(
            f"/home/yash/Projects/neural-genome-assembly/neural-genome-assembly/{run}/checkpoints/epoch*"
        )[0]
    ).to("cuda")
    model.eval()

    for sample in data:
        sample["reads"] = sample["reads"].to("cuda")
        sample["target"] = sample["target"].to("cuda")
        sample["overlap_lens"] = sample["overlap_lens"].to("cuda")

        with torch.no_grad():
            result = model.forward(sample)
        target = sample["target"][1:]
        losses.append(loss_fn(result, target).item())
        accuracy = (torch.argmax(result, dim=-1) == target).sum() / target.numel()
        accuracies.append(accuracy.item())

print(f"Loss = ({statistics.mean(losses)}, {statistics.stdev(losses)})")
print(f"Accuracy = ({statistics.mean(accuracies)}, {statistics.stdev(accuracies)})")
breakpoint()
