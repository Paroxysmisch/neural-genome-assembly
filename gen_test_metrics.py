import statistics

import torch
from torch.utils.data import DataLoader

from dataset import *
from lightning_modules import FusedTrainingModule, TrainingModule

data = GenomeDataset(Genome.CHM13, 18, read_overlap_len=None)
loss_fn = torch.nn.CrossEntropyLoss()

losses = []
accuracies = []

model = FusedTrainingModule.load_from_checkpoint(
    "/home/yash/Projects/neural-genome-assembly/neural-genome-assembly/qrui6zq3/checkpoints/last.ckpt"
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
