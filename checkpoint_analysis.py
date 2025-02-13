from torch.utils.data import DataLoader

from dataset import *
from lightning_modules import TrainingModule

data = GenomeDataset(Genome.CHM13, 19)
sample = data[1]
sample["reads"] = sample["reads"].to("cuda")
sample["target"] = sample["target"].to("cuda")

model = TrainingModule.load_from_checkpoint(
    # "lightning_logs/version_19/checkpoints/epoch=49-step=50000.ckpt",
    "tb_logs/real_run_shuffled/version_0/checkpoints/epoch=99-step=100000.ckpt",
).to("cuda")

result = model.forward(sample)
print(torch.argmax(result, dim=-1))
print(sample["target"][1:])
breakpoint()
