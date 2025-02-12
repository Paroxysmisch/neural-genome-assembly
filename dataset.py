from enum import Enum

import torch
from einops import rearrange
from torch.utils.data import Dataset


class Genome(Enum):
    CHM13 = "chm13"


class GenomeDataset(Dataset):
    def __init__(
        self,
        genome: Genome,
        chromosome: int,
        seq_len: int = 10000,
        read_len: int = 100,
        read_overlap_len=25,
        transform=None,
    ):
        self.read_len = read_len
        self.read_overlap_len = read_overlap_len
        self.transform = transform

        filename = genome.value + "/chromosomes/chr" + str(chromosome) + ".fasta"
        with open(filename, "r") as f:
            data = f.readlines()[1:]
            data = [line.rstrip() for line in data]
            self.raw_data = "".join(data)

        indices = []
        for char in self.raw_data:
            match char:
                case "A":
                    indices.append(0)
                case "T":
                    indices.append(1)
                case "C":
                    indices.append(2)
                case "G":
                    indices.append(3)
        self.data = torch.nn.functional.one_hot(
            torch.tensor(indices), num_classes=4
        ).float()
        num_bases, _ = self.data.size()
        self.data = self.data[: num_bases - num_bases % seq_len]
        self.data = rearrange(self.data, "(b l) f -> b l f", l=seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        complete_sequence = self.data[idx]
        reads = complete_sequence.unfold(
            dimension=0,
            size=self.read_len,
            step=self.read_len - self.read_overlap_len,
        )[
            :-1
        ]  # drop the final read in case the complete seq_len is not divisible by the number of windows produced
        reads = rearrange(
            reads, "num_reads features read_length -> num_reads read_length features"
        )
        random_permutation = torch.randperm(len(reads))
        random_permutation_inverse = torch.argsort(random_permutation)

        sample = {
            "reads": reads[random_permutation],
            "target": random_permutation_inverse,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# genome_dataset = GenomeDataset(Genome.CHM13, 19)
# sample = genome_dataset[1]
# reads = sample["reads"][sample["target"]]
# print(sample["reads"][:5])
# print(sample["target"][:5])
# print(reads[0, -25:] == reads[1, :25])
