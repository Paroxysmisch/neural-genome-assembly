from dataset import Genome, GenomeDataset
from layers import MambaEncoder, MambaDecoder

genome_dataset = GenomeDataset(Genome.CHM13, 19)
sample = genome_dataset[1]
encoder = MambaEncoder().to("cuda")
# decoder = MambaDecoder().to("cuda")

encoded = encoder(sample["reads"].to("cuda"))
# decoded = decoder(encoded, sample["reads"][:-1].to("cuda"))
breakpoint()
