import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dataset import *
from lightning_modules import TrainingModule, FusedTrainingModule

seed_everything(42, workers=True)

train_loader = DataLoader(
    GenomeDataset(Genome.CHM13, 19),
    batch_size=1,
    shuffle=True,
    collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
    num_workers=16,
)
# model = TrainingModule()
model = FusedTrainingModule()

logger = TensorBoardLogger(save_dir="tb_logs", name="fused_multi_head_3_mamba")
trainer = L.Trainer(
    max_epochs=100, log_every_n_steps=1, deterministic=True, logger=logger
)
trainer.fit(model=model, train_dataloaders=train_loader)
