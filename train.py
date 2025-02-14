import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader

from dataset import *
from lightning_modules import FusedTrainingModule, TrainingModule

seed_everything(42, workers=True)

train_loader = DataLoader(
    GenomeDataset(Genome.CHM13, 19),
    batch_size=1,
    shuffle=True,
    collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
    num_workers=8,
)
val_loader = DataLoader(
    GenomeDataset(Genome.CHM13, 18),
    batch_size=1,
    shuffle=True,
    collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
    num_workers=8,
)
# model = TrainingModule()
model = FusedTrainingModule()

checkpoint_callback = ModelCheckpoint(
    monitor="validation_loss",
    mode="min", # "min" for loss, "max" for accuracy
    save_top_k=3,
    filename="epoch{epoch}-validation_loss{validation_loss:.2f}",
    save_last=True,
)
logger = TensorBoardLogger(save_dir="tb_logs", name="cross_entropy_loss")
trainer = L.Trainer(
    max_epochs=100,
    log_every_n_steps=1,
    deterministic=True,
    logger=logger,
    check_val_every_n_epoch=5,
    callbacks=[checkpoint_callback],
)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
