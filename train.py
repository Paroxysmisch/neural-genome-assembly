import argparse

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from dataset import *
from layers import MambaEncoderDecoder
from lightning_modules import FusedTrainingModule, TrainingModule


def run(seed):
    seed_everything(seed, workers=True)

    train_loader = DataLoader(
        GenomeDataset(Genome.CHM13, 15, read_overlap_len=None),
        batch_size=1,
        shuffle=True,
        collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
        num_workers=8,
    )
    val_loader = DataLoader(
        GenomeDataset(Genome.CHM13, 22, read_overlap_len=None),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
        num_workers=8,
    )
    # model = TrainingModule()
    model = FusedTrainingModule()
    # model = FusedTrainingModule(encoded_decoder=MambaEncoderDecoder)

    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        mode="min",  # "min" for loss, "max" for accuracy
        save_top_k=3,
        filename="epoch{epoch}-validation_loss{validation_loss:.2f}",
        save_last=True,
    )
    # logger = TensorBoardLogger(save_dir="tb_logs", name="overlap_info_mlp")
    logger = WandbLogger(
        entity="paroxysmisch-university-of-cambridge",
        project="neural-genome-assembly",
        log_model="all",
    )
    trainer = L.Trainer(
        max_epochs=40,
        log_every_n_steps=1,
        deterministic=True,
        logger=logger,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int)
    args = parser.parse_args()
    run(args.seed)
