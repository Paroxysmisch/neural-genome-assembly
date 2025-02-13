import lightning as L
import torch
from torch.optim.adam import Adam

from layers import MambaDecoder, MambaEncoder


class TrainingModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = MambaEncoder()
        self.decoder = MambaDecoder()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        encoded = self.encoder(batch["reads"])
        decoded = self.decoder(encoded, batch["reads"][batch["target"]][:-1])
        one_hot_target = torch.nn.functional.one_hot(batch["target"])
        loss = self.loss(decoded, one_hot_target[1:].float())
        # loss = self.loss(decoded, one_hot_target[:-1].float())

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def forward(self, batch):
        # Unsqueeze to add additional dimension required by Mamba
        encoded = self.encoder(batch["reads"])
        decoded = self.decoder(encoded, batch["reads"][batch["target"]][:-1])
        return decoded

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4)
        return optimizer
