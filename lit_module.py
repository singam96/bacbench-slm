import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from model import ProteinCausalLM


class LitProteinCausalLM(pl.LightningModule):
    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        max_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        lr: float,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pad_id = int(pad_id)
        self.model = ProteinCausalLM(
            vocab_size=int(vocab_size),
            d_model=int(d_model),
            n_heads=int(n_heads),
            n_layers=int(n_layers),
            max_len=int(max_len),
            dropout=float(dropout),
            pad_id=self.pad_id,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        logits = self(input_ids=input_ids, attention_mask=attention_mask)

        targets = input_ids[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=self.pad_id)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        logits = self(input_ids=input_ids, attention_mask=attention_mask)

        targets = input_ids[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=self.pad_id)
        ppl = torch.exp(loss.detach().clamp(max=20))

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_ppl", ppl, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        lr = float(self.hparams.lr)
        weight_decay = float(getattr(self.hparams, "weight_decay", 0.0))
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
