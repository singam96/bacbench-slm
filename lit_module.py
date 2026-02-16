import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from model import ProteinBiLSTM, ProteinCausalLM


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
        model_type: str = "transformer",
        task_type: str = "causal_lm", # 'causal_lm' or 'classification'
        num_classes: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pad_id = int(pad_id)
        self.task_type = task_type
        
        if model_type == "transformer":
            self.model = ProteinCausalLM(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                max_len=max_len,
                dropout=dropout,
                pad_id=self.pad_id,
            )
        elif model_type == "bilstm":
            self.model = ProteinBiLSTM(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                dropout=dropout,
                pad_id=self.pad_id,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if self.task_type == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification task")
            self.model.set_classification_head(num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        logits = self(input_ids=input_ids, attention_mask=attention_mask)

        if self.task_type == "causal_lm":
            targets = input_ids[:, 1:].contiguous()
            logits = logits[:, :-1, :].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=self.pad_id)
        elif self.task_type == "classification":
            targets = batch["labels"]
            
            # Handle case where all targets are ignored (-100)
            if (targets == -100).all():
                 # Return 0 loss with grad enabled to satisfy DDP/Trainer
                 loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                 loss = F.cross_entropy(logits, targets, ignore_index=-100)
            
            preds = torch.argmax(logits, dim=1)
            
            # Calculate accuracy only on non-ignored targets
            valid_mask = (targets != -100)
            if valid_mask.any():
                acc = (preds[valid_mask] == targets[valid_mask]).float().mean()
            else:
                acc = torch.tensor(0.0, device=self.device)
                
            self.log("train_acc", acc, prog_bar=True)
            
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        logits = self(input_ids=input_ids, attention_mask=attention_mask)

        if self.task_type == "causal_lm":
            targets = input_ids[:, 1:].contiguous()
            logits = logits[:, :-1, :].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=self.pad_id)
            ppl = torch.exp(loss.detach().clamp(max=20))
            self.log("val_ppl", ppl, prog_bar=True, on_step=False, on_epoch=True)
        elif self.task_type == "classification":
            targets = batch["labels"]
            
            # Handle case where all targets are ignored (-100)
            if (targets == -100).all():
                 loss = torch.tensor(0.0, device=self.device) # Validation doesn't need grad
            else:
                 loss = F.cross_entropy(logits, targets, ignore_index=-100)
            
            preds = torch.argmax(logits, dim=1)
            
            valid_mask = (targets != -100)
            if valid_mask.any():
                acc = (preds[valid_mask] == targets[valid_mask]).float().mean()
            else:
                acc = torch.tensor(0.0, device=self.device)
            
            self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        lr = float(self.hparams.lr)
        weight_decay = float(getattr(self.hparams, "weight_decay", 0.0))
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
