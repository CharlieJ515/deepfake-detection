import torch
from torch.utils.tensorboard import SummaryWriter


class BinaryClassificationMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.tn = self.fp = self.fn = 0

    @torch.no_grad()
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        preds = preds.view(-1).long()
        labels = labels.view(-1).long()

        self.tp += ((preds == 1) & (labels == 1)).sum().item()
        self.tn += ((preds == 0) & (labels == 0)).sum().item()
        self.fp += ((preds == 1) & (labels == 0)).sum().item()
        self.fn += ((preds == 0) & (labels == 1)).sum().item()

    @torch.no_grad()
    def update_from_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.5,
    ):
        preds = (logits.sigmoid() > threshold).long()
        self.update(preds, labels)

    @property
    def f1_score(self):
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn
        f1_pos = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        f1_neg = (2 * tn) / (2 * tn + fp + fn + 1e-8)
        f1_score = (f1_pos + f1_neg) / 2
        return f1_score, f1_pos, f1_neg

    @property
    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + 1e-8)

    @property
    def balanced_accuracy(self) -> float:
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn
        tpr = tp / (tp + fn + 1e-8)
        tnr = tn / (tn + fp + 1e-8)
        return (tpr + tnr) / 2

    def log(self, writer: SummaryWriter, tag: str, step: int):
        f1_score, f1_pos, f1_neg = self.f1_score
        accuracy = self.accuracy

        writer.add_scalars(
            main_tag=tag,
            tag_scalar_dict={
                "f1_pos": f1_pos,
                "f1_neg": f1_neg,
                "f1_score": f1_score,
                "accuracy": accuracy,
            },
            global_step=step,
        )
