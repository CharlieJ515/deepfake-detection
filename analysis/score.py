from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class BinaryClassificationMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.tn = self.fp = self.fn = 0
        self._logits = []
        self._labels = []

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
        self._logits.append(logits.detach().view(-1).cpu())
        self._labels.append(labels.detach().view(-1).cpu())

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

    def plot_prob(
        self,
        save_path: Path | None = None,
        title="Probability Distribution",
    ):
        if len(self._logits) == 0:
            print("No logits stored.")
            return

        plt.figure(figsize=(8, 5))

        logits = torch.cat(self._logits)
        labels = torch.cat(self._labels)

        probs = torch.sigmoid(logits)
        pos_probs = probs[labels == 1]
        neg_probs = probs[labels == 0]
        plt.hist(neg_probs, color="blue", bins=40, alpha=0.6, label="Negative(0)")
        plt.hist(pos_probs, color="orange", bins=40, alpha=0.6, label="Positive(1)")

        neg_mid = neg_probs.mean().item()
        pos_mid = pos_probs.mean().item()
        plt.axvline(neg_mid, color="blue", linestyle="--", linewidth=2)
        plt.axvline(pos_mid, color="orange", linestyle="--", linewidth=2)

        plt.title(title)
        plt.xlabel("Probability")
        plt.ylabel("Count")
        plt.legend()

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_logit(
        self,
        save_path: Path | None = None,
        title: str = "Logit Distribution",
    ):
        if len(self._logits) == 0:
            print("No logits stored.")
            return

        plt.figure(figsize=(8, 5))

        logits = torch.cat(self._logits)
        labels = torch.cat(self._labels)

        pos_logits = logits[labels == 1]
        neg_logits = logits[labels == 0]
        plt.hist(neg_logits, color="blue", bins=40, alpha=0.6, label="Negative(0)")
        plt.hist(pos_logits, color="orange", bins=40, alpha=0.6, label="Positive(1)")

        neg_mid = neg_logits.mean().item()
        pos_mid = pos_logits.mean().item()
        plt.axvline(neg_mid, color="blue", linestyle="--", linewidth=2)
        plt.axvline(pos_mid, color="orange", linestyle="--", linewidth=2)

        plt.title(title)
        plt.xlabel("Logit")
        plt.ylabel("Count")
        plt.legend()

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
