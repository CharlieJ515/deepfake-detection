import torch


def compute_macro_f1(preds: torch.Tensor, labels: torch.Tensor) -> float:
    preds = preds.view(-1).long()
    labels = labels.view(-1).long()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    f1_pos = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    f1_neg = (2 * tn) / (2 * tn + fp + fn + 1e-8)
    macro_f1 = (f1_pos + f1_neg) / 2
    return macro_f1


class MacroF1Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all internal counters."""
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    @torch.no_grad()
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """Accumulate counts for later F1 computation."""
        preds = preds.view(-1).long()
        labels = labels.view(-1).long()

        self.tp += ((preds == 1) & (labels == 1)).sum().item()
        self.tn += ((preds == 0) & (labels == 0)).sum().item()
        self.fp += ((preds == 1) & (labels == 0)).sum().item()
        self.fn += ((preds == 0) & (labels == 1)).sum().item()

    def compute(self) -> float:
        """Compute macro F1 from accumulated counts."""
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn
        f1_pos = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        f1_neg = (2 * tn) / (2 * tn + fp + fn + 1e-8)
        return (f1_pos + f1_neg) / 2

    @staticmethod
    @torch.no_grad()
    def compute_batch(preds: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute immediate (non-accumulated) F1 score for a single batch."""
        preds = preds.view(-1).long()
        labels = labels.view(-1).long()

        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        f1_pos = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        f1_neg = (2 * tn) / (2 * tn + fp + fn + 1e-8)
        return (f1_pos + f1_neg) / 2
