import timm
import torch
import torch.nn as nn
import torch.optim as optim
from .config import ModelConfig

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.features = timm.create_model(
            'tf_efficientnet_b7_ns',
            pretrained=True,
            in_chans=config.IN_CHANNELS,
            num_classes=1,
            drop_rate=config.DROP_RATE,
            drop_path_rate=config.DROP_PATH_RATE,
        )

        self.criterion = nn.BCEWithLogitsLoss()

        #self.unfreeze()
        self.build_optimizer()

    def freeze(self):
        for n, p in self.features.named_parameters():
            p.requires_grad = ('classifier' in n)
        for m in self.features.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()

    def unfreeze(self):
        for p in self.features.parameters():
            p.requires_grad = True

    def build_optimizer(self):
        head, body = [], []
        for n, p in self.features.named_parameters():
            if not p.requires_grad:
                continue
            (head if 'classifier' in n else body).append(p)
        self.optimizer = optim.AdamW([
            {'params': body, 'lr': self.config.LR},
            {'params': head, 'lr': self.config.LR * self.config.HEAD_LR_MUL},
        ], betas=self.config.BETAS, weight_decay=self.config.WEIGHT_DECAY,)

    def forward(self, x):
        return self.features(x)
        
    def step(self, images, labels):
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.forward(images).squeeze(-1)
        loss = self.criterion(logits, labels.float().view(-1))
        loss.backward()
        self.optimizer.step()
        return logits, loss.item()
        
    @torch.no_grad()
    def predict(self ,x, threshold: float = 0.5):
        probs = torch.sigmoid(self.forward(x).squeeze(-1))
        return (probs >= threshold).long()