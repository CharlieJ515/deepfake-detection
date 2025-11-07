# models/xceptionnet/model.py
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from .config import ModelConfig

class Model(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.features = timm.create_model(
            'xception', 
            pretrained=True,
            in_chans=config.IN_CHANNELS
        )
        
        # 2. 이진 분류를 위해 최종 레이어 교체
        # 기존 1000개 출력(ImageNet) -> 1개 출력(딥페이크 로짓)
        num_ftrs = self.features.fc.in_features
        self.features.fc = nn.Linear(num_ftrs, 1)

        # 3. engine.py/main.py가 요구하는 속성 정의
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 파인튜닝을 위해 모든 파라미터를 옵티마이저에 전달
        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=config.LR, 
            betas=config.BETAS
        )

    def forward(self, x):
        """모델의 순전파 로직"""
        # (N, C, H, W) -> (N, 1)
        return self.features(x)

    def step(self, images, labels):
        """engine.py의 train_one_epoch에서 호출하는 함수"""
        self.optimizer.zero_grad()
        preds = self.forward(images)
        loss = self.criterion(preds, labels)
        loss.backward()
        self.optimizer.step()
        return preds, loss.item()

    @torch.no_grad()
    def predict(self, x, threshold: float = 0.5):
        """평가 시 사용 (현재 engine.py에서는 미사용)"""
        probs = torch.sigmoid(self.forward(x))
        return (probs >= threshold).long()