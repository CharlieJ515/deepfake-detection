# models/xceptionnet/model.py
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from .config import ModelConfig
from torchvision.transforms import v2, InterpolationMode

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
    
    @staticmethod
    def get_transforms(config: ModelConfig, is_train: bool = True) -> v2.Compose:
        """
        XceptionNet을 위한 커스텀 Augmentation 파이프라인
        (RandomResizedCrop + JPEG)
        """
        transforms_list = []

        if is_train:
            transforms_list.extend([
                v2.RandomResizedCrop(
                    (config.IMG_SIZE, config.IMG_SIZE), # 299x299
                    scale=(0.85, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=InterpolationMode.BICUBIC
                ),
                v2.RandomApply([
                    v2.JPEG(quality=(40, 90))
                ], p=0.5),
                v2.RandomHorizontalFlip(p=0.5),
            ])
        else:
            transforms_list.append(
                 v2.Resize(
                    (config.IMG_SIZE, config.IMG_SIZE), 
                    interpolation=InterpolationMode.BICUBIC
                )
            )

        transforms_list.extend([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=config.MEAN, std=config.STD),
        ])
        
        return v2.Compose(transforms_list)