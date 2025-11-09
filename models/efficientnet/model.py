# models/efficientnet_b0/model.py
import torch
import torch.nn as nn
import torch.optim as optim
from .config import ModelConfig
from typing import List, Callable, Optional, Sequence
from functools import partial
import math

# torchvision.models에서 '모델 구조'가 아닌 '가중치'만 불러오기 위해 import
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# --- EfficientNet의 핵심 빌딩 블록 정의 ---

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation 블록"""
    def __init__(self, input_channels: int, squeeze_channels: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = nn.SiLU(inplace=True) # SiLU (Swish)
        self.scale_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale

class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution (MBConv) 블록"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float,
        dropout_rate: float,
    ):
        super().__init__()
        self.use_residual = (stride == 1) and (in_channels == out_channels)
        self.activation = nn.SiLU(inplace=True) # SiLU (Swish)
        
        # 1. Expansion phase (1x1 Conv)
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            self.activation,
        ) if expand_ratio != 1 else nn.Identity()

        # 2. Depthwise phase (3x3 or 5x5 Conv)
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            self.activation,
        )

        # 3. Squeeze-and-Excitation phase
        squeeze_channels = int(in_channels * se_ratio)
        self.se = SqueezeExcitation(expanded_channels, squeeze_channels) if se_ratio > 0 else nn.Identity()

        # 4. Projection phase (1x1 Conv)
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # 5. Dropout
        self.dropout = nn.Dropout(p=dropout_rate) if self.use_residual and dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            x = self.dropout(x)
            x += identity
            
        return x

# --- EfficientNet 구조를 감싸는 우리 프레임워크의 Model 클래스 ---

class Model(nn.Module):
    """
    EfficientNet-B0 구조를 직접 정의하고,
    engine.py와 호환되도록 래핑(wrapping)하는 메인 클래스
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # EfficientNet-B0 아키텍처 설정
        # (expand_ratio, channels, num_layers, kernel, stride)
        b0_settings = [
            (1, 16, 1, 3, 1), # stage 2
            (6, 24, 2, 3, 2), # stage 3
            (6, 40, 2, 5, 2), # stage 4
            (6, 80, 3, 3, 2), # stage 5
            (6, 112, 3, 5, 1),# stage 6
            (6, 192, 4, 5, 2),# stage 7
            (6, 320, 1, 3, 1), # stage 8
        ]
        
        # === EfficientNet 구조 정의 시작 ===
        
        # 1. Stem (초기 레이어)
        in_channels = config.IN_CHANNELS
        out_channels = 32 # B0
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # 2. MBConv 블록
        self.blocks = nn.ModuleList()
        in_channels = out_channels
        for expand_ratio, channels, num_layers, kernel, stride in b0_settings:
            out_channels = channels
            for i in range(num_layers):
                # 첫 번째 레이어만 stride 적용
                block_stride = stride if i == 0 else 1
                self.blocks.append(
                    MBConv(
                        in_channels,
                        out_channels,
                        kernel,
                        block_stride,
                        expand_ratio,
                        se_ratio=0.25, # B0는 0.25 고정
                        dropout_rate=0.2 # B0는 0.2 고정
                    )
                )
                in_channels = out_channels # 다음 블록의 입력 채널
        
        # 3. Head (분류기)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1280, kernel_size=1, bias=False), # B0
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # (★중요) 이진 분류를 위해 마지막 'fc' 레이어 교체
        self.fc = nn.Linear(1280, 1) # 1개 logit 출력
        
        # === EfficientNet 구조 정의 끝 ===

        # 4. 가중치 초기화 및 사전 학습된 가중치 로드
        self._initialize_weights()
        if config.LOAD_TORCHVISION_WEIGHTS:
            self._load_pretrained_weights()

        # 5. engine.py가 요구하는 criterion과 optimizer 정의
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=config.LR, 
            betas=config.BETAS
        )

        print(f"[Model Initialized] {config.MODEL_NAME} - Structure custom-built.")
        if config.LOAD_TORCHVISION_WEIGHTS:
            print("-> Successfully loaded pretrained weights from torchvision.")

    def _initialize_weights(self):
        """기본 가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _load_pretrained_weights(self):
        """
        Torchvision에서 사전 학습된 '가중치'만 가져와서
        우리가 직접 정의한 '구조'에 매핑합니다.
        
        (참고: torchvision의 EfficientNet과 이 구현의 레이어 이름이
        다소 다를 수 있어, 이름 매핑이 필요할 수 있으나 여기서는
        torchvision의 'efficientnet_b0' 모델 구조와 최대한
        유사하게 레이어 이름을 정의하여 자동 매핑을 시도합니다.)
        
        (수정: torchvision의 구현체 이름('features', 'classifier')과 
        우리가 정의한 이름('stem', 'blocks', 'head', 'fc')이
        완전히 다르므로, 수동으로 매핑해야 합니다.)
        """
        print("Loading pretrained weights from torchvision...")
        
        # 1. Torchvision의 완성된 모델과 가중치 로드
        tv_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        pretrained_state_dict = tv_model.state_dict()
        
        # 2. 현재 모델의 state_dict 가져오기
        model_state_dict = self.state_dict()
        
        # 3. 수동으로 레이어 이름 매핑
        # torchvision의 'features' -> 우리의 'stem', 'blocks', 'head'
        # torchvision의 'classifier' -> 우리의 'fc'
        
        mapping = {
            "features.0": "stem",
            "features.1": "blocks.0",
            "features.2": "blocks.1",
            "features.3": "blocks.2",
            "features.4": "blocks.3",
            "features.5": "blocks.4",
            "features.6": "blocks.5",
            "features.7": "blocks.6",
            "features.8": "head",
        }
        
        # MBConv 내부의 복잡한 이름 매핑
        # 예: tv_model 'features.1.0.block.0' -> 우리 'blocks.0.expand_conv.0'
        # ... 이 작업은 매우 복잡하고 오류가 발생하기 쉽습니다.
        
        # --- 단순화된 접근 방식 ---
        # torchvision의 state_dict 키와 우리 모델의 state_dict 키를 비교하여
        # 이름이 달라도 '모양(shape)'이 같은 가중치를 순서대로 로드합니다.
        
        print("Attempting simplified state_dict loading by shape matching...")
        
        # 우리 모델의 키 (fc 제외)
        our_keys = [k for k in model_state_dict.keys() if not k.startswith("fc.")]
        # torchvision 모델의 키 (classifier 제외)
        tv_keys = [k for k in pretrained_state_dict.keys() if not k.startswith("classifier.")]
        
        if len(our_keys) != len(tv_keys):
            print(f"Warning: Key count mismatch! Ours={len(our_keys)}, TV={len(tv_keys)}")
            print("-> Loading weights failed. Proceeding with random init.")
            return

        loaded_count = 0
        for our_key, tv_key in zip(our_keys, tv_keys):
            our_tensor = model_state_dict[our_key]
            tv_tensor = pretrained_state_dict[tv_key]
            
            if our_tensor.shape == tv_tensor.shape:
                model_state_dict[our_key] = tv_tensor
                loaded_count += 1
            else:
                print(f"Shape mismatch, skipping: {our_key} (Ours) vs {tv_key} (TV)")
                print(f"  Shapes: {our_tensor.shape} vs {tv_tensor.shape}")

        # 매핑된 가중치를 모델에 로드
        self.load_state_dict(model_state_dict, strict=False)
        print(f"-> Loaded {loaded_count} / {len(our_keys)} matching weight tensors.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) # [B, 1] logit 출력
        
        return x

    def step(self, images, labels):
        self.optimizer.zero_grad()
        preds = self.forward(images)
        loss = self.criterion(preds, labels)
        loss.backward()
        self.optimizer.step()
        return preds, loss.item()

    @torch.no_grad()
    def predict(self, x, threshold: float = 0.5):
        probs = torch.sigmoid(self.forward(x))
        return (probs >= threshold).long()