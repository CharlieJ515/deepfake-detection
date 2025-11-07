# models/xceptionnet/config.py
from dataclasses import dataclass, field
from config_global import GlobalConfig
from typing import List, Tuple

@dataclass
class ModelConfig(GlobalConfig): # <-- 상속
    """XceptionNet 모델을 위한 고유 설정"""
    
    # 이 모델의 고유 이름
    MODEL_NAME: str = "XceptionNet"
    
    # --- 모델 파라미터 (고유값) ---
    # XceptionNet은 299x299 입력을 표준으로 사용합니다.
    IMG_SIZE: int = 299 
    IN_CHANNELS: int = 3
    # 파인튜닝(fine-tuning)을 위한 학습률 (기본값보다 낮게 설정)
    LR: float = 0.0001
    BETAS: Tuple[float, float] = (0.9, 0.999) # 표준 Adam 베타

    # --- 정규화 (ImageNet 통계치) ---
    MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    
    CHECKPOINT_DIR: str = "./checkpoints"
    # main.py가 사용할 .pth 파일 경로
    # (이 값을 설정하면 torchvision 가중치 위에 덮어쓰게 됩니다)
    PRETRAINED_WEIGHTS: str = None