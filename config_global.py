from dataclasses import dataclass, field
from typing import List, Tuple
import torch

@dataclass
class GlobalConfig:
    """모든 실험이 공유하는 기본 설정"""
    
    # 1. 환경 설정
    DEVICE: torch.device = field(default_factory=lambda: 
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    

    # 2. 학습 파라미터
    NUM_EPOCHS: int = 2
    BATCH_SIZE: int = 64
    LOG_INTERVAL: int = 100

    # 3. 데이터셋 파라미터
    SHUFFLE_SIZE: int = 3000

    # 4. 데이터 경로 (Train/Test 분리)
    TRAIN_DIFFFACE_SHARDS: List[str] = field(default_factory=lambda: [
        "./data/diffusion_face/ADM-{000..004}.tar",
        "./data/diffusion_face/DDIM-{000..004}.tar",
        "./data/diffusion_face/DDPM-{000..004}.tar",
        "./data/diffusion_face/DiffSwap-{000..004}.tar",
        "./data/diffusion_face/Inpaint-{000..004}.tar",
        "./data/diffusion_face/LDM-{000..004}.tar",
        "./data/diffusion_face/PNDM-{000..004}.tar",
        "./data/diffusion_face/SDv15_DS0.3-{000..004}.tar",
        "./data/diffusion_face/SDv15_DS0.5-{000..004}.tar",
        "./data/diffusion_face/SDv15_DS0.7-{000..004}.tar",
        "./data/diffusion_face/SDv21_DS0.3-{000..004}.tar",
        "./data/diffusion_face/SDv21_DS0.5-{000..004}.tar",
        "./data/diffusion_face/SDv21_DS0.7-{000..004}.tar",
    ])
    TRAIN_CELEBA_SHARDS: List[str] = field(default_factory=lambda: [
        "./data/mm_celeba_hq/mm_celeba_hq-{000..004}.tar"
    ])
    
    TEST_DIFFFACE_SHARDS: List[str] = field(default_factory=lambda: [
        "./data/diffusion_face/ADM-005.tar",
        "./data/diffusion_face/DDIM-005.tar",
        "./data/diffusion_face/DDPM-005.tar",
        "./data/diffusion_face/DiffSwap-005.tar",
        "./data/diffusion_face/Inpaint-005.tar",
        "./data/diffusion_face/LDM-005.tar",
        "./data/diffusion_face/PNDM-005.tar",
        "./data/diffusion_face/SDv15_DS0.3-005.tar",
        "./data/diffusion_face/SDv15_DS0.5-005.tar",
        "./data/diffusion_face/SDv15_DS0.7-005.tar",
        "./data/diffusion_face/SDv21_DS0.3-005.tar",
        "./data/diffusion_face/SDv21_DS0.5-005.tar",
        "./data/diffusion_face/SDv21_DS0.7-005.tar",
    ])
    TEST_CELEBA_SHARDS: List[str] = field(default_factory=lambda: [
        "./data/mm_celeba_hq/mm_celeba_hq-005.tar"
    ])

    # data_loader.py가 필요로 하는 값들
    IMG_SIZE: int = 256  # (기본값을 설정하거나 None으로 설정)
    MEAN: List[float] = None
    STD: List[float] = None
    
    # 모델 고유의 값들 (engine.py는 이 값들이 존재한다고 가정하지 않음)
    MODEL_NAME: str = "BaseModel"
    LR: float = 0.001
    BETAS: Tuple[float, float] = (0.9, 0.999)