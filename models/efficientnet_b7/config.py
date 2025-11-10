from dataclasses import dataclass, field
from config_global import GlobalConfig
from typing import List, Tuple

@dataclass
class ModelConfig(GlobalConfig):
    #Efficientnet-B7 모델용 고유 설정

    MODEL_NAME: str = "EfficientNetB7"

    IMG_SIZE: int = 380
    IN_CHANNELS: int = 3

    LR: float = 2e-4
    BETAS: Tuple[float, float] = (0.9, 0.999)
    WEIGHT_DECAY: float = 1e-4
    HEAD_LR_MUL: float = 2.0

    DROP_RATE: float = 0.45
    DROP_PATH_RATE: float = 0.2

    MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD:  List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    CHECKPOINT_DIR: str = "./checkpoints"
    PRETRAINED_WEIGHTS: str = None