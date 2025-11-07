# data_loader.py
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch

from config_global import GlobalConfig
from datasets.diffface import DiffusionFaceDataset

def get_transforms(config: GlobalConfig) -> v2.Compose:
    """
    config에 정의된 값으로 이미지 변환(transform) 파이프라인을 생성합니다.
    """
    return v2.Compose([
        v2.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=config.MEAN, std=config.STD),
    ])

def get_train_loader(config: GlobalConfig, transform: v2.Compose) -> DataLoader:
    """
    '학습용' DataLoader를 생성합니다.
    - 학습용 경로 사용
    - 데이터 셔플(shuffle) 활성화
    """
    dataset = DiffusionFaceDataset(
        diffface_shards=config.TRAIN_DIFFFACE_SHARDS,
        celeba_shards=config.TRAIN_CELEBA_SHARDS,
        shuffle_size=config.SHUFFLE_SIZE,  # 학습 데이터는 섞어줍니다.
        batch_size=config.BATCH_SIZE,
        transform=transform,
    )
    # batch_sampler=None은 dataset이 이미 배치를 처리함(partial=True)을 의미
    return DataLoader(dataset, batch_size=None, batch_sampler=None, num_workers=0)

def get_test_loader(config: GlobalConfig, transform: v2.Compose) -> DataLoader:
    """
    '테스트용(검증용)' DataLoader를 생성합니다.
    - 테스트용 경로 사용
    - 데이터 셔플(shuffle) 비활성화
    """
    dataset = DiffusionFaceDataset(
        diffface_shards=config.TEST_DIFFFACE_SHARDS,
        celeba_shards=config.TEST_CELEBA_SHARDS,
        shuffle_size=0,  # 테스트 데이터는 섞지 않습니다. (일관된 평가)
        batch_size=config.BATCH_SIZE,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=None, batch_sampler=None, num_workers=0)