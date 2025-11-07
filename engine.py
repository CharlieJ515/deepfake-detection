# engine.py
import torch
import time
from torch.utils.data import DataLoader
from config_global import GlobalConfig
from typing import Tuple

def train_one_epoch(model: torch.nn.Module, 
                      dataloader: DataLoader, 
                      config: GlobalConfig, 
                      epoch: int) -> Tuple[float, float]:
    """
    모델의 한 에포크(epoch) 학습을 수행합니다.
    """
    model.train()  # 모델을 학습 모드로 설정
    
    running_loss = 0.0
    correct = 0
    total = 0
    step = 0
    start_time = time.time()
    device = config.DEVICE

    images: torch.Tensor
    labels: torch.Tensor
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)

        # 1. 모델에 정의된 step 함수로 학습 (forward + backward + optimize)
        logits, loss = model.step(images, labels)

        # 2. 정확도 계산 (원본 train.py 로직 및 버그 수정)
        preds = (torch.sigmoid(logits.view(-1)) > 0.5).long()
        correct += (preds == labels.view(-1)).sum().item()

        # 3. 통계 집계
        running_loss += loss * images.size(0)
        total += images.size(0)
        step += 1

        if step % config.LOG_INTERVAL == 0:
            avg_loss = running_loss / total
            acc = correct / total
            elapsed = time.time() - start_time
            print(f"[epoch {epoch}/{config.NUM_EPOCHS} | step {step}] "
                  f"loss={avg_loss:.4f} acc={acc:.4f} elapsed={elapsed:.1f}s")

    # 에포크 종료
    avg_loss = running_loss / total
    acc = correct / total
    print(f"Epoch {epoch} done: Train loss={avg_loss:.4f} Train acc={acc:.4f}")
    
    # main.py로 결과 반환
    return avg_loss, acc

def evaluate(model: torch.nn.Module, 
             dataloader: DataLoader, 
             config: GlobalConfig) -> Tuple[float, float]:
    """
    모델의 성능을 '평가'합니다. (가중치 업데이트 없음)
    """
    model.eval()  # 모델을 평가 모드로 설정
    
    running_loss = 0.0
    correct = 0
    total = 0
    device = config.DEVICE

    # 평가 시에는 그래디언트 계산을 비활성화
    with torch.no_grad():
        images: torch.Tensor
        labels: torch.Tensor
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)

            # 1. 평가: step() 대신 forward()만 호출
            logits = model(images)  # model.forward(images)
            
            # 2. 평가: criterion으로 손실 '계산'만 수행 (backward X)
            loss = model.criterion(logits, labels)

            # 3. 정확도 계산
            preds = (torch.sigmoid(logits.view(-1)) > 0.5).long()
            correct += (preds == labels.view(-1)).sum().item()

            # 4. 통계 집계
            running_loss += loss.item() * images.size(0)
            total += images.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    print(f"--- Evaluation ---")
    print(f"Test/Validation loss: {avg_loss:.4f}, Test/Validation acc: {acc:.4f}")
    print(f"------------------")
    
    # main.py로 결과 반환
    return avg_loss, acc