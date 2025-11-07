# main.py
import os
import importlib
import data_loader
import engine
import copy
import torch

def discover_experiments() -> list:
    """'models' 폴더를 스캔하여 유효한 실험 폴더 목록을 반환합니다."""
    models_dir = "models"
    experiment_names = []
    for name in os.listdir(models_dir):
        path = os.path.join(models_dir, name)
        # '__'로 시작하지 않는 '폴더'를 실험으로 간주
        if os.path.isdir(path) and not name.startswith("__"):
            experiment_names.append(name)
    print(f"발견된 실험: {experiment_names}")
    return experiment_names

def run_experiment(exp_name: str):
    """
    하나의 실험(모델)에 대한 전체 학습 및 평가를 실행합니다.
    """
    print(f"\n========================================")
    print(f"  [실험 시작] 모델: {exp_name}")
    print(f"========================================")
    
    # 1. 동적으로 해당 모델의 config와 model 모듈을 임포트
    config_module = importlib.import_module(f"models.{exp_name}.config")
    model_module = importlib.import_module(f"models.{exp_name}.model")
    
    # 2. config 객체 생성 (ModelConfig 클래스 사용)
    config = config_module.ModelConfig()

    # 3. 가중치 저장 디렉토리 생성
    save_dir = os.path.join(config.CHECKPOINT_DIR, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"체크포인트 저장 경로: {save_dir}")

    # 4. 데이터 로더 생성 (해당 config 사용)
    # data_loader.py는 어떤 config가 오든 알아서 처리
    print("Loading data with model-specific transforms...")
    transform = data_loader.get_transforms(config)
    train_loader = data_loader.get_train_loader(config, transform)
    test_loader = data_loader.get_test_loader(config, transform)

    # 5. 모델 생성 (해당 config 사용)
    # (모든 model.py는 'Model' 클래스를 가진다고 가정)
    print(f"Building model: {config.MODEL_NAME}...")
    model = model_module.Model(config).to(config.DEVICE)


    if config.PRETRAINED_WEIGHTS:
        if os.path.exists(config.PRETRAINED_WEIGHTS):
            print(f"사전 학습된 가중치 로드: {config.PRETRAINED_WEIGHTS}")
            
            # map_location을 사용하여 GPU/CPU 호환성 보장
            state_dict = torch.load(config.PRETRAINED_WEIGHTS, map_location=config.DEVICE)
            
            # .pth 파일이 딕셔너리 형태('model_state_dict')일 경우 처리
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # strict=False: 전이 학습 시 일부 레이어(예: classifier)가
            # 달라도 나머지 일치하는 레이어는 모두 로드합니다.
            model.load_state_dict(state_dict, strict=False)
            print("가중치 로드가 완료되었습니다. (strict=False)")
        else:
            print(f"경고: {config.PRETRAINED_WEIGHTS} 파일을 찾을 수 없습니다. 학습을 새로 시작합니다.")

    
    # 6. 엔진 실행 (학습 및 평가)
    print(f"Starting training on {config.DEVICE}...")
    history = []
    best_test_acc = 0.0
    final_epoch_results = {}

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = engine.train_one_epoch(
            model, train_loader, config, epoch)
        
        test_loss, test_acc = engine.evaluate(
            model, test_loader, config)
        
        history.append({ "epoch": epoch, "test_acc": test_acc })
        final_epoch_results = {'epoch': epoch, 'test_loss': test_loss, 'test_acc': test_acc}

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            
    
    # 루프가 끝난 시점의 모델을 모델 고유 이름으로 저장합니다.
    # 예: ./checkpoints/cnn_discriminator/cnn_discriminator_final_epoch.pth
    final_model_name = f"{exp_name}_final_epoch.pth"
    final_model_path = os.path.join(save_dir, final_model_name)
    print(f"Training finished. Saving final model from epoch {config.NUM_EPOCHS} to {final_model_path}...")
    torch.save({
        'epoch': config.NUM_EPOCHS,
        'model_state_dict': model.state_dict(), # 이 시점의 모델 가중치
        'test_acc': final_epoch_results.get('test_acc', 0.0)
    }, final_model_path)
    
    print(f"--- [실험 종료] {exp_name} ---")
    
    # 최종 결과 반환
    return {
        "model_name": config.MODEL_NAME,
        "best_test_acc": best_test_acc,
        "final_test_loss": test_loss # 마지막 에포크의 손실
    }

def main():
    """모든 실험을 실행하고 결과를 요약합니다."""
    
    experiment_names = discover_experiments()
    all_results = []

    if not experiment_names:
        print("실행할 실험을 'models' 폴더에서 찾지 못했습니다.")
        return

    # 모든 실험을 순차적으로 실행
    for exp_name in experiment_names:
        try:
            result = run_experiment(exp_name)
            all_results.append(result)
        except Exception as e:
            print(f"!!! [{exp_name}] 실험 중 오류 발생: {e}")
            all_results.append({
                "model_name": exp_name,
                "best_test_acc": 0.0,
                "error": str(e)
            })

    # 6. 최종 비교
    print("\n========================================")
    print("          [최종 실험 결과 요약]")
    print("========================================")
    
    if not all_results:
        print("실행된 실험 결과가 없습니다.")
        return

    # 정확도(acc) 기준으로 내림차순 정렬
    all_results.sort(key=lambda x: x['best_test_acc'], reverse=True)
    
    for i, res in enumerate(all_results):
        rank = f"{i+1}위" if i == 0 else f"   {i+1}위"
        print(f"{rank} | 모델: {res['model_name']:25} | 최고 정확도: {res['best_test_acc']:.4f}")

    print("========================================")

if __name__ == "__main__":
    main()