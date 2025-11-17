"""
피스타치오 이미지 분류 모델 학습 스크립트

두 종류의 피스타치오(Kirmizi, Siirt)를 분류하는 CNN 모델을 학습합니다.
- 데이터 증강(augmentation)을 통한 일반화 성능 향상
- 7:3 비율 train/validation 분할
- 하이퍼파라미터 탐색 및 조기 종료 지원
"""
import argparse
import random
from pathlib import Path
import math
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def set_seed(seed: int):
    """재현성을 위한 랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(input_shape, num_classes, lr=1e-3, dropout=0.4, optimizer_name='adam'):
    """
    CNN 모델 생성 및 컴파일
    
    Args:
        input_shape: 입력 이미지 shape (height, width, channels)
        num_classes: 분류할 클래스 수
        lr: 학습률
        dropout: Dropout 비율
        optimizer_name: 옵티마이저 종류 ('adam' 또는 'sgd')
    
    Returns:
        컴파일된 Keras Sequential 모델
    
    Note:
        - GlobalAveragePooling2D 사용으로 파라미터 수 감소
        - BatchNorm, L2 규제는 소규모 배치 + 증강 환경에서 불안정하여 제외
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(dropout),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(dropout),
        Dense(num_classes, activation='softmax')
    ])

    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer_name == 'sgd':
        from tensorflow.keras.optimizers import SGD
        optimizer = SGD(learning_rate=lr, momentum=0.9)
    else:
        raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_name}")

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_history(history, out_dir: Path, tag: str = ""):
    """
    학습 곡선(accuracy, loss) 시각화 및 저장
    
    Args:
        history: model.fit()의 반환 History 객체
        out_dir: 저장할 디렉터리 경로
        tag: 파일명에 추가할 태그 (예: 'final')
    """
    plt.figure(figsize=(12, 4))

    # 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='train')
    plt.plot(history.history.get('val_accuracy', []), label='val')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='train')
    plt.plot(history.history.get('val_loss', []), label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    suffix = f"_{tag}" if tag else ""
    fig_path = out_dir / f'training_curves{suffix}.png'
    plt.savefig(fig_path)
    plt.show()


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='피스타치오 이미지 분류 모델 학습')
    parser.add_argument('--img-size', type=int, nargs=2, default=(128, 128), help='입력 이미지 크기 (height width)')
    parser.add_argument('--batch-size', type=int, default=32, help='배치 크기')
    parser.add_argument('--search-epochs', type=int, default=8, help='하이퍼파라미터 탐색 시 trial당 에폭 수')
    parser.add_argument('--final-epochs', type=int, default=30, help='최종 학습 에폭 수')
    parser.add_argument('--lr', type=float, default=1e-3, help='학습률')
    parser.add_argument('--seed', type=int, default=123, help='랜덤 시드')
    parser.add_argument('--tune', action='store_true', help='하이퍼파라미터 그리드 탐색 활성화')
    parser.add_argument('--search-trials', type=int, default=6, help='탐색할 최대 조합 수')
    parser.add_argument('--min-val-acc', type=float, default=0.80, help='조기 종료 기준 검증 정확도')
    parser.add_argument('--data-dir', type=str, default=None, help='데이터셋 디렉터리 경로')
    parser.add_argument('--out-dir', type=str, default='outputs', help='결과 저장 디렉터리')
    parser.add_argument('--no-class-weights', action='store_true', help='클래스 가중치 사용 안 함')
    parser.add_argument('--verbose-summary', action='store_true', help='각 trial마다 model.summary() 출력')
    return parser.parse_args()


def make_datagens(validation_split=0.3):
    """
    학습/검증용 ImageDataGenerator 생성
    
    Args:
        validation_split: 검증 데이터 비율 (기본값 0.3)
    
    Returns:
        (train_datagen, test_datagen) 튜플
    
    Note:
        - 학습 데이터는 증강(rotation, shift, shear, zoom, flip) 적용
        - 검증 데이터는 rescale만 적용
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=validation_split)
    return train_datagen, test_datagen


def make_generators(train_datagen, test_datagen, data_dir: Path, img_size, batch_size, seed):
    """
    디렉터리에서 학습/검증 제너레이터 생성
    
    Args:
        train_datagen: 학습용 ImageDataGenerator
        test_datagen: 검증용 ImageDataGenerator
        data_dir: 데이터셋 루트 디렉터리
        img_size: 목표 이미지 크기 (height, width)
        batch_size: 배치 크기
        seed: 랜덤 시드
    
    Returns:
        (train_gen, val_gen) 튜플
    """
    train_gen = train_datagen.flow_from_directory(
        directory=str(data_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=seed,
    )
    val_gen = test_datagen.flow_from_directory(
        directory=str(data_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=seed,
    )
    return train_gen, val_gen


def compute_class_weights(train_generator):
    """
    클래스 불균형 보정을 위한 가중치 계산
    
    Args:
        train_generator: 학습 데이터 제너레이터
    
    Returns:
        {클래스_인덱스: 가중치} 딕셔너리, 실패 시 None
    """
    try:
        classes = train_generator.classes
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(classes), y=classes)
        return {i: float(w) for i, w in enumerate(cw)}
    except Exception:
        return None


def run_trial(cfg: Dict[str, Any], shared: Dict[str, Any]):
    """
    단일 하이퍼파라미터 조합으로 학습 실행
    
    Args:
        cfg: 하이퍼파라미터 설정 (lr, batch, dropout, optimizer, epochs)
        shared: 공통 설정 (datagen, data_dir, IMG_SIZE, seed 등)
    
    Returns:
        학습 결과 딕셔너리 (config, val_acc, val_loss, history, model 등)
    """
    train_gen, val_gen = make_generators(
        shared['train_datagen'],
        shared['test_datagen'],
        shared['data_dir'],
        shared['IMG_SIZE'],
        cfg['batch'],
        shared['seed'],
    )
    num_classes = len(train_gen.class_indices)
    input_shape = (shared['IMG_SIZE'][0], shared['IMG_SIZE'][1], 3)

    model = build_model(input_shape, num_classes,
                        lr=cfg['lr'],
                        dropout=cfg['dropout'],
                        optimizer_name=cfg['optimizer'])

    if shared['verbose_summary']:
        model.summary()

    steps_per_epoch = max(1, math.ceil(train_gen.samples / cfg['batch']))
    val_steps = max(1, math.ceil(val_gen.samples / cfg['batch']))

    class_weights = None if shared['no_class_weights'] else compute_class_weights(train_gen)
    if class_weights:
        print(f"클래스 가중치 사용: {class_weights}")

    checkpoint_path = shared['out_dir'] / (
        f"best_trial_lr{cfg['lr']}_bs{cfg['batch']}_do{cfg['dropout']}_{cfg['optimizer']}.keras"
    )
    callbacks = [
        ModelCheckpoint(str(checkpoint_path), monitor='val_accuracy', save_best_only=True, verbose=0),
        EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0),
    ]
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=cfg['epochs'],
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    loss, acc = model.evaluate(val_gen, steps=val_steps, verbose=0)
    return {
        'config': cfg,
        'val_acc': acc,
        'val_loss': loss,
        'history': history,
        'model': model,
        'train_gen': train_gen,
        'val_gen': val_gen,
        'checkpoint': checkpoint_path,
    }


def final_train(best: Dict[str, Any], shared: Dict[str, Any]):
    """
    최적 하이퍼파라미터로 확장 학습 수행
    
    Args:
        best: run_trial 결과 중 최고 성능 trial 정보
        shared: 공통 설정
    
    Returns:
        최종 학습 결과 딕셔너리 (model, history, val_acc, val_loss, val_gen)
    """
    cfg = best['config']
    batch = cfg['batch']
    train_gen, val_gen = make_generators(
        shared['train_datagen'],
        shared['test_datagen'],
        shared['data_dir'],
        shared['IMG_SIZE'],
        batch,
        shared['seed'],
    )
    num_classes = len(train_gen.class_indices)
    input_shape = (shared['IMG_SIZE'][0], shared['IMG_SIZE'][1], 3)
    model = build_model(input_shape, num_classes,
                        lr=cfg['lr'],
                        dropout=cfg['dropout'],
                        optimizer_name=cfg['optimizer'])
    steps_per_epoch = max(1, math.ceil(train_gen.samples / batch))
    val_steps = max(1, math.ceil(val_gen.samples / batch))
    class_weights = None if shared['no_class_weights'] else compute_class_weights(train_gen)
    callbacks = [
        ModelCheckpoint(str(shared['out_dir'] / 'final_best.keras'), monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ]
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=shared['final_epochs'],
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    loss, acc = model.evaluate(val_gen, steps=val_steps, verbose=1)
    print(f"최종 확장 학습 - 검증 정확도: {acc:.4f}")
    return {'model': model, 'history': history, 'val_acc': acc, 'val_loss': loss, 'val_gen': val_gen}


def main():
    """메인 실행 함수"""
    args = parse_args()
    IMG_SIZE = tuple(args.img_size)
    RANDOM_STATE = args.seed
    set_seed(RANDOM_STATE)

    # 경로 설정
    project_root = Path(__file__).resolve().parents[2]
    data_dir = Path(args.data_dir) if args.data_dir else project_root / 'data' / 'Pistachio_Image_Dataset'
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        raise FileNotFoundError(f"데이터 디렉터리를 찾을 수 없습니다: {data_dir}")

    train_datagen, test_datagen = make_datagens(validation_split=0.3)

    shared = {
        'train_datagen': train_datagen,
        'test_datagen': test_datagen,
        'data_dir': data_dir,
        'out_dir': out_dir,
        'IMG_SIZE': IMG_SIZE,
        'seed': RANDOM_STATE,
        'no_class_weights': args.no_class_weights,
        'verbose_summary': args.verbose_summary,
        'final_epochs': args.final_epochs,
    }

    # 하이퍼파라미터 그리드 생성 및 랜덤 샘플링
    if args.tune:
        full_grid: List[Dict[str, Any]] = []
        for lr in [1e-3, 5e-4, 1e-4]:
            for batch in [32, 64]:
                for dropout in [0.3, 0.4, 0.5]:
                    for optimizer in ['adam', 'sgd']:
                        full_grid.append({
                            'lr': lr,
                            'batch': batch,
                            'dropout': dropout,
                            'optimizer': optimizer,
                            'epochs': args.search_epochs,
                        })
        random.shuffle(full_grid)
        search_grid = full_grid[: args.search_trials]
    else:
        search_grid = [{
            'lr': args.lr,
            'batch': args.batch_size,
            'dropout': 0.4,
            'optimizer': 'adam',
            'epochs': args.search_epochs,
        }]

    # 하이퍼파라미터 탐색
    best_trial = None
    for i, cfg in enumerate(search_grid, start=1):
        print(f"\n[Trial {i}/{len(search_grid)}] lr={cfg['lr']} batch={cfg['batch']} dropout={cfg['dropout']} opt={cfg['optimizer']} epochs={cfg['epochs']}")
        trial_res = run_trial(cfg, shared)
        print(f"Trial 검증 정확도={trial_res['val_acc']:.4f} 검증 손실={trial_res['val_loss']:.4f}")
        if best_trial is None or trial_res['val_acc'] > best_trial['val_acc']:
            best_trial = trial_res
        # 목표 정확도 도달 시 조기 종료
        if trial_res['val_acc'] >= args.min_val_acc:
            print(f"목표 정확도 {args.min_val_acc} 도달. 탐색을 조기 종료합니다.")
            break

    if best_trial is None:
        raise RuntimeError('성공한 trial이 없습니다.')

    print(f"\n최적 trial 설정: {best_trial['config']} 검증 정확도={best_trial['val_acc']:.4f}")

    # 최적 설정으로 확장 학습
    final_res = final_train(best_trial, shared)

    # 최종 평가
    best_batch = best_trial['config']['batch']
    val_gen = final_res['val_gen']
    val_steps = max(1, math.ceil(val_gen.samples / best_batch))
    print('\n== 최종 평가 ==')
    loss, acc = final_res['model'].evaluate(val_gen, steps=val_steps, verbose=1)
    print(f'최종 검증 손실: {loss:.4f}, 정확도: {acc:.4f}')

    # 예측 및 평가 지표 출력
    preds = final_res['model'].predict(val_gen, steps=val_steps, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes[: len(y_pred)]
    idx2class = {v: k for k, v in val_gen.class_indices.items()}
    print('\n== 분류 리포트 ==')
    print(classification_report(y_true, y_pred, target_names=[idx2class[i] for i in sorted(idx2class.keys())]))
    print('\n== 혼동 행렬 ==')
    print(confusion_matrix(y_true, y_pred))

    # 모델 저장 및 학습 곡선 출력
    final_path = out_dir / 'final_extended.keras'
    final_res['model'].save(final_path)
    plot_history(final_res['history'], out_dir, tag='final')
    print(f"최종 모델 저장 완료: {final_path}")


    # (Old evaluation block removed - replaced by final evaluation above)


if __name__ == '__main__':
    main()
