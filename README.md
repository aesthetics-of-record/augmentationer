# Augmentationer

Augmentationer는 이미지와 해당 라벨을 다양한 방식으로 증강하는 파이썬 라이브러리입니다.

## 주요 기능

- 이미지 회전
- 가우시안 노이즈 추가
- 블러 효과 적용
- 그레이스케일 변환
- 랜덤 크롭
- 투시 변환
- 대비 조정
- 소금-후추 노이즈 추가
- 이미지 좌우 반전
- 밝기 조정
- 색상 지터링
- 채널 셔플

## 설치 방법

```
pip install augmentationer
```

## 사용 방법

### 기본 사용법

```python
from augmentationer import augmentationer
from augmentationer.function import *

# 증강 함수 정의
augmentation_functions = [
    lambda img, lbl: rotate_image_and_labels(img, lbl, 30, 0.3),
    lambda img, lbl: (add_gaussian_noise(img)),
    lambda img, lbl: (apply_blur(img)),
    lambda img, lbl: (convert_color(img)),
    lambda img, lbl: apply_random_crop(img, lbl, 0.8),
    lambda img, lbl: apply_perspective_transform(img, lbl, 0.2),
    lambda img, lbl: (change_contrast(img)),
    lambda img, lbl: (add_salt_pepper_noise(img)),
    lambda img, lbl: (flip_image_and_labels(img, lbl)),
    lambda img, lbl: (adjust_brightness(img)),
    lambda img, lbl: (color_jitter(img)),
    lambda img, lbl: (channel_shuffle(img)),
]

# augmentationer 실행
augmentationer("images", "labels", "output", augmentation_functions, "aug_prefix")
```

### 멀티 사용법

여러 가지 증강 설정을 적용하려면 다음과 같이 사용할 수 있습니다:

```python
from augmentationer import augmentationer
from augmentationer.function import *

# 증강 함수 세트 정의
augmentation_functions_1 = [
    lambda img, lbl: rotate_image_and_labels(img, lbl, 30, 0.3),
    lambda img, lbl: (add_gaussian_noise(img)),
    lambda img, lbl: (apply_blur(img)),
    lambda img, lbl: (convert_color(img)),
    lambda img, lbl: apply_random_crop(img, lbl, 0.8),
    lambda img, lbl: apply_perspective_transform(img, lbl, 0.2),
]

augmentation_functions_2 = [
    lambda img, lbl: (change_contrast(img)),
    lambda img, lbl: (add_salt_pepper_noise(img)),
    lambda img, lbl: (flip_image_and_labels(img, lbl)),
    lambda img, lbl: (adjust_brightness(img)),
    lambda img, lbl: (color_jitter(img)),
    lambda img, lbl: (channel_shuffle(img)),
]

augmentation_functions_3 = [
    lambda img, lbl: (add_gaussian_noise(img)),
    lambda img, lbl: (apply_blur(img)),
    lambda img, lbl: (convert_color(img)),
    lambda img, lbl: apply_random_crop(img, lbl, 0.8),
    lambda img, lbl: apply_perspective_transform(img, lbl, 0.2),
]

# 여러 증강 설정 적용
augmentationer("images", "labels", "output", augmentation_functions_1, "augmentation_1")
augmentationer("images", "labels", "output", augmentation_functions_2, "augmentation_2")
augmentationer("images", "labels", "output", augmentation_functions_3, "augmentation_3")
```

이 방법을 사용하면 여러 가지 증강 설정을 독립적으로 적용할 수 있으며, 각 설정에 대해 접두사가 붙은 이름의 파일이 생성됩니다.

### Visualizer 사용법

증강된 이미지와 라벨을 시각화하려면 다음과 같이 `visualizer` 함수를 사용할 수 있습니다:

```python
from augmentationer import visualizer

# 이미지와 라벨 파일 경로 지정
image_path = 'output/images/image2_augmentation_3.jpg'
label_path = 'output/labels/image2_augmentation_3.txt'

# visualizer 실행 (class_colors 지정은 선택. 선택 안할 시 랜덤으로 컬러 지정됨.)
visualizer(image_path, label_path, target_class_ids=[0,1], text=False, class_colors={0: (255, 0, 0), 1: (0, 255, 0)})

# 클래스 지정없이 시각화 (기본 값 : 랜덤 컬러 지정, 모든 클래스 표시, 텍스트 표시)
visualizer(image_path, label_path)
```

이 함수는 지정된 이미지를 열고, 해당하는 라벨 정보를 이미지 위에 표시합니다. 각 객체는 녹색 폴리곤으로 표시되며, 클래스 ID도 함께 표시됩니다.

## 주요 함수 설명

### augmentationer(image_folder, txt_folder, output_folder, augmentation_functions, aug_prefix)

주어진 이미지 폴더와 라벨 폴더에서 이미지를 읽어와 다양한 증강 함수를 적용한 후, 증강된 이미지와 라벨을 출력 폴더에 저장합니다.

매개변수:

- `image_folder` (str 또는 PathLike): 원본 이미지 파일들이 저장된 폴더 경로
- `txt_folder` (str 또는 PathLike): 원본 라벨 파일들이 저장된 폴더 경로
- `output_folder` (str 또는 PathLike): 증강된 이미지와 라벨 파일들을 저장할 출력 폴더 경로
- `augmentation_functions` (List[Callable]): 이미지와 라벨에 적용할 증강 함수들의 리스트
- `aug_prefix` (str): 증강된 파일 이름에 추가할 접두사 (예: "aug_1", "aug_2" 등, 증강된 파일 구분을 위해 사용)

### rotate_image_and_labels(image, labels, angle, padding_ratio=0.1)

이미지를 회전시키고 라벨을 조정합니다.

### add_gaussian_noise(image, mean=0, std=25, weight=0.25)

이미지에 가우시안 노이즈를 추가합니다.

### apply_blur(image, kernel_size=(5, 5), sigma=0)

이미지에 블러 효과를 적용합니다.

### convert_color(image)

이미지를 그레이스케일로 변환한 후 다시 컬러로 변환합니다.

### apply_random_crop(image, labels, crop_ratio=0.8)

이미지를 랜덤하게 자르고 라벨을 조정합니다.

### apply_perspective_transform(image, labels, strength=0.05)

이미지에 투시 변환을 적용하고 라벨을 조정합니다.

### change_contrast(image, alpha=1.5, beta=0)

이미지의 대비를 조정합니다.

### add_salt_pepper_noise(image, salt_vs_pepper=0.2, amount=0.004)

이미지에 소금과 후추 노이즈를 추가합니다.

### flip_image_and_labels(image, labels)

이미지를 수평으로 뒤집고 라벨을 조정합니다.

### adjust_brightness(image, brightness_range=(0.7, 1.3))

이미지의 밝기를 조정합니다.

### color_jitter(image, hue_shift=0.1, saturation_shift=0.3, value_shift=0.3)

이미지의 색조, 채도, 명도를 무작위로 변경합니다.

### channel_shuffle(image)

이미지의 색상 채널을 무작위로 섞습니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다.
