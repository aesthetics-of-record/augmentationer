import os
import cv2
import numpy as np
import random
from glob import glob
from pathlib import Path
from typing import Callable, List, Union
from os import PathLike

def adjust_brightness(image, brightness_range=(0.7, 1.3)):
    """
    이미지의 밝기를 조정합니다.
    
    Parameters:
    image (numpy.ndarray): 밝기를 조정할 이미지
    brightness_range (tuple): 밝기 조정 범위 (최소값, 최대값) (기본값: (0.7, 1.3))
    
    Returns:
    numpy.ndarray: 밝기가 조정된 이미지
    """
    brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
    return cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

def add_gaussian_noise(image, mean=0, std=25, weight=0.25):
    """
    이미지에 가우시안 노이즈를 추가합니다.
    
    Parameters:
    image (numpy.ndarray): 노이즈를 추가할 이미지
    mean (float): 가우시안 분포의 평균값 (기본값: 0)
    std (float): 가우시안 분포의 표준편차 (기본값: 25)
    weight (float): 노이즈의 가중치 (기본값: 0.25)
    
    Returns:
    numpy.ndarray: 가우시안 노이즈가 추가된 이미지
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    return cv2.addWeighted(image, 1 - weight, noise, weight, 0)

def apply_blur(image, kernel_size=(5, 5), sigma=0):
    """
    이미지에 블러 효과를 적용합니다.
    
    Parameters:
    image (numpy.ndarray): 블러 효과를 적용할 이미지
    kernel_size (tuple): 가우시안 커널의 크기 (기본값: (5, 5))
    sigma (float): 가우시안 커널의 X축 표준편차. 0이면 자동으로 계산됩니다. (기본값: 0)
    
    Returns:
    numpy.ndarray: 블러 효과가 적용된 이미지
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)

def convert_color(image):
    """
    이미지를 그레이스케일로 변환한 후 다시 컬러로 변환합니다.
    
    Parameters:
    image (numpy.ndarray): 색상을 변환할 이미지
    
    Returns:
    numpy.ndarray: 그레이스케일로 변환된 후 다시 컬러로 변환된 이미지
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def pad_image(image, padding):
    """
    이미지에 패딩을 추가합니다.
    
    Parameters:
    image (numpy.ndarray): 패딩을 추가할 이미지
    padding (int): 추가할 패딩의 크기
    
    Returns:
    numpy.ndarray: 패딩이 추가된 이미지
    """
    return cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def rotate_image_and_labels(image, labels, angle, padding_ratio=0.1):
    """
    이미지를 회전시키고 라벨을 조정합니다.
    
    Parameters:
    image (numpy.ndarray): 회전시킬 이미지
    labels (list of str): 이미지에 대한 라벨 리스트
    angle (float): 회전 각도
    padding_ratio (float): 이미지에 추가할 패딩의 비율 (기본값: 0.1)
    
    Returns:
    tuple: 회전된 이미지와 조정된 라벨 리스트
    """
    # 이미지 패딩
    padding = int(max(image.shape) * padding_ratio)  # 패딩 비율 적용
    padded_image = pad_image(image, padding)
    
    # 패딩된 이미지 회전
    (h, w) = padded_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(padded_image, M, (w, h))

    # 라벨 회전 및 조정
    rotated_labels = []
    for label in labels:
        parts = label.strip().split()
        class_id = parts[0]
        coords = np.array(parts[1:9], dtype=np.float32).reshape(-1, 2)
        
        # 좌표에 패딩 추가 및 스케일 조정
        coords[:, 0] = coords[:, 0] * image.shape[1] + padding
        coords[:, 1] = coords[:, 1] * image.shape[0] + padding
        
        # 회전 변환 적용
        ones = np.ones(shape=(len(coords), 1))
        coords_homogeneous = np.hstack([coords, ones])
        rotated_coords = M.dot(coords_homogeneous.T).T
        
        # 회전된 좌표를 0-1 범위로 정규화
        rotated_coords[:, 0] /= w
        rotated_coords[:, 1] /= h
        
        # 라벨 문자열 생성 (class x1 y1 x2 y2 x3 y3 x4 y4 x1 y1 형식)
        rotated_coords = rotated_coords.flatten()
        rotated_label = f"{class_id} " + " ".join(map(lambda x: f"{x:.6f}", rotated_coords)) + f" {rotated_coords[0]:.6f} {rotated_coords[1]:.6f}\n"
        rotated_labels.append(rotated_label)

    return rotated_image, rotated_labels

def flip_image_and_labels(image, labels):
    """
    이미지를 수평으로 뒤집고 라벨을 조정합니다.
    
    Parameters:
    image (numpy.ndarray): 뒤집을 이미지
    labels (list of str): 이미지에 대한 라벨 리스트
    
    Returns:
    tuple: 뒤집힌 이미지와 조정된 라벨 리스트
    """
    flipped_image = cv2.flip(image, 1)  # 수평 뒤집기
    flipped_labels = []
    
    for label in labels:
        parts = label.strip().split()
        class_id = parts[0]
        coords = np.array(parts[1:9], dtype=np.float32).reshape(-1, 2)
        
        # x 좌표 뒤집기
        coords[:, 0] = 1 - coords[:, 0]
        
        # 좌표 순서 변경 (시계 방향 유지)
        coords = coords[[1, 0, 3, 2], :]
        
        flipped_coords = coords.flatten()
        flipped_label = f"{class_id} " + " ".join(map(lambda x: f"{x:.6f}", flipped_coords)) + f" {flipped_coords[0]:.6f} {flipped_coords[1]:.6f}\n"
        flipped_labels.append(flipped_label)
    
    return flipped_image, flipped_labels

def change_contrast(image, alpha=1.5, beta=0):
    """
    이미지의 대비를 조정합니다.
    
    Parameters:
    image (numpy.ndarray): 대비를 조정할 이미지
    alpha (float): 대비 조정 계수 (기본값: 1.5)
    beta (int): 밝기 조정 계수 (기본값: 0)
    
    Returns:
    numpy.ndarray: 대비가 조정된 이미지
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_salt_pepper_noise(image, salt_vs_pepper=0.2, amount=0.004):
    """
    이미지에 소금과 후추 노이즈를 추가합니다.
    
    Parameters:
    image (numpy.ndarray): 노이즈를 추가할 이미지
    salt_vs_pepper (float): 소금과 후추 노이즈의 비율 (기본값: 0.2)
    amount (float): 노이즈의 양 (기본값: 0.004)
    
    Returns:
    numpy.ndarray: 소금과 후추 노이즈가 추가된 이미지
    """
    noisy_image = np.copy(image)
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))
    
    # 소금 노이즈 추가
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 255

    # 후추 노이즈 추가
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 0
    
    return noisy_image

def apply_random_crop(image, labels, crop_ratio=0.8):
    """
    이미지를 랜덤하게 자르고 라벨을 조정합니다.
    
    Parameters:
    image (numpy.ndarray): 자를 이미지
    labels (list of str): 이미지에 대한 라벨 리스트
    crop_ratio (float): 자를 비율 (기본값: 0.8)
    
    Returns:
    tuple: 자른 이미지와 조정된 라벨 리스트
    """
    h, w = image.shape[:2]
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    
    top = np.random.randint(0, h - crop_h)
    left = np.random.randint(0, w - crop_w)
    
    cropped_image = image[top:top+crop_h, left:left+crop_w]
    
    cropped_labels = []
    for label in labels:
        parts = label.strip().split()
        class_id = parts[0]
        coords = np.array(parts[1:9], dtype=np.float32).reshape(-1, 2)
        
        # 좌표 조정
        coords[:, 0] = (coords[:, 0] * w - left) / crop_w
        coords[:, 1] = (coords[:, 1] * h - top) / crop_h
        
        # 범위 내 좌표만 유지
        if np.all((coords >= 0) & (coords <= 1)):
            cropped_coords = coords.flatten()
            cropped_label = f"{class_id} " + " ".join(map(lambda x: f"{x:.6f}", cropped_coords)) + f" {cropped_coords[0]:.6f} {cropped_coords[1]:.6f}\n"
            cropped_labels.append(cropped_label)
    
    return cropped_image, cropped_labels

def apply_perspective_transform(image, labels, strength=0.05):
    """
    이미지에 투시 변환을 적용하고 라벨을 조정합니다.
    
    Parameters:
    image (numpy.ndarray): 투시 변환을 적용할 이미지
    labels (list of str): 이미지에 대한 라벨 리스트
    strength (float): 투시 변환의 강도 (기본값: 0.05)
    
    Returns:
    tuple: 투시 변환이 적용된 이미지와 조정된 라벨 리스트
    """
    h, w = image.shape[:2]
    
    # 원본 이미지의 모서리 좌표
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # 변형된 이미지의 모서리 좌표
    dst_pts = np.float32([
        [np.random.uniform(0, strength*w), np.random.uniform(0, strength*h)],
        [np.random.uniform((1-strength)*w, w), np.random.uniform(0, strength*h)],
        [np.random.uniform((1-strength)*w, w), np.random.uniform((1-strength)*h, h)],
        [np.random.uniform(0, strength*w), np.random.uniform((1-strength)*h, h)]
    ])
    
    # 투시 변환 행렬 계산
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # 이미지에 투시 변환 적용
    transformed_image = cv2.warpPerspective(image, M, (w, h))
    
    # 라벨 변환
    transformed_labels = []
    for label in labels:
        parts = label.strip().split()
        class_id = parts[0]
        coords = np.array(parts[1:9], dtype=np.float32).reshape(-1, 2)
        
        # 좌표에 투시 변환 적용
        coords_homogeneous = np.column_stack((coords * [w, h], np.ones((4, 1))))
        transformed_coords = M.dot(coords_homogeneous.T).T
        transformed_coords /= transformed_coords[:, 2:]
        transformed_coords = transformed_coords[:, :2] / [w, h]
        
        transformed_coords = transformed_coords.flatten()
        transformed_label = f"{class_id} " + " ".join(map(lambda x: f"{x:.6f}", transformed_coords)) + f" {transformed_coords[0]:.6f} {transformed_coords[1]:.6f}\n"
        transformed_labels.append(transformed_label)
    
    return transformed_image, transformed_labels

def color_jitter(image, hue_shift=0.1, saturation_shift=0.3, value_shift=0.3):
    """
    이미지의 색조, 채도, 명도를 무작위로 변경합니다.
    
    Parameters:
    image (numpy.ndarray): 색상을 변경할 이미지
    hue_shift (float): 색조 변경 범위 (0-1)
    saturation_shift (float): 채도 변경 범위 (0-1)
    value_shift (float): 명도 변경 범위 (0-1)
    
    Returns:
    numpy.ndarray: 색상이 변경된 이미지
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    h, s, v = cv2.split(hsv)
    h += np.random.uniform(-hue_shift, hue_shift) * 180
    s *= np.random.uniform(1 - saturation_shift, 1 + saturation_shift)
    v *= np.random.uniform(1 - value_shift, 1 + value_shift)
    
    h = np.clip(h, 0, 180)
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)
    
    hsv = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def channel_shuffle(image):
    """
    이미지의 색상 채널을 무작위로 섞습니다.
    
    Parameters:
    image (numpy.ndarray): 채널을 섞을 이미지
    
    Returns:
    numpy.ndarray: 채널이 섞인 이미지
    """
    channels = list(cv2.split(image))
    np.random.shuffle(channels)
    return cv2.merge(channels)

def apply_augmentations(image, labels, augmentation_functions):
    aug_image = image.copy()
    aug_labels = labels.copy()
    for aug_func in augmentation_functions:
        result = aug_func(aug_image, aug_labels)
        if isinstance(result, tuple):
            aug_image, aug_labels = result
        else:
            aug_image = result
    return aug_image, aug_labels


def augmentationer(image_folder: Union[str, PathLike], txt_folder: Union[str, PathLike], output_folder: Union[str, PathLike], augmentation_functions: List[Callable], aug_prefix: str):
    """
    주어진 이미지 폴더와 라벨 폴더에서 이미지를 읽어와서 다양한 증강 함수를 적용한 후, 
    증강된 이미지와 라벨을 출력 폴더에 저장하는 함수입니다.

    Parameters:
    image_folder (Union[str, PathLike]): 원본 이미지 파일들이 저장된 폴더 경로
    txt_folder (Union[str, PathLike]): 원본 라벨 파일들이 저장된 폴더 경로
    output_folder (Union[str, PathLike]): 증강된 이미지와 라벨 파일들을 저장할 출력 폴더 경로
    augmentation_functions (List[Callable]): 이미지와 라벨에 적용할 증강 함수들의 리스트
    aug_prefix (str): 증강된 파일 이름에 추가할 접두사

    Returns:
    None
    """
    # 이미지 폴더에서 모든 jpg 및 png 파일 경로를 가져옵니다.
    image_paths = glob(os.path.join(image_folder, "*.jpg")) + glob(os.path.join(image_folder, "*.png"))
    
    # 출력 폴더 내에 images와 labels 폴더를 생성합니다.
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels"), exist_ok=True)

    for image_path in image_paths:
        filename = Path(image_path).stem
        txt_path = os.path.join(txt_folder, f"{filename}.txt")

        # 원본 이미지 및 라벨 읽기
        image = cv2.imread(image_path)
        with open(txt_path, 'r') as file:
            labels = file.readlines()

        # 모든 증강 함수를 순차적으로 적용
        aug_image, aug_labels = apply_augmentations(image, labels, augmentation_functions)
        
        # 증강된 이미지와 라벨 저장
        _save_augmented_data(output_folder, filename, f"_{aug_prefix}", aug_image, aug_labels)

def _save_augmented_data(output_folder, filename, suffix, image, labels):
    image_path = os.path.join(output_folder, "images", f"{filename}{suffix}.jpg")
    label_path = os.path.join(output_folder, "labels", f"{filename}{suffix}.txt")
    cv2.imwrite(image_path, image)
    with open(label_path, 'w') as file:
        file.writelines(labels)

if __name__ == "__main__":
    # 예시 사용법
    image_folder_path = "images"
    txt_folder_path = "labels"
    output_folder_path = "output"

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



    augmentationer(image_folder_path, txt_folder_path, output_folder_path, augmentation_functions_1, "augmentation_1")
    augmentationer(image_folder_path, txt_folder_path, output_folder_path, augmentation_functions_2, "augmentation_2")
    augmentationer(image_folder_path, txt_folder_path, output_folder_path, augmentation_functions_3, "augmentation_3")

