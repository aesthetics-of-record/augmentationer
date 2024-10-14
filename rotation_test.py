import cv2
import numpy as np

def rotate_and_pad_coordinates(coords, angle, image_size, new_size):
    # 중심점 계산
    center_x, center_y = image_size[1] // 2, image_size[0] // 2
    
    # 회전 행렬 생성
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    
    # 새로운 이미지 크기 계산
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((image_size[0] * sin) + (image_size[1] * cos))
    new_h = int((image_size[0] * cos) + (image_size[1] * sin))
    
    # 평행 이동 조정
    M[0, 2] += (new_w / 2) - center_x
    M[1, 2] += (new_h / 2) - center_y
    
    # 좌표 변환
    new_coords = []
    for i in range(0, len(coords), 2):
        x, y = coords[i], coords[i+1]
        new_point = np.dot(M, [x, y, 1])
        new_coords.extend(new_point[:2])
    
    # 패딩 조정
    pad_x = (new_size[1] - new_w) // 2
    pad_y = (new_size[0] - new_h) // 2
    new_coords = [c + pad_x if i % 2 == 0 else c + pad_y for i, c in enumerate(new_coords)]
    
    return new_coords

# 사용 예시
original_size = (1000, 1000)  # 원본 이미지 크기 (높이, 너비)
new_size = (1200, 1200)  # 새 이미지 크기 (높이, 너비)
rotation_angle = 50  # 회전 각도

# 원본 좌표
original_coords = [100, 100, 200, 200, 300, 300, 400, 400, 100, 100]

# 좌표 변환
new_coords = rotate_and_pad_coordinates(original_coords, rotation_angle, original_size, new_size)

print("변환된 좌표:", new_coords)