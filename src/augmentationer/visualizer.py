import cv2
import numpy as np
from typing import Union
from pathlib import Path

def visualizer(image_path: Union[str, Path], label_path: Union[str, Path]):
    """
    주어진 이미지와 라벨 파일을 시각화하는 함수입니다.
    
    Parameters:
    image_path (str): 시각화할 이미지 파일의 경로
    label_path (str): 시각화할 라벨 파일의 경로
    """
    # 이미지 읽기
    image = cv2.imread(image_path)
    
    # 라벨 파일 읽기
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # 이미지 크기 가져오기
    height, width = image.shape[:2]
    
    for line in lines:
        # OBB 형식의 라벨 파싱 (class x1 y1 x2 y2 x3 y3 x4 y4 x1 y1)
        values = list(map(float, line.strip().split()))
        class_id = int(values[0])
        points = np.array(values[1:9]).reshape(-1, 2)  # x1 y1부터 x4 y4까지만 사용
        
        # 정규화된 좌표를 픽셀 좌표로 변환
        points[:, 0] *= width
        points[:, 1] *= height
        points = points.astype(np.int32)
        
        # 폴리곤 그리기
        cv2.polylines(image, [points], True, (0, 255, 0), 2)
        
        # 클래스 ID 표시
        cv2.putText(image, f'Class: {class_id}', tuple(points[0]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 결과 이미지 표시
    window_name = 'Rotated Image with OBB Labels'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)  # 원하는 크기로 조절 (예: 800x600)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":  
    name = "image2_augmentation_3"
    # 함수 사용 예시
    visualizer(f'output/images/{name}.jpg', f'output/labels/{name}.txt')
