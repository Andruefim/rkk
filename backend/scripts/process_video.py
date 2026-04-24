import os
import sys
import cv2
import numpy as np

# Пытаемся импортировать MediaPipe
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("❌ Ошибка: Не установлена библиотека MediaPipe.")
    print("Установите её командой: pip install mediapipe opencv-python")
    sys.exit(1)

def calculate_angle(a, b, c):
    """
    Вычисляет угол между тремя точками (b - вершина).
    Точки в формате [x, y, z].
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def process_video(video_path: str, output_name: str):
    print(f"Processing video: {video_path}")
    
    # Загружаем модель MediaPipe
    model_path = os.path.join(os.path.dirname(__file__), 'pose_landmarker_heavy.task')
    if not os.path.exists(model_path):
        print("Downloading MediaPipe model (approx 30MB)...")
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded!")

    # Настройка детектора для видео
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO)
    
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    
    frames_data = []
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # MediaPipe требует RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp_ms = int(frame_index * (1000.0 / fps))
        frame_index += 1
        
        results = detector.detect_for_video(mp_image, timestamp_ms)
        
        if not results.pose_world_landmarks:
            continue
            
        landmarks = results.pose_world_landmarks[0]
        
        # Индексы MediaPipe (World Landmarks)
        # 11: left_shoulder, 23: left_hip, 25: left_knee, 27: left_ankle, 31: left_foot_index
        # 12: right_shoulder, 24: right_hip, 26: right_knee, 28: right_ankle, 32: right_foot_index
        
        def get_pt(idx):
            return [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]

            
        l_sh, l_hip, l_knee, l_ank, l_foot = get_pt(11), get_pt(23), get_pt(25), get_pt(27), get_pt(31)
        r_sh, r_hip, r_knee, r_ank, r_foot = get_pt(12), get_pt(24), get_pt(26), get_pt(28), get_pt(32)
        
        # Вычисляем углы (в градусах)
        # Hip flexion: угол между плечом-бедром-коленом (около 180 когда стоит прямо)
        l_hip_angle = calculate_angle(l_sh, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_sh, r_hip, r_knee)
        
        # Knee flexion: бедро-колено-лодыжка (около 180 когда нога прямая)
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ank)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ank)
        
        # Ankle: колено-лодыжка-стопа (около 90-110 градусов обычно)
        l_ank_angle = calculate_angle(l_knee, l_ank, l_foot)
        r_ank_angle = calculate_angle(r_knee, r_ank, r_foot)
        
        # --- Нормализация для GNN/PyBullet (0.05 - 0.95) ---
        def norm_hip(deg):
            val = 0.5 + ((180 - deg) / 90.0) * 0.3
            return float(np.clip(val, 0.05, 0.95))
            
        def norm_knee(deg):
            val = 0.5 - ((180 - deg) / 90.0) * 0.3
            return float(np.clip(val, 0.05, 0.95))
            
        def norm_ankle(deg):
            val = 0.5 + ((deg - 90) / 45.0) * 0.2
            return float(np.clip(val, 0.05, 0.95))
            
        frame_vec = [
            norm_hip(l_hip_angle),   # 0: lhip
            norm_hip(r_hip_angle),   # 1: rhip
            norm_knee(l_knee_angle), # 2: lknee
            norm_knee(r_knee_angle), # 3: rknee
            norm_ankle(l_ank_angle), # 4: lankle
            norm_ankle(r_ank_angle), # 5: rankle
        ]
        
        frames_data.append(frame_vec)

    cap.release()
    
    if not frames_data:
        print("Error: Could not extract poses from video.")
        return
        
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mocap_data"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{output_name}.npz")
    
    np.savez(out_path, angles=np.array(frames_data))
    print(f"Successfully extracted {len(frames_data)} frames!")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python process_video.py <путь_к_видео> <имя_выхода>")
        print("Пример: python process_video.py walk.mp4 walk_clip_01")
        sys.exit(1)
        
    process_video(sys.argv[1], sys.argv[2])
