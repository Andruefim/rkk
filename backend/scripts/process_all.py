import os
import glob
import subprocess
import sys

def process_all_clips(clips_dir):
    # Поддерживаемые форматы
    extensions = ("*.mp4", "*.mpg", "*.avi", "*.mov")
    video_files = []
    
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(clips_dir, ext)))
        
    if not video_files:
        print(f"В папке {clips_dir} не найдено видео файлов.")
        return

    print(f"Найдено {len(video_files)} видео для обработки.")
    
    # Путь к нашему скрипту process_video.py
    script_path = os.path.join(os.path.dirname(__file__), "process_video.py")
    
    success = 0
    for video in video_files:
        filename = os.path.basename(video)
        name_without_ext = os.path.splitext(filename)[0]
        
        print(f"\n--- Обработка: {filename} ---")
        try:
            # Запускаем process_video.py
            subprocess.run([sys.executable, script_path, video, name_without_ext], check=True)
            success += 1
        except subprocess.CalledProcessError:
            print(f"❌ Ошибка при обработке {filename}")
            
    print(f"\n✅ Готово! Успешно обработано: {success} из {len(video_files)}")
    print("Не забудьте запустить `python scripts/train_amp.py` чтобы обучить дискриминатор на всех новых видео!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        clips_dir = sys.argv[1]
    else:
        # Папка по умолчанию, исходя из твоего пути
        clips_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "clips"))
        
    if not os.path.exists(clips_dir):
        print(f"Папка не найдена: {clips_dir}")
        print("Использование: python process_all.py <путь_к_папке_с_видео>")
    else:
        process_all_clips(clips_dir)
