import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Добавляем корень backend в sys.path, чтобы импортировать engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.mocap_loader import MoCapDataLoader, MotionDiscriminator

def train_amp():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mocap_data"))
    loader = MoCapDataLoader(data_dir=data_dir)
    
    # Если загрузился только сгенерированный fallback (длина 1), предупреждаем
    real_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    if not real_files:
        print("Warning: No real .npz files found in mocap_data!")
        print("Training will run on the synthetic fallback clip.")
        print("Please run scripts/process_video.py on a real video first.")
        print("-" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator = MotionDiscriminator(state_dim=8).to(device)
    
    optimizer = optim.Adam(discriminator.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    batch_size = 256
    epochs = 2000
    steps_per_epoch = 10
    
    print(f"Starting AMP Discriminator training on {device}...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for _ in range(steps_per_epoch):
            # 1. Собираем батч РЕАЛЬНЫХ данных из MoCap (Label = 1)
            # Формат каждого фрейма: [lhip, rhip, lknee, rknee, lankle, rankle, com_z, posture]
            real_batch = []
            for _ in range(batch_size):
                # Берем случайный кусок длиной 1 (просто один кадр)
                frame = loader.sample_clip(1)[0]
                # Добавляем фиктивные com_z и posture_stability (в парсере их может не быть)
                if len(frame) == 6:
                    frame = np.append(frame, [0.85, 0.95]) # high com, stable
                real_batch.append(frame)
            
            real_tensor = torch.tensor(np.array(real_batch), dtype=torch.float32).to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            
            # 2. Генерируем батч ФЕЙКОВЫХ данных (Label = 0)
            # Фейк 1: Случайный шум в пределах суставов [0.05, 0.95]
            fake_noise = torch.rand(batch_size // 2, 8).to(device) * 0.9 + 0.05
            
            # Фейк 2: Неправильная осанка (низкий com_z, низкий posture)
            fake_fallen = torch.tensor(np.array(real_batch[:batch_size // 2]), dtype=torch.float32).to(device)
            fake_fallen[:, 6] = torch.rand(batch_size // 2).to(device) * 0.4  # fallen com_z
            fake_fallen[:, 7] = torch.rand(batch_size // 2).to(device) * 0.4  # bad posture
            
            fake_tensor = torch.cat([fake_noise, fake_fallen], dim=0)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Объединяем
            states = torch.cat([real_tensor, fake_tensor], dim=0)
            labels = torch.cat([real_labels, fake_labels], dim=0)
            
            # 3. Шаг обучения
            optimizer.zero_grad()
            logits = discriminator(states)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/steps_per_epoch:.4f}")

    # Сохраняем веса
    os.makedirs(os.path.join(data_dir, "weights"), exist_ok=True)
    save_path = os.path.join(data_dir, "weights", "amp_discriminator.pth")
    torch.save(discriminator.state_dict(), save_path)
    print(f"Discriminator weights saved to: {save_path}")

if __name__ == "__main__":
    train_amp()
