import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import glob

# ==========================================
# 1. DnCNN MODEL MİMARİSİ (17 Katmanlı Standart Ağ)
# ==========================================
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        
        # İlk Katman: Conv + ReLU
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Ara Katmanlar: Conv + BatchNorm + ReLU
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            
        # Son Katman: Conv (Sadece gürültüyü tahmin eder)
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # DnCNN residual (artık) öğrenme yapar. Resmi alır, gürültüyü bulur, resimden çıkarır.
        out = self.dncnn(x)
        return x - out

# ==========================================
# 2. EĞİTİM İÇİN VERİ YÜKLEYİCİ (Dataset)
# ==========================================
class SPNoiseDataset(Dataset):
    def __init__(self, clean_img_paths):
        self.img_paths = clean_img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Temiz resmi oku (OpenCV BGR okur, RGB'ye çevir)
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # %10 ile %90 arası RASTGELE tuz-biber gürültüsü ekle
        noise_density = np.random.uniform(0.1, 0.9)
        noisy_img = np.copy(img)
        
        row, col, ch = noisy_img.shape
        num_salt = np.ceil(noise_density * img.size * 0.5)
        num_pepper = np.ceil(noise_density * img.size * 0.5)
        
        # Tuz (Beyaz)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in noisy_img.shape]
        noisy_img[tuple(coords)] = 255
        # Biber (Siyah)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in noisy_img.shape]
        noisy_img[tuple(coords)] = 0
        
        # Tensöre çevir (0-1 aralığına normalize et)
        noisy_tensor = torch.from_numpy(noisy_img.transpose((2, 0, 1))).float() / 255.0
        clean_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        
        return noisy_tensor, clean_tensor

# ==========================================
# 3. TEST KLASÖRLERİNİ TEMİZLEME (Inference)
# ==========================================
def denoise_test_folders(model, device):
    noise_levels = [10, 30, 50, 70, 90]
    model.eval()
    
    with torch.no_grad():
        for level in noise_levels:
            input_folder = f'Tiny_Noisy_{level}'
            output_folder = f'Tiny_Cleaned_DnCNN_{level}'
            
            if not os.path.exists(input_folder):
                continue
            os.makedirs(output_folder, exist_ok=True)
            
            img_paths = glob.glob(os.path.join(input_folder, '*.png'))
            print(f'\nDnCNN: {level}% Seviyesi Temizleniyor... ({len(img_paths)} resim)')
            
            for path in tqdm(img_paths):
                img_name = os.path.basename(path)
                
                # Resmi oku ve tensöre çevir
                noisy_img = cv2.imread(path)
                noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
                noisy_tensor = torch.from_numpy(noisy_img.transpose((2, 0, 1))).float() / 255.0
                noisy_tensor = noisy_tensor.unsqueeze(0).to(device) # Batch boyutu ekle
                
                # DnCNN ile temizle
                cleaned_tensor = model(noisy_tensor)
                
                # Tensörü tekrar resme (0-255) çevir
                cleaned_img = cleaned_tensor.squeeze().cpu().numpy().transpose((1, 2, 0))
                cleaned_img = np.clip(cleaned_img * 255.0, 0, 255).astype(np.uint8)
                cleaned_img = cv2.cvtColor(cleaned_img, cv2.COLOR_RGB2BGR)
                
                # Kaydet
                cv2.imwrite(os.path.join(output_folder, img_name), cleaned_img)

# ==========================================
# ANA ÇALIŞTIRMA BLOĞU
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Kullanılan Cihaz: {device}')
    
    model = DnCNN(channels=3, num_of_layers=17).to(device)
    
    # --- SENARYO 1: MODELİ EĞİT ---
    # Eğitmek istiyorsan burayı True yap. (Sadece 1 kez yapman yeterli)
    TRAIN_MODE = False 
    
    if TRAIN_MODE:
        print("DnCNN Tuz-Biber gürültüsü için eğitiliyor...")
        # Tiny ImageNet orijinal klasöründeki temiz resimleri bul
        # Tiny ImageNet'in bilgisayarındaki ana train klasörünün yolunu buraya yaz
        train_dir = 'tiny-imagenet-200/train' # Klasör adı sende farklıysa burayı güncelle

        # '**' ve recursive=True parametreleri sayesinde tüm alt klasörlerin içindeki JPEG'leri bulur
        clean_train_images = glob.glob(os.path.join(train_dir, '**', '*.JPEG'), recursive=True)

        print(f"Eğitim için toplam {len(clean_train_images)} görsel bulundu!")
        
        dataset = SPNoiseDataset(clean_train_images)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        epochs = 10 # Makale için 30-50 arası idealdir, test için 10 yeterli.
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for noisy_imgs, clean_imgs in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
                noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                
                optimizer.zero_grad()
                outputs = model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            print(f'Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.6f}')
        
        # Eğitilen modeli kaydet
        torch.save(model.state_dict(), 'dncnn_sp_noise.pth')
        print("Model başarıyla eğitildi ve kaydedildi!")
    
    else:
        # --- SENARYO 2: EĞİTİLMİŞ MODELİ YÜKLE ---
        print("Kayıtlı model yükleniyor...")
        model.load_state_dict(torch.load('dncnn_sp_noise.pth', map_location=device))
    
    # --- TEST KLASÖRLERİNİ TEMİZLE ---
    denoise_test_folders(model, device)
    print("DnCNN Temizleme İşlemleri Bitti!")