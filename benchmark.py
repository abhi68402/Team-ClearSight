
import os
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from google.colab import files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule, self).__init__()
        c = in_channels
        d = int(c * distillation_rate)
        r = c - d
        self.c1 = nn.Conv2d(c, c, 3, 1, 1)
        self.act = nn.LeakyReLU(0.05, inplace=True)
        self.c2 = nn.Conv2d(r, c, 3, 1, 1)
        self.fuse = nn.Conv2d(c + d, c, 1, 1, 0)

    def forward(self, x):
        out = self.act(self.c1(x))
        d = int(out.shape[1] * 0.25)
        distilled, remaining = torch.split(out, [d, out.shape[1] - d], dim=1)
        out = self.act(self.c2(remaining))
        out = self.fuse(torch.cat([distilled, out], dim=1))
        return out + x

class IMDN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nf=64, num_modules=6, scale=2):
        super(IMDN, self).__init__()
        self.fea_conv = nn.Conv2d(in_channels, nf, 3, 1, 1)
        self.blocks = nn.Sequential(*[IMDModule(nf, distillation_rate=0.25) for _ in range(num_modules)])
        self.lr_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(nf, out_channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        fea = self.fea_conv(x)
        out = self.blocks(fea)
        out = self.lr_conv(out) + fea
        out = self.upsampler(out)
        return out

student = IMDN(scale=2).to(device)
state_dict = torch.load('/content/drive/MyDrive/archive/student2/student_epoch_43.pth', map_location=device)
state_dict = {k: v for k, v in state_dict.items() if k in student.state_dict()}
student.load_state_dict(state_dict, strict=False)
student.eval()

def load_and_preprocess(img_path, size=None):
    image = Image.open(img_path).convert('RGB')
    if size:
        image = image.resize(size, Image.BICUBIC)
    transform = T.ToTensor()
    return transform(image).unsqueeze(0)

def tensor_to_image(tensor):
    image = tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(image, 0, 1)

def compute_ssim(sr_image, hr_image):
    sr_image = (sr_image * 255).astype(np.uint8)
    hr_image = (hr_image * 255).astype(np.uint8)
    h, w = min(sr_image.shape[0], hr_image.shape[0]), min(sr_image.shape[1], hr_image.shape[1])
    sr_crop = sr_image[:h, :w]
    hr_crop = hr_image[:h, :w]
    return ssim(sr_crop, hr_crop, channel_axis=2, data_range=255)

def compute_psnr(sr_image, hr_image):
    return psnr(hr_image, sr_image, data_range=1.0)


print("Upload LR image:")
uploaded = files.upload()
lr_path = list(uploaded.keys())[0]

print("Upload HR image:")
uploaded = files.upload()
hr_path = list(uploaded.keys())[0]

hr_pil = Image.open(hr_path).convert('RGB')
hr_size = hr_pil.size
lr_size = (hr_size[0] // 2, hr_size[1] // 2)

lr_tensor = load_and_preprocess(lr_path, size=lr_size).to(device)
hr_tensor = load_and_preprocess(hr_path, size=hr_size).to(device)
with torch.no_grad():
    sr_tensor = student(lr_tensor)


sr_img_np = tensor_to_image(sr_tensor)
hr_img_np = tensor_to_image(hr_tensor)

lr_pil = Image.open(lr_path).convert('RGB').resize(hr_size, Image.BICUBIC)
lr_img_np = np.array(lr_pil).astype(np.float32) / 255.

ssim_score = compute_ssim(sr_img_np, hr_img_np)
psnr_score = compute_psnr(sr_img_np, hr_img_np)

print(f"SSIM Accuracy: {ssim_score:.4f}")
print(f" PSNR: {psnr_score:.2f} dB")
zoom_factor = 3

def crop_zoom(image_np):
    h, w = image_np.shape[:2]
    zh, zw = h // zoom_factor, w // zoom_factor
    ch, cw = h // 2, w // 2
    crop = image_np[ch - zh // 2: ch + zh // 2, cw - zw // 2: cw + zw // 2]
    crop = Image.fromarray((crop * 255).astype(np.uint8)).resize((zw * zoom_factor, zh * zoom_factor), Image.NEAREST)
    return np.array(crop) / 255.

lr_zoom = crop_zoom(lr_img_np)
sr_zoom = crop_zoom(sr_img_np)
hr_zoom = crop_zoom(hr_img_np)

plt.figure(figsize=(15, 8))
plt.subplot(2, 3, 1)
plt.imshow(lr_img_np)
plt.title('LR')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(sr_img_np)
plt.title('SR Output')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(lr_zoom)
plt.title(f'LR Zoom x{zoom_factor}')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(sr_zoom)
plt.title(f'SR Zoom x{zoom_factor}')
plt.axis('off')
plt.suptitle(f"SSIM: {ssim_score:.4f} | PSNR: {psnr_score:.2f} dB", fontsize=16)
plt.tight_layout()
plt.show()
