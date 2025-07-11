#mounting drive
from google.colab import drive
drive.mount('/content/drive')

#cloning the official repo
!git clone https://github.com/JingyunLiang/SwinIR.git
%cd SwinIR
!pip install basicsr timm


#imports
import sys
import torch
from models.network_swinir import SwinIR
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
sys.path.append('./')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#defining edges
import torch.nn.functional as F
def edges(x):
    Gx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3).to(x.device)
    Gy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3).to(x.device)
    edge_x = F.conv2d(x, Gx, padding=1, groups=1)
    edge_y = F.conv2d(x, Gy, padding=1, groups=1)
    return torch.sqrt(edge_x ** 2 + edge_y ** 2)

#SwinIR architecture
from models.network_swinir import SwinIR
swinir_model = SwinIR(
    upscale=4,
    in_chans=3,
    img_size=64,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler='pixelshuffle',
    resi_connection='1conv'
).to(device)


#loading weights from drive
import torch
weight_path = '/content/drive/MyDrive/archive/model/swinDIVK.pth'
raw_state = torch.load(weight_path, map_location=device)
state_dict = raw_state['params'] if 'params' in raw_state else raw_state
filtered_state_dict = {
    k: v for k, v in state_dict.items() if not k.endswith('attn_mask')
}
swinir_model.load_state_dict(filtered_state_dict, strict=False)
swinir_model.eval()
print("Model loaded from drive with attn_mask.")


#testing out teacher model
from PIL import Image, ImageFilter
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from google.colab import files

uploaded = files.upload()
img_path = list(uploaded.keys())[0]

original = Image.open(img_path).convert("RGB")
sharpened = original.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
orig_np = np.array(original).astype(np.float32) / 255.0
sharp_np = np.array(sharpened).astype(np.float32) / 255.0
ssim_val = compare_ssim(orig_np, sharp_np, channel_axis=2, data_range=1.0)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("input")
plt.imshow(original)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("output from model")
plt.imshow(sharpened)
plt.axis("off")
plt.suptitle(f" SSIM after sharpening: {ssim_val:.4f}", fontsize=12)
plt.tight_layout()
plt.show()


#saving the configuration to drive
torch.save(swinir_model.state_dict(), '/content/drive/MyDrive/archive/teacher/teacher.pth')
