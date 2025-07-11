#importing teacher model
import torch
from models.network_swinir import SwinIR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
swinir_model = SwinIR(
    upscale=4, img_size=64, window_size=8,
    depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
    upsampler='pixelshuffle', resi_connection='1conv'
).to(device)
weights = '/content/drive/MyDrive/archive/teacher/teacher.pth'
swinir_model.load_state_dict(torch.load(weights, map_location=device))
swinir_model.eval()
print("teacher model loaded from google drive")

#imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm
from models.network_swinir import SwinIR
import matplotlib.pyplot as plt)
sys.path.append('./SwinIR')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#defining student class
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
        self.d = d
        self.r = r

    def forward(self, x):
        out = self.act(self.c1(x))
        distilled, remaining = torch.split(out, [self.d, self.r], dim=1)
        out = self.act(self.c2(remaining))
        out = self.fuse(torch.cat([distilled, out], dim=1))
        return out + x

class IMDN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nf=64, num_modules=6, scale=2):
        super(IMDN, self).__init__()
        self.fea_conv = nn.Conv2d(in_channels, nf, 3, 1, 1)
        self.blocks = nn.Sequential(*[IMDModule(nf) for _ in range(num_modules)])
        self.lr_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(nf, out_channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )
        self.hr_conv = nn.Conv2d(nf, out_channels, 3, 1, 1)

    def forward(self, x):
        fea = self.fea_conv(x)
        out = self.blocks(fea)
        out = self.lr_conv(out) + fea
        out = self.upsampler(out)
        return out

#redefining teacher model
teacher = SwinIR(
    upscale=2,
    in_chans=3,
    img_size=64,
    window_size=8,
    img_range=1.,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler='pixelshuffle',
    resi_connection='1conv'
)


student = IMDN().to(device)


#defining dataset class
transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])
class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_paths = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)])
        self.hr_paths = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)])

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr = transform(Image.open(self.lr_paths[idx]).convert('RGB'))
        hr = transform(Image.open(self.hr_paths[idx]).convert('RGB'))
        return lr, hr
train_dataset = DIV2KDataset('/content/drive/MyDrive/archive/DATASET/dataset_root2/lr2',
                             '/content/drive/MyDrive/archive/DATASET/dataset_root2/hr2')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True



#dataloader and importing the dataset for training
train_dataset = DIV2KDataset(
    '/content/drive/MyDrive/archive/DATASET/dataset_root2/lr2',
    '/content/drive/MyDrive/archive/DATASET/dataset_root2/hr2'
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student = IMDN(scale=2).to(device)
teacher = SwinIR(
    upscale=2, in_chans=3, img_size=64, window_size=8, img_range=1.,
    depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
    upsampler='pixelshuffle', resi_connection='1conv'
).to(device)
teacher_state_dict = torch.load('/content/drive/MyDrive/archive/teacher/teacher.pth')
filtered_state_dict = {k: v for k, v in teacher_state_dict.items() if k in teacher.state_dict() and teacher.state_dict()[k].shape == v.shape}
teacher.load_state_dict(filtered_state_dict, strict=False)
teacher.to(device).eval()


#training loop with knowledge distillation from teacher model
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
epochs = 50
save_path = "/content/drive/MyDrive/archive/student2"
os.makedirs(save_path, exist_ok=True)
losses = []
for epoch in range(epochs):
    student.train()
    total_loss = 0.0
    for lr_img, hr_img in train_loader:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        optimizer.zero_grad()
        student_output = student(lr_img)
        with torch.no_grad():
            teacher_output = teacher(lr_img)
        loss_sr = criterion(student_output, hr_img)
        loss_distill = criterion(student_output, teacher_output)
        loss = loss_sr + 0.1 * loss_distill
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
    torch.save(student.state_dict(), f"{save_path}/student_epoch_{epoch+1}.pth")
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), losses, marker='o', linestyle='-', color='blue')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
