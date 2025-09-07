import argparse, random, os, csv
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Basic config
NUM_CLASSES = 10
IMG_SIZE = 32
LIDAR_DIM = 16
GPS_DIM = 2

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# Toy multimodal dataset (CIFAR10 + synthetic GPS/LiDAR)
class ToyMultiModal(Dataset):
    """
    Returns (image, lidar_vec, gps_vec, label)
    - image: CIFAR-10 32x32 RGB
    - gps:   2D vector weakly correlated with class
    - lidar: 16D vector weakly correlated with class
    Optional perturbations simulate harsh environments.
    """
    def __init__(self, train=True, perturb=None):
        tfm = transforms.Compose([transforms.ToTensor()])
        self.base = datasets.CIFAR10(root="./data", train=train, download=True, transform=tfm)
        self.perturb = perturb or {}

    def __len__(self):
        return len(self.base)

    def _gen_gps(self, y):
        # class-dependent mean centers
        centers = np.linspace(-1, 1, NUM_CLASSES)
        mu_x = centers[y]; mu_y = centers[::-1][y]
        gps = np.array([np.random.normal(mu_x, 0.3), np.random.normal(mu_y, 0.3)], dtype=np.float32)
        return gps

    def _gen_lidar(self, y):
        # simple class-dependent 16D vector
        rng = np.random.default_rng()
        base = np.zeros(LIDAR_DIM, dtype=np.float32)
        base[y % LIDAR_DIM] = 1.0
        vec = base + 0.1 * rng.standard_normal(LIDAR_DIM)
        return vec.astype(np.float32)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        gps = torch.from_numpy(self._gen_gps(y))
        lidar = torch.from_numpy(self._gen_lidar(y))

        # perturbations (simulate environment)
        if self.perturb.get("camera_blur", False):
            img = transforms.GaussianBlur(kernel_size=3)(img)
        gps_noise = self.perturb.get("gps_noise", 0.0)
        if gps_noise > 0:
            gps = gps + gps_noise * torch.randn_like(gps)
        lidar_dropout = self.perturb.get("lidar_dropout", 0.0)
        if lidar_dropout > 0:
            mask = (torch.rand_like(lidar) > lidar_dropout).float()
            lidar = lidar * mask

        return img, lidar, gps, y

# Encoders
class CamEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*(IMG_SIZE//4)*(IMG_SIZE//4), out_dim), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class LiDAREncoder(nn.Module):
    def __init__(self, in_dim=LIDAR_DIM, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, out_dim), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class GPSEncoder(nn.Module):
    def __init__(self, in_dim=GPS_DIM, out_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, out_dim), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

# Fusion heads
class ConcatHead(nn.Module):
    """Feature-level fusion (concatenate embeddings)"""
    def __init__(self, dim_cam=64, dim_lidar=32, dim_gps=16, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_cam + dim_lidar + dim_gps, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, zc, zl, zg):
        z = torch.cat([zc, zl, zg], dim=1)
        return self.fc(z)

class AttnFusionHead(nn.Module):
    """Attention-based fusion over modality embeddings"""
    def __init__(self, dim_cam=64, dim_lidar=32, dim_gps=16,
                 num_classes=NUM_CLASSES, num_heads=4, d_model=64):
        super().__init__()
        self.proj_c = nn.Linear(dim_cam, d_model)
        self.proj_l = nn.Linear(dim_lidar, d_model)
        self.proj_g = nn.Linear(dim_gps, d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.cls = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, zc, zl, zg):
        # BxD -> Bx1xD, stack to Bx3xD
        seq = torch.stack([self.proj_c(zc), self.proj_l(zl), self.proj_g(zg)], dim=1)
        out, _ = self.mha(seq, seq, seq)  # self-attention over modalities
        pooled = out.mean(dim=1)          # simple pooling; can try attentive pooling/gating
        return self.cls(pooled)

# Train/Eval loops
def run(model_cam, model_lid, model_gps, head, loader, device, opt=None):
    training = opt is not None
    for m in [model_cam, model_lid, model_gps, head]:
        m.train() if training else m.eval()

    loss_sum, correct, n = 0.0, 0, 0
    for img, lidar, gps, y in loader:
        img, lidar, gps, y = img.to(device), lidar.to(device), gps.to(device), y.to(device)

        zc = model_cam(img)
        zl = model_lid(lidar)
        zg = model_gps(gps)
        logits = head(zc, zl, zg)
        loss = F.cross_entropy(logits, y)

        if training:
            opt.zero_grad(); loss.backward(); opt.step()

        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)

    return loss_sum / n, correct / n

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    # perturbations
    ap.add_argument("--blur", action="store_true", help="simulate foggy camera")
    ap.add_argument("--gps-noise", type=float, default=0.0)
    ap.add_argument("--lidar-dropout", type=float, default=0.0)
    # choose variant
    ap.add_argument("--only", choices=["concat", "attn"], default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print("Device:", device)

    # Data
    trainset = ToyMultiModal(train=True,  perturb=None)
    test_clean = ToyMultiModal(train=False, perturb=None)
    test_noisy = ToyMultiModal(train=False, perturb={
        "camera_blur": args.blur,
        "gps_noise": args.gps_noise,
        "lidar_dropout": args.lidar_dropout
    })
    train_loader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2)
    test_loader_clean = DataLoader(test_clean, batch_size=args.batch)
    test_loader_noisy = DataLoader(test_noisy, batch_size=args.batch)

    results = {"variant": [], "clean_acc": [], "noisy_acc": []}

    # helper: train one variant from scratch
    def train_variant(head_cls, name):
        cam = CamEncoder().to(device)
        lid = LiDAREncoder().to(device)
        gps = GPSEncoder().to(device)
        head = head_cls().to(device)
        opt = torch.optim.Adam(
            list(cam.parameters()) + list(lid.parameters()) +
            list(gps.parameters()) + list(head.parameters()),
            lr=args.lr
        )
        for ep in range(args.epochs):
            tr_loss, tr_acc = run(cam, lid, gps, head, train_loader, device, opt)
            te_loss, te_acc = run(cam, lid, gps, head, test_loader_clean, device)
            print(f"[{name}] epoch {ep+1}/{args.epochs}: train_acc={tr_acc:.3f} clean_acc={te_acc:.3f}")

        clean = run(cam, lid, gps, head, test_loader_clean, device)[1]
        noisy = run(cam, lid, gps, head, test_loader_noisy, device)[1]
        print(f"[{name}] CLEAN={clean:.3f}  NOISY={noisy:.3f}  DROP={clean-noisy:.3f}")

        results["variant"].append(name); results["clean_acc"].append(clean); results["noisy_acc"].append(noisy)

    # run variants
    if args.only in (None, "concat"):
        train_variant(ConcatHead, "concat")
    if args.only in (None, "attn"):
        train_variant(AttnFusionHead, "attn")

    # Save CSV
    os.makedirs("outputs", exist_ok=True)
    csv_path = os.path.join("outputs", "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "clean_acc", "noisy_acc", "drop"])
        for v, c, n in zip(results["variant"], results["clean_acc"], results["noisy_acc"]):
            w.writerow([v, f"{c:.4f}", f"{n:.4f}", f"{(c-n):.4f}"])
    print("Saved:", csv_path)

if __name__ == "__main__":
    main()
