import os, glob, math, argparse, random, csv
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# utils
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def ensure_divisible(a, b):
    assert a % b == 0, f"{a} must be divisible by {b}"

# dataset
class KITTISpeedDataset(Dataset):
    def __init__(self, root, seqs, split="train", resize=(128, 384),
                 bev_cfg=(0.0, 50.0, -20.0, 20.0, 32, 32),
                 perturb=None):
        self.root, self.seqs = root, seqs
        self.resize = resize
        self.bev_cfg = bev_cfg
        self.perturb = perturb or {}
        self.tfm = transforms.Compose([transforms.ToTensor()])

        # index all frames across sequences where three modalities exist
        self.samples = []
        for seq in seqs:
            base = os.path.join(root, "sequences", f"{seq:02d}")
            img_dir = os.path.join(base, "image_2")
            vel_dir = os.path.join(base, "velodyne")
            oxt_dir = os.path.join(base, "oxts", "data")
            imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            vels = sorted(glob.glob(os.path.join(vel_dir, "*.bin")))
            oxts = sorted(glob.glob(os.path.join(oxt_dir, "*.txt")))
            n = min(len(imgs), len(vels), len(oxts))
            for i in range(n):
                self.samples.append((imgs[i], vels[i], oxts[i]))
        if split == "train":
            self.samples = self.samples[: int(0.8 * len(self.samples))]
        elif split == "val":
            self.samples = self.samples[int(0.8 * len(self.samples)) :]

        if len(self.samples) == 0:
            raise RuntimeError("No KITTI samples found. Check root and sequence IDs.")

    def __len__(self): return len(self.samples)

    # quality priors
    @staticmethod
    def image_quality(img_gray_np):
        # contrast proxy: std of grayscale [0,1]
        s = float(np.std(img_gray_np))
        return np.clip(s / 0.25, 0.0, 1.0)  # normalize roughly

    @staticmethod
    def bev_quality(bev):
        # occupancy ratio
        occ = float((bev > 0).mean())
        return float(np.clip(occ / 0.10, 0.0, 1.0))  # 10% occupancy -> quality 1

    @staticmethod
    def gps_quality(gps_vec):
        # small magnitude (|noise| not known) -> use finite check
        m = float(np.linalg.norm(gps_vec) + 1e-6)
        return 1.0 if np.isfinite(m) else 0.0

    # parsers
    @staticmethod
    def read_velodyne_bin(path):
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)  # x,y,z,reflect
        return pts

    @staticmethod
    def read_oxts_speed_vec(path):
        # KITTI oxts/data/*.txt: see devkit; we only take vn (forward), ve (left)
        with open(path, "r") as f:
            vals = f.readline().strip().split()
        vals = [float(v) for v in vals]
        # indices: vn=11, ve=12 in the 30+ fields (KITTI doc)
        vn, ve = vals[11], vals[12]
        spd = math.sqrt(vn * vn + ve * ve)  # m/s
        gps_vec = np.array([vn, ve], dtype=np.float32)
        # 3 bins: 0–5, 5–10, >10 m/s
        if   spd < 5.0: y = 0
        elif spd < 10.: y = 1
        else: y = 2
        return gps_vec, y

    def make_bev(self, pts):
        x_min, x_max, y_min, y_max, H, W = self.bev_cfg
        x, y = pts[:, 0], pts[:, 1]
        mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        x, y = x[mask], y[mask]
        # histogram2d -> occupancy grid
        H2, xedges, yedges = np.histogram2d(
            x, y, bins=[H, W], range=[[x_min, x_max], [y_min, y_max]]
        )
        H2 = (H2 > 0).astype(np.float32)  # occupancy only
        return H2  # H x W

    def __getitem__(self, i):
        img_p, vel_p, oxt_p = self.samples[i]
        # camera
        img = Image.open(img_p).convert("RGB").resize(self.resize[::-1], Image.BILINEAR)
        # optional fog/blur
        if self.perturb.get("camera_blur", False):
            img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
            img = ImageEnhance.Brightness(img).enhance(0.85)
        img_t = self.tfm(img)  # 3xHxW
        img_gray = np.array(img.convert("L"), dtype=np.float32) / 255.0

        # lidar -> BEV
        pts = self.read_velodyne_bin(vel_p)
        # optional dropout
        ldrop = self.perturb.get("lidar_dropout", 0.0)
        if ldrop > 0.0 and pts.shape[0] > 0:
            m = np.random.rand(pts.shape[0]) > ldrop
            pts = pts[m]
        bev = self.make_bev(pts)                          # H x W
        bev_t = torch.from_numpy(bev.reshape(-1))         # (H*W,)
        # gps/imu -> vec + label
        gps_vec, y = self.read_oxts_speed_vec(oxt_p)
        gsn = self.perturb.get("gps_noise", 0.0)
        if gsn > 0:
            gps_vec = gps_vec + np.random.normal(0, gsn, size=2).astype(np.float32)
        gps_t = torch.from_numpy(gps_vec)

        # quality priors
        q_img = self.image_quality(img_gray)
        q_bev = self.bev_quality(bev)
        q_gps = self.gps_quality(gps_vec)
        q = torch.tensor([q_img, q_bev, q_gps], dtype=torch.float32)

        return img_t, bev_t, gps_t, int(y), q

#  models
class CamEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 128x384 -> 64x192
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64x192 -> 32x96
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
            nn.Linear(32, out_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class LiDAREncoder(nn.Module):
    def __init__(self, in_dim=32*32, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, out_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class GPSEncoder(nn.Module):
    def __init__(self, in_dim=2, out_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, out_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class ConcatHead(nn.Module):
    def __init__(self, dc=64, dl=32, dg=16, num_classes=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dc+dl+dg, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
    def forward(self, zc, zl, zg):
        return self.fc(torch.cat([zc, zl, zg], dim=1))

class QAwareAttnHead(nn.Module):
    def __init__(self, dc=64, dl=32, dg=16, num_classes=3,
                 d_model=120, num_heads=6, attn_dropout=0.15,
                 temperature=1.5, alpha_prior=2.0):
        super().__init__()
        ensure_divisible(d_model, num_heads)
        self.proj_c = nn.Sequential(nn.Linear(dc, d_model), nn.LayerNorm(d_model))
        self.proj_l = nn.Sequential(nn.Linear(dl, d_model), nn.LayerNorm(d_model))
        self.proj_g = nn.Sequential(nn.Linear(dg, d_model), nn.LayerNorm(d_model))
        self.mha = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.dropout = nn.Dropout(attn_dropout)
        self.cls = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(128, num_classes))
        self.gate_mlp = nn.Sequential(nn.Linear(d_model*3, 64), nn.ReLU(), nn.Linear(64, 3))
        self.temperature = temperature
        self.alpha_prior = alpha_prior
        self.last_gate_probs = None        # (B,3)
        self.last_attn_probs = None        # (B,T,T)

    def forward(self, zc, zl, zg, q_prior=None):
        # encode
        zc, zl, zg = self.proj_c(zc), self.proj_l(zl), self.proj_g(zg)  # BxD
        seq = torch.stack([zc, zl, zg], dim=1)                          # Bx3xD
        # gating with prior
        gate_logits = self.gate_mlp(torch.cat([zc, zl, zg], dim=1))     # Bx3
        if q_prior is not None:
            eps = 1e-6
            gate_logits = gate_logits + self.alpha_prior * (q_prior.clamp_min(eps).log())
        gp = torch.softmax(gate_logits, dim=-1)                         # Bx3
        self.last_gate_probs = gp
        seq = seq * gp[:, :, None]                                      # Bx3xD
        # attention with temperature
        seq = seq / self.temperature
        out, attn = self.mha(seq, seq, seq, need_weights=True)          # (B,T,D), (B,T,T)
        self.last_attn_probs = attn
        out = self.dropout(out.mean(dim=1))
        return self.cls(out)

# train
def _entropy(p, eps=1e-8):
    return -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=-1)

def run_epoch(cam, lid, gps, head, loader, device, opt=None,
              reg=None, modality_dropout_p=0.0,
              sup_drop_p=0.5, sup_gate_w=0.10, sup_attn_w=0.05):
    train = opt is not None
    for m in [cam, lid, gps, head]: m.train() if train else m.eval()

    n, loss_sum, correct = 0, 0.0, 0
    for img, bev, gvec, y, q in loader:
        img, bev, gvec, y, q = img.to(device), bev.to(device), gvec.to(device), y.to(device), q.to(device)
        zc, zl, zg = cam(img), lid(bev), gps(gvec)

        if train and modality_dropout_p > 0 and np.random.rand() < modality_dropout_p:
            pick = np.random.choice([0,1,2])
            if pick==0: zc=zc*0
            elif pick==1: zl=zl*0
            else: zg=zg*0

        corrupted = None
        if train and np.random.rand() < sup_drop_p:
            corrupted = np.random.choice([0,1,2])
            if corrupted==0: zc=zc*0
            elif corrupted==1: zl=zl*0
            else: zg=zg*0

        # forward
        if isinstance(head, QAwareAttnHead):
            logits = head(zc, zl, zg, q_prior=q)
        else:
            logits = head(zc, zl, zg)
        loss = F.cross_entropy(logits, y)

        if train and corrupted is not None and isinstance(head, QAwareAttnHead):
            gp = head.last_gate_probs                  # Bx3
            loss = loss + sup_gate_w * gp[:, corrupted].mean()
            attn = head.last_attn_probs               # BxT×T
            loss = loss + sup_attn_w * attn[:, :, corrupted].mean()

        if train and isinstance(head, QAwareAttnHead) and reg is not None:
            if reg.get("lambda_gate", 0) > 0:
                Hg = _entropy(head.last_gate_probs.mean(dim=0))   # scalar
                loss = loss - reg["lambda_gate"] * Hg
            if reg.get("lambda_attn", 0) > 0:
                Ha = _entropy(head.last_attn_probs.mean(dim=1)).mean()
                loss = loss - reg["lambda_attn"] * Ha

        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(cam.parameters())+list(lid.parameters())+
                                           list(gps.parameters())+list(head.parameters()), 1.0)
            opt.step()

        loss_sum += loss.item() * y.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        n += y.size(0)

    return loss_sum / n, correct / n

#main 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="KITTI root with sequences/*")
    ap.add_argument("--seqs", type=int, nargs="+", default=[0,1,2])      # choose a few sequences
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    # noise (test-time)
    ap.add_argument("--blur", action="store_true")
    ap.add_argument("--gps-noise", type=float, default=0.0)
    ap.add_argument("--lidar-dropout", type=float, default=0.0)
    # train-time light noise defaults (half of test)
    ap.add_argument("--train-gps-noise", type=float, default=None)
    ap.add_argument("--train-lidar-dropout", type=float, default=None)
    ap.add_argument("--modality-dropout-p", type=float, default=0.3)
    # attention
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=120)   # divisible by 6
    ap.add_argument("--attn-dropout", type=float, default=0.15)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--lambda-gate", type=float, default=0.02)
    ap.add_argument("--lambda-attn", type=float, default=0.02)
    ap.add_argument("--only", choices=["concat","attn"], default=None)
    args = ap.parse_args()

    ensure_divisible(args.d_model, args.heads)
    set_seed(42)
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print("Device:", device)

    # datasets
    train_perturb = {
        "camera_blur": args.blur,
        "gps_noise": args.train_gps_noise if args.train_gps_noise is not None else max(0.0, args.gps_noise*0.5),
        "lidar_dropout": args.train_lidar_dropout if args.train_lidar_dropout is not None else max(0.0, args.lidar_dropout*0.5)
    }
    test_clean = {"camera_blur": False, "gps_noise": 0.0, "lidar_dropout": 0.0}
    test_noisy = {"camera_blur": args.blur, "gps_noise": args.gps_noise, "lidar_dropout": args.lidar_dropout}

    train_set = KITTISpeedDataset(args.root, args.seqs, split="train", perturb=train_perturb)
    clean_set = KITTISpeedDataset(args.root, args.seqs, split="val",   perturb=test_clean)
    noisy_set = KITTISpeedDataset(args.root, args.seqs, split="val",   perturb=test_noisy)
    print("Sizes:", len(train_set), len(clean_set), len(noisy_set))

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    clean_loader = DataLoader(clean_set, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
    noisy_loader = DataLoader(noisy_set, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    results = {"variant":[], "clean_acc":[], "noisy_acc":[]}

    def train_one(name):
        cam, lid, gps = CamEncoder().to(device), LiDAREncoder().to(device), GPSEncoder().to(device)
        if name == "attn":
            head = QAwareAttnHead(d_model=args.d_model, num_heads=args.heads,
                                  attn_dropout=args.attn_dropout, temperature=args.temperature).to(device)
            reg = {"lambda_gate": args.lambda_gate, "lambda_attn": args.lambda_attn}
        else:
            head = ConcatHead().to(device); reg = None

        params = list(cam.parameters())+list(lid.parameters())+list(gps.parameters())+list(head.parameters())
        opt = torch.optim.Adam(params, lr=args.lr, weight_decay=5e-4)

        for ep in range(args.epochs):
            tr_loss, tr_acc = run_epoch(cam, lid, gps, head, train_loader, device, opt,
                                        reg, modality_dropout_p=(args.modality_dropout_p if name=="attn" else 0.0))
            te_loss, te_acc = run_epoch(cam, lid, gps, head, clean_loader, device)
            if (ep+1)%2==0:
                print(f"[{name}] epoch {ep+1}/{args.epochs} train_acc={tr_acc:.3f} clean_acc={te_acc:.3f}")

        cacc = run_epoch(cam, lid, gps, head, clean_loader, device)[1]
        nacc = run_epoch(cam, lid, gps, head, noisy_loader, device)[1]
        print(f"[{name}] CLEAN={cacc:.4f}  NOISY={nacc:.4f}  DROP={cacc-nacc:.4f}")
        results["variant"].append(name); results["clean_acc"].append(cacc); results["noisy_acc"].append(nacc)

    if args.only in (None, "concat"): train_one("concat")
    if args.only in (None, "attn"):   train_one("attn")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/results.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["variant","clean_acc","noisy_acc","drop"])
        for v,c,n in zip(results["variant"], results["clean_acc"], results["noisy_acc"]):
            w.writerow([v, f"{c:.4f}", f"{n:.4f}", f"{(c-n):.4f}"])
    print("Saved outputs/results.csv")

if __name__ == "__main__":
    main()
