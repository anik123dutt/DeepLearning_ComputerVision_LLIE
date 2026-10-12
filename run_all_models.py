import os
import sys
import time
import glob
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# ==========================================
# Paths & Settings
# ==========================================
WORKSPACE = "/home/intern/internship_projects/Aniket"
CODEBASES = os.path.join(WORKSPACE, "codebases")
DATASET_DIR = os.path.join(WORKSPACE, "dataset", "Exclusively-Dark-Image-Dataset-master", "SPIC")
OUTPUT_BASE = os.path.join(WORKSPACE, "output")
os.makedirs(OUTPUT_BASE, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using {DEVICE} for all models.")

def clear_module_cache():
    for mod in list(sys.modules.keys()):
        if mod in ['model', 'operations', 'genotypes', 'multi_read_data', 'common', 'option']:
            del sys.modules[mod]

# ==========================================
# Quality & Enhancement Metrics
# ==========================================
def calculate_metrics(img_rgb_np):
    gray = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2GRAY)
    mean_brightness = float(np.mean(gray))
    std_contrast = float(np.std(gray))
    
    # Shannon Entropy (Information density)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    prob = hist / (hist.sum() + 1e-12)
    prob = prob[prob > 0]
    entropy = float(-np.sum(prob * np.log2(prob)))
    
    # Dynamic Range (5th to 95th percentile)
    p5, p95 = np.percentile(gray, (5, 95))
    dynamic_range = float(p95 - p5)
    
    return {
        "mean_brightness": round(mean_brightness, 2),
        "contrast_std": round(std_contrast, 2),
        "entropy": round(entropy, 2),
        "dynamic_range": round(dynamic_range, 2)
    }

# ==========================================
# 1. Classical Algorithms
# ==========================================
def run_classic_he(img_pil):
    img_np = np.array(img_pil.convert('RGB'))
    ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    res = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(res)

def run_classic_clahe(img_pil):
    img_np = np.array(img_pil.convert('RGB'))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    res = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(res)

def run_classic_retinex(img_pil):
    img = np.array(img_pil.convert('RGB')).astype(np.float64) + 1.0
    scales = [15, 80, 250]
    retinex = np.zeros_like(img)
    for s in scales:
        blur = cv2.GaussianBlur(img, (0, 0), s) + 1.0
        retinex += np.log10(img) - np.log10(blur)
    retinex /= len(scales)
    
    res = np.zeros_like(retinex)
    for c in range(3):
        mean = np.mean(retinex[:, :, c])
        std = np.std(retinex[:, :, c]) + 1e-6
        min_v = mean - 2 * std
        max_v = mean + 2 * std
        channel = np.clip((retinex[:, :, c] - min_v) / (max_v - min_v), 0, 1)
        res[:, :, c] = channel * 255.0
    return Image.fromarray(res.astype(np.uint8))

# ==========================================
# 2. Deep Learning Loaders & Runners
# ==========================================
def get_zero_dce():
    clear_module_cache()
    p = os.path.join(CODEBASES, "Zero-DCE-master", "Zero-DCE_code")
    sys.path.insert(0, p)
    import model
    net = model.enhance_net_nopool().to(DEVICE)
    ckpt = os.path.join(p, "snapshots", "Epoch99.pth")
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    net.eval()
    sys.path.remove(p)
    return net

def exec_zero_dce(net, img_pil):
    img_np = np.asarray(img_pil.convert('RGB')) / 255.0
    t = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, enhanced, _ = net(t)
    out_np = (enhanced.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(out_np)

def get_zero_dce_pp():
    clear_module_cache()
    p = os.path.join(CODEBASES, "Zero-DCE_extension-main", "Zero-DCE++")
    sys.path.insert(0, p)
    import model
    net = model.enhance_net_nopool(scale_factor=12).to(DEVICE)
    ckpt = os.path.join(p, "snapshots_Zero_DCE++", "Epoch99.pth")
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    net.eval()
    sys.path.remove(p)
    return net

def exec_zero_dce_pp(net, img_pil):
    scale_factor = 12
    img_np = np.asarray(img_pil.convert('RGB')) / 255.0
    h = (img_np.shape[0] // scale_factor) * scale_factor
    w = (img_np.shape[1] // scale_factor) * scale_factor
    crop = img_np[0:h, 0:w, :]
    t = torch.from_numpy(crop).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        enhanced, _ = net(t)
    out_np = (enhanced.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(out_np)

def get_ruas(variant="lol"):
    clear_module_cache()
    p = os.path.join(CODEBASES, "RUAS-main")
    cwd = os.getcwd()
    os.chdir(p)
    sys.path.insert(0, p)
    from model import Network
    net = Network().to(DEVICE)
    ckpt_path = os.path.join(p, "ckpt", f"{variant}.pt")
    net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    net.eval()
    sys.path.remove(p)
    os.chdir(cwd)
    return net

def exec_ruas(net, img_pil, variant="lol"):
    img_np = np.asarray(img_pil.convert('RGB')) / 255.0
    t = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        u_list, r_list = net(t)
        out_t = u_list[-1] if variant == "lol" else u_list[-2]
    out_np = (out_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(out_np)

def get_sci_cvpr(variant="medium"):
    clear_module_cache()
    p = os.path.join(CODEBASES, "SCI-main", "CVPR")
    cwd = os.getcwd()
    os.chdir(p)
    sys.path.insert(0, p)
    from model import Finetunemodel
    ckpt_path = os.path.join(p, "weights", f"{variant}.pt")
    net = Finetunemodel(ckpt_path).to(DEVICE)
    net.eval()
    sys.path.remove(p)
    os.chdir(cwd)
    return net

def exec_sci(net, img_pil):
    img_np = np.asarray(img_pil.convert('RGB')) / 255.0
    t = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        i, r = net(t)
    out_np = (r.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(out_np)

def get_sci_tpami():
    clear_module_cache()
    p = os.path.join(CODEBASES, "SCI-main", "TPAMI")
    cwd = os.getcwd()
    os.chdir(p)
    sys.path.insert(0, p)
    from model import Finetunemodel
    ckpt_path = os.path.join(p, "weights", "weights_1_3500.pt")
    net = Finetunemodel(ckpt_path).to(DEVICE)
    net.eval()
    sys.path.remove(p)
    os.chdir(cwd)
    return net

def get_drbn_stage1():
    clear_module_cache()
    p = os.path.join(CODEBASES, "CVPR-2020-Semi-Low-Light-master", "DRBL-stage1", "src")
    cwd = os.getcwd()
    os.chdir(p)
    sys.path.insert(0, p)
    from option import args
    from model.drbn import DRBN
    net = DRBN(args).to(DEVICE)
    ckpt_path = os.path.join(p, "pretrain", "model_s1.pt")
    net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    net.eval()
    sys.path.remove(p)
    os.chdir(cwd)
    return net

def exec_drbn(net, img_pil):
    img_np = np.asarray(img_pil.convert('RGB')).astype(np.float32)
    h, w, _ = img_np.shape
    h_pad = ((h + 7) // 8) * 8
    w_pad = ((w + 7) // 8) * 8
    padded = np.pad(img_np, ((0, h_pad - h), (0, w_pad - w), (0, 0)), mode='reflect')
    t = torch.from_numpy(padded).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = net(t)
        out_t = out[0] if isinstance(out, (list, tuple)) else out
    out_np = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_np = np.clip(out_np, 0, 255.0).astype(np.uint8)
    return Image.fromarray(out_np[:h, :w, :])

def get_cue():
    clear_module_cache()
    p = os.path.join(CODEBASES, "CUE-master")
    cwd = os.getcwd()
    os.chdir(p)
    sys.path.insert(0, p)
    from basicsr.archs.LearnablePriorsCondUNetFreReflRestore_arch import LearnablePriorEnhanceNet
    net = LearnablePriorEnhanceNet(stage=3, illu_channel=1, refl_channel=3, noise_channel=3, 
                                  num_feat=32, ratio=5, alpha=0.001, beta=0.001, mu=10.0).to(DEVICE)
    ckpt_path = os.path.join(p, "pretrained_model", "enhanceNet.pth")
    weights = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = weights['params_ema'] if 'params_ema' in weights else (weights['params'] if 'params' in weights else weights)
    net.load_state_dict(state)
    net.eval()
    sys.path.remove(p)
    os.chdir(cwd)
    return net

def exec_cue(net, img_pil):
    img_np = np.asarray(img_pil.convert('RGB')) / 255.0
    h, w, _ = img_np.shape
    h_pad = ((h + 15) // 16) * 16
    w_pad = ((w + 15) // 16) * 16
    padded = np.pad(img_np, ((0, h_pad - h), (0, w_pad - w), (0, 0)), mode='reflect')
    t = torch.from_numpy(padded).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        I_enhance, _, _, _, _, _, _ = net(t)
    out_np = (I_enhance.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(out_np[:h, :w, :])

def get_dimma():
    from diffusers import UNet2DModel
    from omegaconf import OmegaConf
    dimma_dir = os.path.join(CODEBASES, "Dimma-main")
    cfg = OmegaConf.load(os.path.join(dimma_dir, 'configs', 'LOL', 'stage2', '3shot-lol-ft.yaml'))
    
    backbone = UNet2DModel(
        in_channels=10, 
        out_channels=3,
        block_out_channels=cfg.model.channels, 
        layers_per_block=cfg.model.layers_per_block,
        down_block_types=tuple(cfg.model.downblock for _ in range(len(cfg.model.channels))),
        up_block_types=tuple(cfg.model.upblock for _ in range(len(cfg.model.channels))),
        add_attention=cfg.model.add_attention, 
        attention_head_dim=cfg.model.attention_head_dim,
    ).to(DEVICE)
    
    ckpt_path = os.path.join(WORKSPACE, 'Models', 'DIMMA_MODELS', 'mdn-mixhq-3shot-ft.ckpt')
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    backbone_sd = {k.replace('backbone.', ''): v for k, v in sd.items() if k.startswith('backbone.')}
    
    remapped = {}
    for k, v in backbone_sd.items():
        k2 = k
        if '.query.' in k: k2 = k.replace('.query.', '.to_q.')
        elif '.key.' in k: k2 = k.replace('.key.', '.to_k.')
        elif '.value.' in k: k2 = k.replace('.value.', '.to_v.')
        elif '.proj_attn.' in k: k2 = k.replace('.proj_attn.', '.to_out.0.')
        remapped[k2] = v
        
    backbone.load_state_dict(remapped)
    backbone.eval()
    return backbone

def exec_dimma(net, img_pil):
    img_np = np.array(img_pil.convert('RGB'))
    
    norm_img = img_np / 255.0
    R = norm_img / (norm_img.sum(axis=2, keepdims=True) + 1e-6)
    L = (norm_img / (3 * R + 1e-6)).max(axis=2)
    
    ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    hist_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB) / 255.0
    
    feat = np.concatenate([norm_img, hist_eq, R, L[:, :, None]], axis=2).transpose(2, 0, 1)
    feat_t = torch.from_numpy(feat).float().unsqueeze(0).to(DEVICE)
    
    source_lightness = torch.tensor([L.mean()]).float().to(DEVICE)
    target_lightness = torch.tensor([0.65]).float().to(DEVICE)
    light_diff = (1000 * (target_lightness - source_lightness)).long()
    
    h, w = feat_t.shape[2], feat_t.shape[3]
    h_pad = ((h + 15) // 16) * 16
    w_pad = ((w + 15) // 16) * 16
    padded_feat = F.pad(feat_t, (0, w_pad - w, 0, h_pad - h), mode='reflect')
    
    with torch.no_grad():
        out = net(padded_feat, timestep=light_diff).sample
        # Dimma dim head: clamp(y + x[:, :3], 0, 1)
        enhanced = torch.clamp(out[:, :, :h, :w] + feat_t[:, :3, :, :], 0, 1)
        
    out_np = (enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(out_np)


# ==========================================
# Main Execution Pipeline
# ==========================================
def main():
    print("=" * 80)
    print("  AUTOMATED MULTI-MODEL LOW-LIGHT IMAGE ENHANCEMENT BENCHMARK")
    print("=" * 80)
    
    # 1. Discover all test images (PNG, JPG, JPEG)
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
    test_images = [f for f in all_files if f.lower().endswith(valid_exts)]
    
    print(f"\n[Dataset] Loaded {len(test_images)} ExDark test images from {DATASET_DIR}:")
    for p in test_images:
        print(f"  • {os.path.basename(p)}")

    # 2. Register all model instances
    print("\n[Loading Models] Initializing neural network weights...")
    
    model_suite = []
    
    # DL Models
    dl_definitions = [
        ("Zero-DCE", get_zero_dce, exec_zero_dce),
        ("Zero-DCE++", get_zero_dce_pp, exec_zero_dce_pp),
        ("RUAS_lol", lambda: get_ruas("lol"), lambda m, img: exec_ruas(m, img, "lol")),
        ("RUAS_dark", lambda: get_ruas("dark"), lambda m, img: exec_ruas(m, img, "dark")),
        ("RUAS_upe", lambda: get_ruas("upe"), lambda m, img: exec_ruas(m, img, "upe")),
        ("SCI_CVPR_easy", lambda: get_sci_cvpr("easy"), exec_sci),
        ("SCI_CVPR_medium", lambda: get_sci_cvpr("medium"), exec_sci),
        ("SCI_CVPR_difficult", lambda: get_sci_cvpr("difficult"), exec_sci),
        ("SCI_TPAMI", get_sci_tpami, exec_sci),
        ("Dimma", get_dimma, exec_dimma),
        ("DRBN_stage1", get_drbn_stage1, exec_drbn),
        ("CUE", get_cue, exec_cue),
    ]
    
    for name, getter, runner in dl_definitions:
        try:
            m = getter()
            model_suite.append((name, m, runner, "Deep Learning"))
            print(f"  [✓] Successfully loaded {name}")
        except Exception as e:
            print(f"  [✗] Failed loading {name}: {e}")

    # Classical baselines
    model_suite.append(("Classic_HE", None, lambda _, img: run_classic_he(img), "Classical"))
    model_suite.append(("Classic_CLAHE", None, lambda _, img: run_classic_clahe(img), "Classical"))
    model_suite.append(("Classic_Retinex", None, lambda _, img: run_classic_retinex(img), "Classical"))
    print(f"  [✓] Registered Classical Baselines: HE, CLAHE, Retinex")

    print(f"\n[Execution] Running {len(model_suite)} algorithms on {len(test_images)} dataset images...")

    results_data = {
        "device": DEVICE,
        "total_algorithms": len(model_suite),
        "total_images": len(test_images),
        "summary": {},
        "per_image": {}
    }

    # 3. Iterate Images
    for img_path in test_images:
        img_name = os.path.basename(img_path)
        img_stem = os.path.splitext(img_name)[0]
        raw_pil = Image.open(img_path).convert('RGB')
        raw_metrics = calculate_metrics(np.array(raw_pil))
        
        print("\n" + "=" * 70)
        print(f"Image: {img_name} ({raw_pil.size[0]}x{raw_pil.size[1]})")
        print(f"  Original: Brightness={raw_metrics['mean_brightness']}, Contrast={raw_metrics['contrast_std']}, Entropy={raw_metrics['entropy']}, DynRange={raw_metrics['dynamic_range']}")
        print("-" * 70)
        
        results_data["per_image"][img_name] = {
            "original_metrics": raw_metrics,
            "algorithms": {}
        }
        
        panels_for_grid = [("Original (Input)", raw_pil)]
        for algo_name, model_obj, runner, category in model_suite:
            img_subfolder = os.path.join(OUTPUT_BASE, algo_name, img_stem)
            os.makedirs(img_subfolder, exist_ok=True)
            clean_raw = Image.fromarray(np.array(raw_pil.convert('RGB')))
            clean_raw.save(os.path.join(img_subfolder, "original.png"), "PNG")
                
            try:
                # GPU sync for accurate timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                
                enhanced = runner(model_obj, raw_pil)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                
                clean_enh = Image.fromarray(np.array(enhanced.convert('RGB')))
                clean_enh.save(os.path.join(img_subfolder, f"{algo_name}.png"), "PNG")
                
                enh_metrics = calculate_metrics(np.array(enhanced))
                gain = round(enh_metrics["mean_brightness"] / max(raw_metrics["mean_brightness"], 1.0), 2)
                
                results_data["per_image"][img_name]["algorithms"][algo_name] = {
                    "category": category,
                    "latency_ms": round(latency_ms, 2),
                    "mean_brightness": enh_metrics["mean_brightness"],
                    "contrast_std": enh_metrics["contrast_std"],
                    "entropy": enh_metrics["entropy"],
                    "dynamic_range": enh_metrics["dynamic_range"],
                    "brightness_gain_x": gain,
                    "subfolder": img_subfolder
                }
                
                print(f"  [✓] {algo_name:<18} | Time: {latency_ms:6.1f} ms | Brightness: {enh_metrics['mean_brightness']:5.1f} ({gain}x) | Contrast: {enh_metrics['contrast_std']:4.1f} | Entropy: {enh_metrics['entropy']:4.2f}")
                    
            except Exception as e:
                print(f"  [✗] {algo_name:<18} | FAILED: {e}")
                results_data["per_image"][img_name]["algorithms"][algo_name] = {"error": str(e)}

    # 4. Global Summary Aggregation
    for algo_name, _, _, category in model_suite:
        latencies, gains, entropies, contrasts, dyn_ranges = [], [], [], [], []
        for img_name in results_data["per_image"]:
            res = results_data["per_image"][img_name]["algorithms"].get(algo_name, {})
            if "latency_ms" in res:
                latencies.append(res["latency_ms"])
                gains.append(res["brightness_gain_x"])
                entropies.append(res["entropy"])
                contrasts.append(res["contrast_std"])
                dyn_ranges.append(res["dynamic_range"])
                
        if latencies:
            results_data["summary"][algo_name] = {
                "category": category,
                "avg_latency_ms": round(float(np.mean(latencies)), 2),
                "avg_brightness_gain_x": round(float(np.mean(gains)), 2),
                "avg_contrast_std": round(float(np.mean(contrasts)), 2),
                "avg_entropy": round(float(np.mean(entropies)), 2),
                "avg_dynamic_range": round(float(np.mean(dyn_ranges)), 2),
                "success_rate": f"{len(latencies)}/{len(test_images)}"
            }

    print("\n" + "=" * 80)
    print("                     ALL CODEBASES & MODELS BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Algorithm':<20} | {'Type':<12} | {'Latency':<11} | {'Gain':<8} | {'Contrast':<10} | {'Entropy':<8} | {'Status':<6}")
    print("-" * 80)
    for algo, s in results_data["summary"].items():
        print(f"{algo:<20} | {s['category']:<12} | {s['avg_latency_ms']:>7.1f} ms | {s['avg_brightness_gain_x']:>6.2f}x | {s['avg_contrast_std']:>8.1f} | {s['avg_entropy']:>6.2f} | {s['success_rate']}")
    print("=" * 80)
    print(f"\n[✓] Results, images, and visual comparison grids saved to: {OUTPUT_BASE}")

if __name__ == "__main__":
    main()
