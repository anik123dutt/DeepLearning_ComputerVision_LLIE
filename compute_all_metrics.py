#!/usr/bin/env python3
"""
Comprehensive Comparative Study of Low-Light Image Enhancement (LLIE) Algorithms
Evaluates 15 algorithms across 3 test images using:
- PSNR     (Peak Signal-to-Noise Ratio) [dB, Higher is better]
- SSIM     (Structural Similarity Index Measure) [0 to 1, Higher is better]
- RGB_SSIM (Mean Structural Similarity across R, G, B channels) [0 to 1, Higher is better]
- LPIPS    (Learned Perceptual Image Patch Similarity) [Lower is better]
- DeltaE   (CIEDE2000 Perceptual Color Difference in CIELAB) [Lower is better]
- NIQE     (Natural Image Quality Evaluator) [Lower is better]
- EME      (Enhancement Measure / Measure of Enhancement) [Higher is better]
- DE       (Discrete Shannon Entropy) [Bits, Higher is better]
- LOE      (Lightness Order Error) [Lower is better]
"""

import os
import sys
import glob
import math
import shutil
import numpy as np
import pandas as pd
import cv2
import torch
import lpips
import skimage.color
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

# Add CUE codebase for official NIQE implementation
CUE_PATH = "/home/intern/internship_projects/Aniket/codebases/CUE-master"
if CUE_PATH not in sys.path:
    sys.path.insert(0, CUE_PATH)
from basicsr.metrics.niqe import calculate_niqe

WORKSPACE = "/home/intern/internship_projects/Aniket"
OUTPUT_BASE = os.path.join(WORKSPACE, "output")

# Map algorithms to their technological category
CATEGORY_MAP = {
    "Classic_HE": "Classical (Histogram Equalization)",
    "Classic_CLAHE": "Classical (Adaptive Histogram Equalization)",
    "Classic_Retinex": "Classical (Multi-Scale Retinex)",
    "Zero-DCE": "Zero-Reference Deep Learning (Curve Estimation)",
    "Zero-DCE++": "Zero-Reference Deep Learning (Depthwise Separable)",
    "RUAS_lol": "Unsupervised NAS Retinex (LOL model)",
    "RUAS_dark": "Unsupervised NAS Retinex (Dark model)",
    "RUAS_upe": "Unsupervised NAS Retinex (UPE model)",
    "SCI_CVPR_easy": "Self-Calibrated Illumination (CVPR Easy)",
    "SCI_CVPR_medium": "Self-Calibrated Illumination (CVPR Medium)",
    "SCI_CVPR_difficult": "Self-Calibrated Illumination (CVPR Difficult)",
    "SCI_TPAMI": "Self-Calibrated Illumination (TPAMI 2025)",
    "Dimma": "Semi-Supervised Deep Learning (Adaptive Dimming)",
    "DRBN_stage1": "Semi-Supervised Deep Learning (Recursive Band Network)",
    "CUE": "Customized Learnable Prior (ICCV 2023)"
}

def compute_de(img_bgr):
    """Calculate Discrete Shannon Entropy (DE) on 8-bit grayscale intensity."""
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    prob = hist / (hist.sum() + 1e-12)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))

def compute_eme(img_bgr, block_size=8, eps=1e-4):
    """
    Calculate Enhancement Measure (EME) [Agaian et al., 2007].
    EME = (1 / (k1*k2)) * sum(20 * ln(I_max / (I_min + eps)))
    """
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = img_bgr.astype(np.float64)
    h, w = gray.shape
    nb_h = h // block_size
    nb_w = w // block_size
    if nb_h == 0 or nb_w == 0:
        return 0.0
    cropped = gray[:nb_h * block_size, :nb_w * block_size]
    blocks = cropped.reshape(nb_h, block_size, nb_w, block_size).transpose(0, 2, 1, 3).reshape(-1, block_size * block_size)
    max_v = np.max(blocks, axis=1)
    min_v = np.min(blocks, axis=1)
    
    ratios = np.ones_like(max_v)
    valid = max_v > 0
    ratios[valid] = max_v[valid] / (min_v[valid] + eps)
    eme = float(np.mean(20.0 * np.log(ratios)))
    return eme

def compute_loe(img_ref_bgr, img_enh_bgr, size=100):
    """
    Calculate Lightness Order Error (LOE) [Wang et al., IEEE TIP 2013].
    Measures the degree of relative lightness order distortion.
    Lower LOE indicates better natural lightness order preservation.
    """
    L_ref = np.max(img_ref_bgr, axis=2).astype(np.float32)
    L_enh = np.max(img_enh_bgr, axis=2).astype(np.float32)
    h, w = L_ref.shape
    scale = size / min(h, w)
    if scale < 1.0:
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        L_ref = cv2.resize(L_ref, (new_w, new_h), interpolation=cv2.INTER_AREA)
        L_enh = cv2.resize(L_enh, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
    t_ref = torch.from_numpy(L_ref.flatten())
    t_enh = torch.from_numpy(L_enh.flatten())
    if torch.cuda.is_available():
        t_ref = t_ref.cuda()
        t_enh = t_enh.cuda()
        
    U_ref = t_ref.unsqueeze(1) >= t_ref.unsqueeze(0)
    U_enh = t_enh.unsqueeze(1) >= t_enh.unsqueeze(0)
    xor = U_ref ^ U_enh
    loe = float((xor.sum().float() / t_ref.numel()).item())
    return loe

def compute_rgb_ssim(img_ref_rgb, img_enh_rgb):
    """Calculate mean SSIM across R, G, and B color channels."""
    ssim_r = calc_ssim(img_ref_rgb[:, :, 0], img_enh_rgb[:, :, 0], data_range=255)
    ssim_g = calc_ssim(img_ref_rgb[:, :, 1], img_enh_rgb[:, :, 1], data_range=255)
    ssim_b = calc_ssim(img_ref_rgb[:, :, 2], img_enh_rgb[:, :, 2], data_range=255)
    return float((ssim_r + ssim_g + ssim_b) / 3.0)

def compute_delta_e(img_ref_rgb, img_enh_rgb):
    """Calculate CIEDE2000 perceptual color difference in CIELAB color space."""
    lab0 = skimage.color.rgb2lab(img_ref_rgb.astype(np.float32) / 255.0)
    lab1 = skimage.color.rgb2lab(img_enh_rgb.astype(np.float32) / 255.0)
    dE = skimage.color.deltaE_ciede2000(lab0, lab1)
    return float(np.mean(dE))

def compute_lpips(img_ref_rgb, img_enh_rgb, loss_fn):
    """Calculate LPIPS perceptual distance using AlexNet feature activations."""
    t0 = torch.from_numpy(img_ref_rgb).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
    t1 = torch.from_numpy(img_enh_rgb).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
    if torch.cuda.is_available():
        t0 = t0.cuda()
        t1 = t1.cuda()
    with torch.no_grad():
        val = loss_fn(t0, t1).item()
    return float(val)

def main():
    algos = sorted([d for d in os.listdir(OUTPUT_BASE) if os.path.isdir(os.path.join(OUTPUT_BASE, d))])
    test_images = ["2015_00003", "2015_02446", "2015_06400"]
    
    print(f"==================================================================")
    print(f" LLIE COMPREHENSIVE BENCHMARKING ENGINE")
    print(f" Metrics: PSNR, SSIM, RGB-SSIM, LPIPS, DeltaE, NIQE, EME, DE, LOE")
    print(f" Evaluating {len(algos)} algorithms across {len(test_images)} test images")
    print(f"==================================================================")

    # Initialize LPIPS model
    print("[Init] Loading LPIPS AlexNet metric network...")
    lpips_loss_fn = lpips.LPIPS(net='alex', verbose=False)
    if torch.cuda.is_available():
        lpips_loss_fn = lpips_loss_fn.cuda()

    records = []
    
    # First: evaluate the Original (Input) image baselines
    for img_id in test_images:
        sample_folder = os.path.join(OUTPUT_BASE, algos[0], img_id)
        orig_p = os.path.join(sample_folder, "original.png")
        img_orig_bgr = cv2.imread(orig_p)
        
        niqe_val = calculate_niqe(img_orig_bgr, crop_border=0, input_order='HWC', convert_to='y')
        eme_val = compute_eme(img_orig_bgr)
        de_val = compute_de(img_orig_bgr)
        
        records.append({
            "Algorithm": "Original (Input Baseline)",
            "Category": "Reference Ground / Input",
            "Image": img_id,
            "PSNR_dB": float("inf"),
            "SSIM": 1.0000,
            "RGB_SSIM": 1.0000,
            "LPIPS": 0.0000,
            "DeltaE": 0.00,
            "NIQE": round(niqe_val, 3),
            "EME": round(eme_val, 3),
            "DE": round(de_val, 3),
            "LOE": 0.00
        })

    for algo in algos:
        category = CATEGORY_MAP.get(algo, "Deep Learning")
        print(f"\nProcessing Algorithm: {algo:<20} [{category}]")
        
        algo_metrics = []
        for img_id in test_images:
            folder = os.path.join(OUTPUT_BASE, algo, img_id)
            orig_p = os.path.join(folder, "original.png")
            enh_p = os.path.join(folder, f"{algo}.png")
            
            if not os.path.exists(orig_p) or not os.path.exists(enh_p):
                print(f"  [!] Missing file for {algo} on {img_id}")
                continue
                
            img_orig_bgr = cv2.imread(orig_p)
            img_enh_bgr = cv2.imread(enh_p)
            
            # Handle minor spatial dimension differences if model cropped (e.g. Zero-DCE++ scale_factor=12)
            if img_orig_bgr.shape != img_enh_bgr.shape:
                min_h = min(img_orig_bgr.shape[0], img_enh_bgr.shape[0])
                min_w = min(img_orig_bgr.shape[1], img_enh_bgr.shape[1])
                img_orig_bgr_eval = img_orig_bgr[:min_h, :min_w]
                img_enh_bgr_eval = img_enh_bgr[:min_h, :min_w]
            else:
                img_orig_bgr_eval = img_orig_bgr
                img_enh_bgr_eval = img_enh_bgr
                
            img_orig_rgb_eval = cv2.cvtColor(img_orig_bgr_eval, cv2.COLOR_BGR2RGB)
            img_enh_rgb_eval = cv2.cvtColor(img_enh_bgr_eval, cv2.COLOR_BGR2RGB)
            
            # 1. PSNR
            psnr_val = calc_psnr(img_orig_bgr_eval, img_enh_bgr_eval, data_range=255)
            # 2. SSIM (standard multichannel)
            ssim_val = calc_ssim(img_orig_bgr_eval, img_enh_bgr_eval, channel_axis=2, data_range=255)
            # 3. RGB-SSIM (mean across R, G, B)
            rgb_ssim_val = compute_rgb_ssim(img_orig_rgb_eval, img_enh_rgb_eval)
            # 4. LPIPS
            lpips_val = compute_lpips(img_orig_rgb_eval, img_enh_rgb_eval, lpips_loss_fn)
            # 5. DeltaE (CIEDE2000)
            delta_e_val = compute_delta_e(img_orig_rgb_eval, img_enh_rgb_eval)
            # 6. NIQE
            niqe_val = calculate_niqe(img_enh_bgr, crop_border=0, input_order='HWC', convert_to='y')
            # 7. EME
            eme_val = compute_eme(img_enh_bgr)
            # 8. DE
            de_val = compute_de(img_enh_bgr)
            # 9. LOE
            loe_val = compute_loe(img_orig_bgr_eval, img_enh_bgr_eval)
            
            row = {
                "Algorithm": algo,
                "Category": category,
                "Image": img_id,
                "PSNR_dB": round(float(psnr_val), 3),
                "SSIM": round(float(ssim_val), 4),
                "RGB_SSIM": round(float(rgb_ssim_val), 4),
                "LPIPS": round(float(lpips_val), 4),
                "DeltaE": round(float(delta_e_val), 2),
                "NIQE": round(float(niqe_val), 3),
                "EME": round(float(eme_val), 3),
                "DE": round(float(de_val), 3),
                "LOE": round(float(loe_val), 2)
            }
            records.append(row)
            algo_metrics.append(row)
            print(f"  • {img_id}: PSNR={row['PSNR_dB']:5.2f} | RGB-SSIM={row['RGB_SSIM']:.4f} | LPIPS={row['LPIPS']:.4f} | DeltaE={row['DeltaE']:5.2f} | NIQE={row['NIQE']:5.2f} | LOE={row['LOE']:6.2f}")
            
        # Add summary row for this algorithm
        if algo_metrics:
            avg_row = {
                "Algorithm": algo,
                "Category": category,
                "Image": "Average (All 3 Images)",
                "PSNR_dB": round(float(np.mean([m["PSNR_dB"] for m in algo_metrics])), 3),
                "SSIM": round(float(np.mean([m["SSIM"] for m in algo_metrics])), 4),
                "RGB_SSIM": round(float(np.mean([m["RGB_SSIM"] for m in algo_metrics])), 4),
                "LPIPS": round(float(np.mean([m["LPIPS"] for m in algo_metrics])), 4),
                "DeltaE": round(float(np.mean([m["DeltaE"] for m in algo_metrics])), 2),
                "NIQE": round(float(np.mean([m["NIQE"] for m in algo_metrics])), 3),
                "EME": round(float(np.mean([m["EME"] for m in algo_metrics])), 3),
                "DE": round(float(np.mean([m["DE"] for m in algo_metrics])), 3),
                "LOE": round(float(np.mean([m["LOE"] for m in algo_metrics])), 2)
            }
            records.append(avg_row)

    df_full = pd.DataFrame(records)
    
    # Save to all target paths
    primary_full_path = os.path.join(WORKSPACE, "llie_metrics_comparative_study.csv")
    df_full.to_csv(primary_full_path, index=False)
    
    # Summary table ranked by NIQE
    df_avg = df_full[df_full["Image"] == "Average (All 3 Images)"].copy()
    df_avg.sort_values(by="NIQE", inplace=True)
    summary_path = os.path.join(WORKSPACE, "llie_metrics_summary_table.csv")
    df_avg.to_csv(summary_path, index=False)

    # Replicate to all user-facing aliases in root and output/
    destinations = [
        os.path.join(WORKSPACE, "metrics.csv"),
        os.path.join(WORKSPACE, "comparison.csv"),
        os.path.join(OUTPUT_BASE, "llie_metrics_comparative_study.csv"),
        os.path.join(OUTPUT_BASE, "llie_metrics_summary_table.csv"),
        os.path.join(OUTPUT_BASE, "metrics.csv"),
        os.path.join(OUTPUT_BASE, "comparison.csv")
    ]
    
    for dst in destinations:
        if "summary" in dst:
            shutil.copyfile(summary_path, dst)
        else:
            shutil.copyfile(primary_full_path, dst)
        print(f"[✓] Updated: {dst}")

    print("\n[✓] All CSV files successfully updated with LPIPS, DeltaE, RGB-SSIM, and all existing metrics!")

if __name__ == "__main__":
    main()
