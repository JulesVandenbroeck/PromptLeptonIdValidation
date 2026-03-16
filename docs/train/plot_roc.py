#!/bin/env python3

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

import os, sys
import uproot
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.metrics import roc_curve, auc

hep.style.use("CMS")

if not os.path.exists("pics"):
    os.makedirs("pics")

# --- Configuration ---
flav = "muon" 
year = "2022"
WPS = [0.97, 0.93, 0.88, 0.80] 
TARGET_WP = 0.93 # Point for arrows

PART_FILE = f"predict/{flav}_{year}_particletransformer.root"
PNET_FILE = f"predict/{flav}_{year}_particlenet.root"
BDT_FILE  = f"predict/{flav}_{year}_bdt.root"

def get_roc_data(filepath, is_pnet=True):
    with uproot.open(filepath) as file:
        tree = file["tree"]
        label_p = tree["label_prompt"].array(library="np")
        label_t = tree["label_tau"].array(library="np")
        truth = ((label_p == 1) | (label_t == 1)).astype(int)

        if is_pnet:
            scores = tree["score_prompt"].array(library="np") + tree["score_tau"].array(library="np")
        else:
            scores = tree["score_prompt"].array(library="np")

        fpr, tpr, _ = roc_curve(truth, scores)
        return fpr, tpr

fpr_part, tpr_part = get_roc_data(PART_FILE, is_pnet=True)
fpr_pnet, tpr_pnet = get_roc_data(PNET_FILE, is_pnet=True)
fpr_bdt,  tpr_bdt  = get_roc_data(BDT_FILE,  is_pnet=False)

fig, ax = plt.subplots(figsize=(10, 8))

# Plot Curves
ax.plot(fpr_bdt, tpr_bdt, color='black', lw=1.5, label='PromptMVA')
ax.plot(fpr_pnet, tpr_pnet, color='orange', lw=1.5, linestyle='-.', label='PromptPNet')
ax.plot(fpr_part, tpr_part, color='red', lw=1.5, linestyle='--', label='PromptParT')

# Add Working Points (Pure circles, no tick lines)
dot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
for wp, color in zip(WPS, dot_colors):
    idx = np.argmin(np.abs(tpr_pnet - wp))
    ax.scatter(fpr_pnet[idx], tpr_pnet[idx], color=color, s=120, zorder=6)

# --- Dual Arrows at 93% WP ---
idx_pnet = np.argmin(np.abs(tpr_pnet - TARGET_WP))
idx_bdt  = np.argmin(np.abs(tpr_bdt - TARGET_WP))

fpr_p = fpr_pnet[idx_pnet]
fpr_b = fpr_bdt[idx_bdt]

# Horizontal Arrow (Teal)
ax.annotate('', xy=(fpr_p, TARGET_WP), xytext=(fpr_b, TARGET_WP),
            arrowprops=dict(arrowstyle='->', color='teal', lw=2, mutation_scale=20), zorder=7)
rel_bg = (fpr_p - fpr_b) / fpr_b * 100
ax.text(fpr_p * 0.35, TARGET_WP + 0.015, f'{rel_bg:+.0f}% nonprompt', 
        color='teal', fontsize=15, fontweight='bold')

# Vertical Arrow (Purple)
idx_bdt_fixed_fpr = np.argmin(np.abs(fpr_bdt - fpr_p))
tpr_b_at_p_fpr = tpr_bdt[idx_bdt_fixed_fpr]
abs_sig = (TARGET_WP - tpr_b_at_p_fpr) * 100

ax.annotate('', xy=(fpr_p, TARGET_WP), xytext=(fpr_p, tpr_b_at_p_fpr),
            arrowprops=dict(arrowstyle='->', color='purple', lw=2, mutation_scale=20), zorder=7)
ax.text(fpr_p * 0.35, TARGET_WP - 0.03, f'{abs_sig:+.0f}% prompt', 
        color='purple', fontsize=15, fontweight='bold')

# --- Formatting ---
ax.set_xscale('log')
ax.set_xlim(1e-3, 1.0)
ax.set_ylim(0.7, 1.0)
ax.set_xlabel('Nonprompt lepton efficiency', fontsize=22)
ax.set_ylabel('Prompt lepton efficiency', fontsize=22)

# Grid setup matching the reference style
ax.grid(True, which="major", linestyle='-', alpha=0.5, color='gray')
ax.grid(True, which="minor", linestyle=':', alpha=0.3, color='gray')

# Muon Label Bottom Right
ax.text(0.95, 0.05, 'Muon', fontsize=32, transform=ax.transAxes, 
        verticalalignment='bottom', horizontalalignment='right')

hep.cms.label(ax=ax, label="Preliminary", data=True, year=year)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=18)

plt.tight_layout()
output_path = "pics/roc.pdf"
plt.savefig(output_path)
print(f"Final plot saved to {output_path}")
