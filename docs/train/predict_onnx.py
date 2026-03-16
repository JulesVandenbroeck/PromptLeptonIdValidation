#!/bin/env python3

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

import os, sys, json, gc
import uproot
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Styling
hep.style.use("CMS")
if not os.path.exists("pics"): os.makedirs("pics")

# --- Configuration ---
flav = 'muon'
year = '2022'
nw = 'particletransformer' 
#MAX_EVENTS = 100000
#MAX_EVENTS = 20000
MAX_EVENTS = 1
CHUNK_SIZE = 1  # Larger chunks are faster with vectorization

MODEL_PATH = f"models/model_{nw}_{flav}_{year}_mclass_optimal.onnx"
PREPROCESS_JSON = f"models/preprocess_{nw}_{flav}_{year}_mclass_optimal.json"

FILES = {
    "test":  f"/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/preprocess/{flav}_{year}_test_0.root",
    "train": f"/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/preprocess/{flav}_{year}_train_0.root"
}
BDT_FILE = f"predict/{flav}_{year}_bdt.root"

# Input configuration based on network architecture
if nw == 'particlenet':
    INPUT_GROUPS = {
        "pf_points": 60, "pf_features": 60, "pf_mask": 60,
        "sv_points": 5,  "sv_features": 5,  "sv_mask": 5,
        "high_level": 1
    }
else: # particletransformer
    INPUT_GROUPS = {
        "pf_features": 60, "pf_points": 60, "pf_vectors": 60, "pf_mask": 60,
        "sv_features": 5,  "sv_points": 5, "sv_vectors": 5,  "sv_mask": 5,
        "highlevel": 1
    }

def prepare_input_tensors(chunk_data, prep_data, groups, n_entries, model_meta):
    """Vectorized preparation: handles padding and normalization in bulk."""
    input_tensors = {}

    for group, max_len in groups.items():
        if group not in model_meta: continue
        
        info_group = prep_data.get(group, {})
        var_names = info_group.get("var_names", [])
        var_infos = info_group.get("var_infos", {})
        
        # (Batch, Channels, Length)
        tensor = np.zeros((n_entries, len(var_names), max_len), dtype=np.float32)
        
        for i, var in enumerate(var_names):
            if var not in chunk_data: continue
            raw = chunk_data[var]
            info = var_infos.get(var, {})
            
            # Fast padding
            padded = np.zeros((n_entries, max_len), dtype=np.float32)
            for j, ev in enumerate(raw):
                v = np.atleast_1d(ev)[:max_len]
                padded[j, :len(v)] = v
            
            # Vectorized Math (In-place to save memory)
            padded = np.nan_to_num(padded, nan=info.get("replace_inf_value", 0.0))
            padded -= info.get("median", 0.0)
            padded *= info.get("norm_factor", 1.0)
            np.clip(padded, info.get("lower_bound", -np.inf), info.get("upper_bound", np.inf), out=padded)
            
            tensor[:, i, :] = padded

        # Shape adjustment for ParticleTransformer
        expected_shape = model_meta[group]
        if group == "highlevel" and len(expected_shape) == 2:
            input_tensors[group] = tensor.squeeze(axis=2) # (Batch, Channels)
        elif len(expected_shape) == 3 and expected_shape[1] == max_len:
            input_tensors[group] = np.transpose(tensor, (0, 2, 1)) # (Batch, Length, Channels)
        else:
            input_tensors[group] = tensor
            
    return input_tensors

# --- 1. Session Setup ---
with open(PREPROCESS_JSON, 'r') as f:
    prep_data = json.load(f)

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 8
#opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
model_meta = {i.name: i.shape for i in session.get_inputs()}

# Check for fixed batch size (e.g., if model expects [1, C, N])
is_fixed_batch = any(isinstance(s, int) and s > 0 for s in next(iter(model_meta.values()))[:1])
batch_dim = 1 if is_fixed_batch else CHUNK_SIZE

results = {}

# --- 2. Inference Loop ---
for mode, path in FILES.items():
    print(f"Processing {mode} file...")
    
    scores_list, truth_list, y_true_list = [], [], []
    cursor = 0

    for chunk in uproot.iterate(f"{path}:tree", step_size=CHUNK_SIZE, entry_stop=MAX_EVENTS, library="np"):
        n_ev = len(chunk[list(chunk.keys())[0]])
        
        # Prepare Tensors
        inputs = prepare_input_tensors(chunk, prep_data, INPUT_GROUPS, n_ev, model_meta)
        
        # Inference (Handle fixed vs dynamic batching)
        if is_fixed_batch:
            # If model is fixed at Batch=1, we must loop.
            raw_scores = []
            for i in range(n_ev):
                single_in = {k: v[i:i+1] for k, v in inputs.items()}
                raw_scores.append(session.run(None, single_in)[0])
            raw_scores = np.concatenate(raw_scores, axis=0)
        else:
            raw_scores = session.run(None, inputs)[0]
        
        scores_list.append(raw_scores)
        
        # Labels
        l_p, l_t = chunk["label_prompt_mclass"], chunk["label_tau_mclass"]
        truth_list.append(((l_p == 1) | (l_t == 1)).astype(np.int8))
        
        if mode == "test":
            lbl_vars = ["label_prompt_mclass", "label_tau_mclass", "label_heavy_mclass", "label_light_mclass", "label_fake_mclass"]
            combined = np.stack([chunk[l] for l in lbl_vars], axis=1)
            y_true_list.append(np.argmax(combined, axis=1).astype(np.int8))

        cursor += n_ev
        del chunk, inputs, raw_scores
        gc.collect()

    # Consolidate results
    results[mode] = {
        "scores": np.concatenate(scores_list),
        "truth": np.concatenate(truth_list)
    }
    if mode == "test":
        results["y_true"] = np.concatenate(y_true_list)
        results["cm"] = confusion_matrix(results["y_true"], np.argmax(results[mode]["scores"], axis=1))

# --- 3. BDT & Plotting ---
if os.path.exists(BDT_FILE):
    with uproot.open(BDT_FILE) as f:
        t = f["tree"]
        b_score = t["score_prompt"].array(library="np")
        b_truth = ((t["label_prompt"].array(library="np") == 1) | (t["label_tau"].array(library="np") == 1)).astype(np.int8)
        fpr_b, tpr_b, _ = roc_curve(b_truth, b_score)
        results["bdt"] = {"fpr": fpr_b, "tpr": tpr_b, "auc": auc(fpr_b, tpr_b)}

# Plot ROC
plt.figure(figsize=(10, 10))
for m in ["train", "test"]:
    if m in results:
        res = results[m]
        sig_score = res["scores"][:, 0] + res["scores"][:, 1]
        fpr, tpr, _ = roc_curve(res["truth"], sig_score)
        plt.plot(fpr, tpr, label=f"PNet {m} (AUC={auc(fpr, tpr):.4f})")

if "bdt" in results:
    plt.plot(results["bdt"]["fpr"], results["bdt"]["tpr"], label=f"BDT (AUC={results['bdt']['auc']:.4f})", color='red')

plt.yscale('log'); plt.xscale('log'); plt.xlim(1e-3, 1); plt.ylim(0.1, 1)
plt.xlabel("Background efficiency"); plt.ylabel("Signal efficiency")
plt.legend(); plt.grid(True, which="both", ls="-")
plt.savefig("pics/roc_curve.pdf")

# Plot CM
if "cm" in results:
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=results["cm"], display_labels=CLASS_NAMES)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.savefig("pics/confusion_matrix.pdf")

print("Done! Plots saved in pics/")
