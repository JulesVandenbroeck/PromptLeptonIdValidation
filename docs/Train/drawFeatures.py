#!/bin/env python3
import os, sys, glob
import ROOT as r
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# Set batch mode for ROOT so it doesn't interfere
r.gROOT.SetBatch(True)

try:
    hep.style.use("CMS") 
except:
    print("mplhep not installed or style not available, using default matplotlib style.")

nmax = 200000
vars = ["SV_dlenSig", "PF_eta_rel", "PF_phi_rel", "SV_eta_rel", "PF_dxySig", "PF_pt", "SV_dxy"]

fdir = '/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8_Run3Summer22_NanoAODv15_PNet_LeptonID_20251215/251213_205345/0000'
ff = glob.glob(fdir+"/tree_*.root")[:100]

trinput = r.TChain('ntuplizer_electron/Events')
for f in ff: trinput.Add(f)

df = r.RDataFrame(trinput)
if nmax < df.Count().GetValue(): 
    df = df.Range(0, nmax)

df = df.Define("PF_dr_rel", "sqrt(PF_eta_rel*PF_eta_rel + PF_phi_rel*PF_phi_rel)")
df = df.Define("SV_dr_rel", "sqrt(SV_eta_rel*SV_eta_rel + SV_phi_rel*SV_phi_rel)")

vars += ["PF_dr_rel", "SV_dr_rel"]

dfcl = {
    "prompt": df.Filter("(genPartFlav == 1) || (genPartFlav == 15)"),
    "nonprompt": df.Filter("(genPartFlav != 1) && (genPartFlav != 15)")
}

colors = {"prompt": 'tab:red', "nonprompt": 'tab:blue'}
BINNING = 70

for v in vars:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data_arrays = {}
    all_data_combined = []

    for label, dff in dfcl.items():
        npy_data = dff.AsNumpy(columns=[v])
        arr = npy_data[v]
        
        # --- ROBUST FIX: Explicitly extract every float into a flat list ---
        # This bypasses all issues with RVec types or complex NumPy masking
        flat_list_of_floats = []
        for event_data in arr:
            # event_data should now be iterable (list or RVec of floats)
            for value in event_data:
                if value != 0:
                    flat_list_of_floats.append(float(value))
                
        # Convert the pure Python list to a standard NumPy array
        data_arrays[label] = np.array(flat_list_of_floats, dtype=np.float64)
        all_data_combined.append(data_arrays[label])
    
    # Combine all data into a single, pure numpy float array for percentile calculation
    combined_flat = np.concatenate(all_data_combined)

    # Calculate 5th and 95th percentile (should now work perfectly)
    x_min = np.percentile(combined_flat, 5)
    x_max = np.percentile(combined_flat, 95)
    X_RANGE = (x_min, x_max)
#    X_RANGE = (-0.1, 0.1)
    
    # Matplotlib plotting
    for label in ["prompt", "nonprompt"]:
        data = data_arrays[label]
        
        ax.hist(data, bins=BINNING, range=X_RANGE, density=True, 
                histtype='step', label=label, color=colors[label], linewidth=2)
    
    ax.set_xlabel(v)
    ax.set_ylabel('Normalized to unity')
    ax.set_xlim(X_RANGE)
    ax.set_yscale('log')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(True, axis='both', linestyle='--', alpha=0.6)
    
    plt.tight_layout() 
    
    plt.savefig(f"pics/features_{v}.pdf")
    plt.close(fig)
