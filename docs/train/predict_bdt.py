#!/bin/env python3
import uproot
import numpy as np
import os

# Configuration
flav = 'muon'
year = '2022'
nw = 'bdt'

ROOT_FILE = f"/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/preprocess/{flav}_{year}_test_0.root"
OUTPUT_FILE = f"predict/{flav}_{year}_{nw}.root"
TREE_NAME = "tree"
MAX_EVENTS = 20000

if not os.path.exists("predict"):
    os.makedirs("predict")

print(f"Opening {ROOT_FILE}...")
with uproot.open(ROOT_FILE) as file:
    tree = file[TREE_NAME]
    
    # Identify labels to extract
    label_branches = ["label_prompt_mclass", "label_tau_mclass", 
                      "label_heavy_mclass", "label_light_mclass", "label_fake_mclass"]
    
    # Only grab branches that actually exist in the file
    available_branches = [b for b in label_branches + ["Lepton_promptMVA"] if b in tree]
    
    print(f"Reading {len(available_branches)} branches for {MAX_EVENTS} events...")
    data = tree.arrays(available_branches, library="np", entry_stop=MAX_EVENTS)

# Map to your existing output structure
output_data = {
    "score_prompt": data["Lepton_promptMVA"],
    "label_prompt": data["label_prompt_mclass"].astype(np.int32)
}

# Add other labels if they exist to match your ParticleNet output format
for lb in label_branches[1:]:
    if lb in data:
        output_data[lb.replace("_mclass", "")] = data[lb].astype(np.int32)

print(f"Saving BDT results to {OUTPUT_FILE}...")
with uproot.recreate(OUTPUT_FILE) as fout:
    fout["tree"] = output_data

print("Done!")
