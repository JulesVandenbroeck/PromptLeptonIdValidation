#!/bin/env python3

import uproot

# Open the file and access the tree
tree = uproot.open("/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/preprocess/electron_2022_train_0.root")["tree"]

# Define your labels
labels = [
    "label_prompt_mclass", "label_heavy_mclass", "label_light_mclass", 
    "label_tau_mclass", "label_fake_mclass"
]

# Load as a dataframe or dictionary and sum the '1's
data = tree.arrays(labels, library="np")

total_entries = len(data[labels[0]])
print(f"Total entries: {total_entries}\n")

for label in labels:
    count = sum(data[label])
    fraction = count / total_entries
    print(f"{label:20}: {count:>8} events ({fraction:.2%})")
