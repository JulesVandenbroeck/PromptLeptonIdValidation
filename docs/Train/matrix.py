#!/bin/env python3

import os, sys
import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal for")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)
import uproot
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fName = 'prompt_heavy_light_fake'
#fName = 'prompt_tau_heavy_lightfake'
#fName = 'prompt_heavy_lightfake'

#file = uproot.open("predict/predict_muon_2022_TOP_prompt_tau_heavy_lightfake.root")
#file = uproot.open("predict/predict_muon_2022_TOP_"+fName+"_tau.root")
file = uproot.open("predict/predict_muon_2022_TOP_"+fName+".root")
#file = uproot.open("predict/predict_particlenet_muon_2022_tau.root")
#file = uproot.open("predict/predict_muon_cms_multiclass.root")
tree = file["Events"]

if 'prompt_heavy_light_fake' in fName:
    truth_data = tree.arrays(["label_prompt", "label_heavy", "label_light", "label_fake"], library="np")
    y_true_matrix = np.stack([
        truth_data["label_prompt"],
        truth_data["label_heavy"], 
        truth_data["label_light"],
        truth_data["label_fake"]
    ], axis=1)
    true_labels = np.argmax(y_true_matrix, axis=1)
    score_data = tree.arrays(["score_label_prompt", "score_label_heavy", "score_label_light", "score_label_fake"], library="np")
    y_pred = np.argmax(np.stack([
        score_data["score_label_prompt"],
        score_data["score_label_heavy"], 
        score_data["score_label_light"],
        score_data["score_label_fake"]
    ], axis=1), axis=1)
    cm = confusion_matrix(true_labels, y_pred, normalize=None)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=['Prompt', 'Heavy', 'Light', 'Fake']
    )
elif 'prompt_tau_heavy_lightfake' in fName:
    truth_data = tree.arrays(["label_prompt", "label_tau", "label_heavy", "label_lightfake"], library="np")
    y_true_matrix = np.stack([
        truth_data["label_prompt"],
        truth_data["label_tau"], 
        truth_data["label_heavy"],
        truth_data["label_lightfake"]
    ], axis=1)
    true_labels = np.argmax(y_true_matrix, axis=1)
    score_data = tree.arrays(["score_label_prompt", "score_label_tau", "score_label_heavy", "score_label_lightfake"], library="np")
    y_pred = np.argmax(np.stack([
        score_data["score_label_prompt"],
        score_data["score_label_tau"], 
        score_data["score_label_heavy"],
        score_data["score_label_lightfake"]
    ], axis=1), axis=1)
    cm = confusion_matrix(true_labels, y_pred, normalize=None)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=['Prompt', 'Tau', 'Heavy', 'LightFake']
    )

else:
    truth_data = tree.arrays(["label_prompt", "label_heavy", "label_lightfake"], library="np")
    y_true_matrix = np.stack([
        truth_data["label_prompt"],
        truth_data["label_heavy"],
        truth_data["label_lightfake"]
    ], axis=1)
    true_labels = np.argmax(y_true_matrix, axis=1)
    score_data = tree.arrays(["score_label_prompt", "score_label_heavy", "score_label_lightfake"], library="np")
    y_pred = np.argmax(np.stack([
        score_data["score_label_prompt"],
        score_data["score_label_heavy"],
        score_data["score_label_lightfake"]
    ], axis=1), axis=1)
    cm = confusion_matrix(true_labels, y_pred, normalize=None)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=['Prompt', 'Heavy', 'LightFake']
    )
    
disp.plot(cmap='Blues')
plt.savefig('pics/matrix.pdf', bbox_inches='tight')
