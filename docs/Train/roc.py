#!/bin/env python3

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal for")

import uproot
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

onlyBDT = False
includeTau = True
rocTypes = ['promptVSnonprompt', 'promptVStau', 'heavyVSlight']
rocType = rocTypes[0] # Note: original script used here, my previous versions used 'rocTypes' without index
ann = False

year = ['2022']
#year = ['2016', '2017', '2018', '2022', '2023', '2024']
#year = ['2018', '2022', '2023']
flav = 'electron'
#topMVAs = ['PromptMVA', 'TOP-UL', 'TOP_binary', 'TOPPNet_binary', 'TOPUParT_binary', 'TOP_multiclass']
#topMVAs = ['PromptMVA', 'TOP-UL', 'TOP_binary', 'TOP_prompt_tau_heavy_lightfake', 'TOP_prompt_heavy_lightfake', 'TOP_prompt_heavy_light_fake']
#topMVAs = ['TOP_binary', 'TOPPNet_binary', 'TOPUParT_binary', 'TOPMvaId_binary']
topMVAs = []
#topMVAs = ['PromptMVA', 'TOP_binary']
if rocType == 'promptVStau': topMVAs = ['TOP_prompt_tau_heavy_lightfake']
if rocType == 'heavyVSlight': topMVAs = ['TOP_prompt_tau_heavy_lightfake', 'TOP_prompt_heavy_lightfake', 'TOP_prompt_heavy_light_fake']
#pnetMVAs = ['PNet', 'PNet_multiclass', 'cms_binary']
#pnetMVAs = ['cms_multiclass', 'PNet_multiclass']
#pnetMVAs = ['cms_binary', 'cms_multiclass', 'PNet', 'PNet_multiclass']
#pnetMVAs = ['PNet', 'PNet_multiclass']
#pnetMVAs = ['cms_binary']
#pnetMVAs = ['cms_multiclass', 'PNet', 'PNet_four', 'PNet_three', 'PNet_tau']
#pnetMVAs = ['cms_multiclass', 'PNet_four']
pnetMVAs = ['PNet_tau']

cols = ['black', 'blue', 'green', 'red', 'orange', 'yellow']

f, ax = plt.subplots()

# Dictionary to store ROC curve data for easy comparison later
curve_data = {}

for iyear, y in enumerate(year):
    conf = {
        'PromptMVA': {'label': 'PromptMVA', 'fname': 'predict/predict_'+flav+'_'+y+'_mvaTTH.root', 'color': 'green', 'style': '-'},
        'TOP-UL': {'label': 'TOP-UL', 'fname': 'predict/predict_'+flav+'_'+y+'_TOP-UL.root', 'color': 'black', 'style': '--'},
        'TOP_binary': {'label': 'TOP', 'fname': 'predict/predict_'+flav+'_'+y+'_TOP_binary.root', 'color': 'black', 'style': '-'},
        'TOP_prompt_tau_heavy_lightfake': {'label': 'TOP-tau', 'fname': 'predict/predict_'+flav+'_'+y+'_TOP_prompt_tau_heavy_lightfake.root', 'color': 'black', 'style': ':'},
        'TOP_prompt_heavy_lightfake': {'label': 'TOP-3class', 'fname': 'predict/predict_'+flav+'_'+y+'_TOP_prompt_heavy_lightfake.root', 'color': (0, (3, 5, 1, 5, 1, 5))},
        'TOP_prompt_heavy_light_fake': {'label': 'TOP-4class', 'fname': 'predict/predict_'+flav+'_'+y+'_TOP_prompt_heavy_light_fake.root', 'color': '-.'},
        'TOPPNet_binary': {'label': 'TOP-bPNet', 'fname': 'predict/predict_'+flav+'_'+y+'_TOPPNet_binary.root', 'color': 'blue', 'style': '-'},
        'TOPUParT_binary': {'label': 'TOP-bUParT', 'fname': 'predict/predict_'+flav+'_'+y+'_TOPUParT_binary.root', 'color': 'orange', 'style': '-'},
        'TOPMvaId_binary': {'label': 'TOP-MvaId', 'fname': 'predict/predict_'+flav+'_'+y+'_TOPMvaId_binary.root', 'color': 'yellow', 'style': '-'},
        'PNet': {'label': 'PNet', 'fname': 'predict/predict_particlenet_'+flav+'_'+y+'_binary.root', 'color': 'red', 'style': '-'},
        'cms_binary': {'label': 'PNet (v9)', 'fname': 'predict/predict_'+flav+'_cms_binary.root', 'color': 'red', 'style': '--'},
        'cms_multiclass': {'label': 'PNet-mclass (v9)', 'fname': 'predict/predict_'+flav+'_cms_multiclass.root', 'color': 'gray', 'style': '--'},
        'PNet_tau': {'label': 'PNet-tau', 'fname': 'predict/predict_particlenet_'+flav+'_'+y+'_tau.root', 'color': 'orange', 'style': '--'},
        'PNet_four': {'label': 'PNet-4class', 'fname': 'predict/predict_particlenet_'+flav+'_'+y+'_four.root', 'color': 'orange', 'style': '-.'},
##        'PNet_four': {'label': 'PNet-4class', 'fname': 'predict/predict_particlenet_'+flav+'_2022_four_check_'+y+'.root', 'color': 'orange', 'style': '-.'},
        'PNet_three': {'label': 'PNet-3class', 'fname': 'predict/predict_particlenet_'+flav+'_'+y+'_three.root', 'color': 'orange', 'style': (0, (3, 5, 1, 5, 1, 5))},
    }

    fname, label, col, sty = [], [], [], []
    for mva in topMVAs:
        c = conf[mva]
        fName = c['fname']
        if rocType == 'promptVStau': fName = fName.replace('.root', '_tau.root')
        fname.append(fName)
        label.append(c['label'])
        if len(year) == 1: col.append(c['color'])
        else: col.append(cols[iyear])
        sty.append(c['style'])
    if not onlyBDT:
        for mva in pnetMVAs:
            c = conf[mva]
            fName = c['fname']
            if rocType == 'promptVStau': fName = fName.replace('.root', '_tau.root')
            fname.append(fName)
            label.append(c['label'])
            if len(year) == 1: col.append(c['color'])
            else: col.append(cols[iyear])
            sty.append(c['style'])

    for i in range(len(fname)):
        df = uproot.open(fname[i])["Events"].arrays(library='pd')
    
        if rocType == 'promptVSnonprompt':
            if ("label_tau" not in df.columns) or (not includeTau):
                y_true = df["label_prompt"].values
#                y_pred = df.eval("score_label_prompt/(score_label_tau+score_label_heavy+score_label_lightfake)")
#                y_pred = df.eval("score_label_prompt/(score_label_heavy+score_label_light+score_label_fake)")
                y_pred = df.eval("score_label_prompt")                
            elif includeTau:
                y_true = df.eval("(label_prompt == 1) | (label_tau == 1)").astype(int).values
                y_pred = df.eval("score_label_prompt + score_label_tau").values
        elif rocType == 'promptVStau':
            mask = (df["label_prompt"] == 1) | (df["label_tau"] == 1)
            df_filtered = df[mask]
            y_true = (df_filtered["label_prompt"] == 1).astype(int)
            y_pred = df_filtered.eval("score_label_prompt")
        elif rocType == 'heavyVSlight':
            mask = (df["label_prompt"] != 1)
            if "label_tau" in df: mask &= (df["label_tau"] != 1)
            df_filtered = df[mask]
            y_true = (df_filtered["label_heavy"] == 1).astype(int)
            y_pred = df_filtered.eval("score_label_heavy")
        
        y_pred[np.isnan(y_pred)] = 0

        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        
        current_label = label[i]
        
        if len(year) == 1:
            ax.plot(fpr, tpr, label='%s'%(current_label), color=col[i], linestyle=sty[i], linewidth=3)
        else:
            ax.plot(fpr, tpr, label='%s %s'%(current_label, y), color=col[i], linestyle=sty[i], linewidth=3)
            
        curve_data[current_label] = {'fpr': fpr, 'tpr': tpr, 'color': col[i]}

        if len(year) == 1: ax.text(0.05, 0.92, y, transform=ax.transAxes, fontsize=27)
    
#    ax.legend(bbox_to_anchor=(1.04,1), loc="best")


if ann:
    top_data = curve_data.get('PromptMVA')
    pnet_data = curve_data.get('PNet-4class')

    if top_data and pnet_data:
        # --- Reference Point A: At 95% Prompt efficiency (TPR = 0.95) ---
        ref_tpr_A = 0.95 
        fpr_top_A = np.interp(ref_tpr_A, top_data['tpr'], top_data['fpr'])
        fpr_pnet_A = np.interp(ref_tpr_A, pnet_data['tpr'], pnet_data['fpr'])
        
        if fpr_top_A > 0 and fpr_pnet_A > 0:
            # Horizontal Arrow: Measures FPR reduction at constant TPR
            fpr_reduction_percent = ((fpr_top_A - fpr_pnet_A) / fpr_top_A) * 100
            fpr_mid_point_A = np.exp( (np.log(fpr_top_A) + np.log(fpr_pnet_A)) / 2 )
        
            ax.annotate('', 
                        xy=(fpr_pnet_A, ref_tpr_A),       
                        xytext=(fpr_top_A, ref_tpr_A),   
                        arrowprops=dict(arrowstyle='->', color='teal', lw=3, mutation_scale=20),
                        annotation_clip=True)

            ax.text(0.002, 0.935, 
                    f'-{fpr_reduction_percent:.0f}% nonprompt',
                    color='teal', ha='left', va='bottom', fontsize=20, fontweight='bold')
                
            # Vertical Arrow: Measures TPR improvement at the specific FPR of point A
            # The start/end point of the horizontal arrow defines the X coordinate we need
            ref_fpr_B = fpr_pnet_A # Anchor vertical comparison at the PNet-4class X value
            tpr_top_B = np.interp(ref_fpr_B, top_data['fpr'], top_data['tpr'])
            tpr_pnet_B = ref_tpr_A # This is the Y value where they meet (0.95)
            
            tpr_improvement_percent = ((tpr_pnet_B - tpr_top_B) / tpr_top_B) * 100
            tpr_mid_point_B = (tpr_top_B + tpr_pnet_B) / 2
        
            ax.annotate('', 
                        xy=(ref_fpr_B, tpr_pnet_B),       
                        xytext=(ref_fpr_B, tpr_top_B),   
                        arrowprops=dict(arrowstyle='->', color='purple', lw=3, mutation_scale=20),
                        annotation_clip=True)

            ax.text(0.002, 0.92, 
                    f'+{tpr_improvement_percent:.0f}% prompt',
                    color='purple', ha='left', va='center', fontsize=20, fontweight='bold')

ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set_xlim(1e-3, 1.0)
ax.set_ylim(0.70, 1.0)
ax.set_xscale('log')
if rocType == 'promptVStau':
    ax.set_xlim(0.01, 1.0)
    ax.set_ylim(0.01, 1.0)
    ax.set_xscale('linear')
if rocType == 'heavyVSlight':
    ax.set_xlim(0.01, 1.0)
    ax.set_ylim(0.01, 1.0)
    ax.set_xscale('linear')
ax.set_ylabel('Prompt lepton efficiency', horizontalalignment='center', x=1.0)
if rocType == 'heavyVSlight':
    ax.set_ylabel('Nonprompt (heavy) lepton efficiency', horizontalalignment='center', x=1.0)
ax.set_xlabel('Nonprompt lepton efficiency', horizontalalignment='center', y=1.0)
if rocType == 'promptVStau':
    ax.set_xlabel('Tau lepton efficiency', horizontalalignment='center', y=1.0)
if rocType == 'heavyVSlight':
    ax.set_xlabel('Nonprompt (light+fake) lepton efficiency', horizontalalignment='center', y=1.0)
#ax.text(0.05, 0.92, y, transform=ax.transAxes, fontsize=27)
ax.text(0.75, 0.1, flav.title(), transform=ax.transAxes, fontsize=27)
plt.grid(which='both')
plt.savefig("pics/test.pdf", bbox_inches='tight')
