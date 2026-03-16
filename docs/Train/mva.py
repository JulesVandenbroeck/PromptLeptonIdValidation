import os
import sys
import math
import pickle, json
from array import array

import ROOT
import xgboost as xgb

class mva():

    def __init__(self, model, year):

        self.isValid = False

        self.XGBReader = {}
        models = model.split(',')
        mconf = 'n_estimators-2000__max_depth-4__eta-0.1__gamma-5__min_child_weight-500'

        for kl in ['Electron', 'Muon']:

            self.XGBReader[kl] = {}
            for m in models:
                if 'TOP-UL' in m: mpath = '/user/kskovpen/analysis/LeptonID/CMSSW_10_6_28/src/LeptonID/TopLeptonMVA/Train/xgboost/weights'
                elif 'binary' in m or 'prompt' in m: mpath = '/user/kskovpen/analysis/LeptonID/ParticleNet/CMSSW_15_0_15_patch4/src/TopLeptonMVA/Train/xgboost/jobs'
                self.XGBReader[kl][m] = xgb.Booster()
                yearm = year
                if 'binary' in m or 'prompt' in m:
                    if any(y in year for y in ['2022', '2022EE', '2023', '2023BPix']): yearm = 'Run3Summer'+year.replace('20', '')
                    elif '2024' in year: yearm = 'Run3Winter'+year.replace('20', '')
                    else: yearm = 'RunIISummer20UL'+year.replace('20', '')
                if 'binary' not in m and 'prompt' not in m: self.XGBReader[kl][m].load_model(mpath+'/'+(m+'.TOP_v1').replace('TOP-UL.','')+'_'+kl.lower()[:4]+'_'+yearm+'/'+mconf+'/xgb.bin')
                else:
                    print(mpath+'/'+m+'_'+kl.lower()+'_'+yearm+'/'+mconf+'/xgb.bin')
                    self.XGBReader[kl][m].load_model(mpath+'/'+m+'_'+kl.lower()+'_'+yearm+'/'+mconf+'/xgb.bin')
                self.isValid = True
            
    def predict(self, lep, x, m):

        return self.XGBReader[lep][m].predict(x)
        
