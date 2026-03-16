#!/bin/env python3

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal for")
warnings.filterwarnings("ignore", module="xgboost")

#from google.protobuf.json_format import MessageToDict
import xgboost as xgb
import numpy as np
import pandas as pd
import awkward as ak
import onnxruntime as ort
import onnx
from pathlib import Path
import ROOT
from array import array
import sys, os, math
import argparse
import common as c
import mva

sig = '(genPartFlav == 1) | (genPartFlav == 15)' # prompt+tau
#sig = '(genPartFlav == 1) & (1)' # prompt

#TopMVA = 'TOP-UL.TOP_v1'
#TopMVA = 'TOP_binary'
#TopMVA = 'TOP-UL.TOP_v1,TOP_binary,TOPMvaId_binary,TOPPNet_binary,TOPUParT_binary,TOP_prompt_tau_heavy_lightfake,TOP_prompt_heavy_lightfake,TOP_prompt_heavy_light_fake'
#TopMVA = 'TOP-UL.TOP_v1,TOP_binary,TOP_prompt_tau_heavy_lightfake,TOP_prompt_heavy_lightfake,TOP_prompt_heavy_light_fake'
#TopMVA = 'TOP_prompt_tau_heavy_lightfake'
#TopMVA = 'TOP_prompt_heavy_light_fake'
TopMVA = 'TOP_binary'
evaluateBDT = False
evaluateONNX = True

parser = argparse.ArgumentParser(description='Args')
parser.add_argument('--batch_size', default='512')
#parser.add_argument('--nmax', type=int, default=20000)
parser.add_argument('--nmax', type=int, default=200000)
#parser.add_argument('--model', default=['models/model_network_flav_year_multiclass.onnx'])
#parser.add_argument('--model', default=['models/model_network_flav_year_binary.onnx'])
#parser.add_argument('--model', default=['models/model_network_flav_year_three.onnx'])
parser.add_argument('--model', default=['models/model_network_flav_year_four.onnx'])
#parser.add_argument('--model', default=['models/model_network_flav_year_tau.onnx'])
#parser.add_argument('--model', default=['models/model_flav_cms_multiclass.onnx'])
parser.add_argument('--year', default='2022', choices=['2016', '2016APV', '2017', '2018', '2022', '2022EE', '2023', '2023BPix', '2024'])
#parser.add_argument('--name', default=['PNet (cmssw)', 'PNet (year)'])
parser.add_argument('--name', default=['PNet (year)'])
#parser.add_argument('--name', default=['PNet (cmssw)'])
parser.add_argument('--flav', default='electron', choices=['electron', 'muon'])
parser.add_argument('--pt', default='10')
parser.add_argument('--network', default='particlenet')
options = parser.parse_args()

for im in range(len(options.model)):
    options.model[im] = options.model[im].replace('year', options.year)
    options.model[im] = options.model[im].replace('flav', options.flav)
    options.model[im] = options.model[im].replace('network', options.network)
for im in range(len(options.name)):
    options.name[im] = options.name[im].replace('year', options.year)

def process(df, mvaName):
    if options.flav == 'muon': df = df.Filter("(MuonSelected_LepGood_pt[0] > "+options.pt+") & (abs(MuonSelected_LepGood_eta[0]) < 2.4) & (MuonSelected_mediumId[0] == 1)")
    else: df = df.Filter("(ElectronSelected_pt[0] > "+options.pt+") & (abs(ElectronSelected_eta[0]) < 2.5) & (ElectronSelected_lostHits[0] <= 1) & (ElectronSelected_isGSF[0] == 1)")
#    df = df.Filter("(genPartFlav != 22)") # remove conversions

    if 'tau' not in mvaName and not 'cms_multiclass' in mvaName:
        df = df.Define("label_prompt", sig)
    else:
        df = df.Define("label_prompt", sig)
        df = df.Define("label_tau", "(genPartFlav == 15) & (1)")

    if 'cms' not in mvaName:
        if '_heavy_lightfake' in mvaName or 'three' in mvaName or 'tau' in mvaName:
            df = df.Define("label_heavy", "(genPartFlav == 5) & (1)")
            df = df.Define("label_lightfake", "(genPartFlav != 1) & (genPartFlav != 15) & (genPartFlav != 5)")
        if '_heavy_light_fake' in mvaName or 'four' in mvaName:
            df = df.Define("label_heavy", "(genPartFlav == 5) & (1)")
            df = df.Define("label_light", "(genPartFlav == 4) | (genPartFlav == 3)")
            df = df.Define("label_fake", "(genPartFlav != 1) & (genPartFlav != 15) & (genPartFlav != 5) & (genPartFlav != 4) & (genPartFlav != 3)")
    elif 'multiclass' in mvaName and 'muon' in mvaName:
        df = df.Define("label_heavy", "(genPartFlav == 5) & (1)")
        df = df.Define("label_lightfake", "(genPartFlav != 1) & (genPartFlav != 15) & (genPartFlav != 5)")
    elif 'multiclass' in mvaName and 'electron' in mvaName:
        df = df.Define("label_heavy", "(genPartFlav == 5) & (1)")
        df = df.Define("label_light", "(genPartFlav == 4) | (genPartFlav == 3)")
        df = df.Define("label_fake", "(genPartFlav != 1) & (genPartFlav != 15) & (genPartFlav != 5) & (genPartFlav != 4) & (genPartFlav != 3)")
        
#    df = df.Filter("(genPartFlav != 15)") # remove taus
#    df = df.Define("label_prompt", "(genPartFlav == 1) | (0)")
    entries = df.Count().GetValue()
    if entries > options.nmax: entries = options.nmax
    df = df.Range(0, entries+1)
    return df, entries
    
trinput = ROOT.TChain('ntuplizer_'+options.flav+'/Events')
for ds in c.getdataset(options.year):
    trinput.Add(ds)

if evaluateBDT:
    topMVAs = TopMVA.replace('.TOP_v1', '').split(',')
    dfm = ROOT.RDataFrame(trinput)
    dfm = dfm.Define("lep_abseta", "abs(MuonSelected_LepGood_eta)") if options.flav == "muon" else dfm.Define("lep_abseta", "abs(ElectronSelected_eta)")
    rTopMVA = {}
    for topMVA in topMVAs:
        if 'TOP-UL' in topMVA: yearTopMVA = options.year if options.year in ['2016', '2016APV', '2017', '2018'] else '2018'
        else: yearTopMVA = options.year
        varTopMVA = c.var[options.flav.title()][topMVA]
        dl = mva.mva(topMVA, yearTopMVA)
        d = {}
        for v in varTopMVA.values():
            vdata = dfm.AsNumpy([v])[v]
            d[v] = vdata
        dfr = pd.DataFrame(d)
        dfr = dfr.map(lambda x: x[0])
        x = xgb.DMatrix(dfr)
        flavTopMVA = options.flav.title()+'Selected_'+topMVA.replace('-', '_')
        rTopMVA[topMVA] = dl.predict(options.flav.title(), x, topMVA)
        if 'prompt' in topMVA:
            sigClass = 1
            if 'prompt_tau_heavy_lightfake' in topMVA:
                tauTopMVA = rTopMVA.copy()
                tauTopMVA[topMVA] = [el[0] for el in tauTopMVA[topMVA]]
                heavyTopMVA = rTopMVA.copy()
                heavyTopMVA[topMVA] = [el[2] for el in heavyTopMVA[topMVA]]
                lightfakeTopMVA = rTopMVA.copy()
                lightfakeTopMVA[topMVA] = [el[3] for el in lightfakeTopMVA[topMVA]]
            elif 'prompt_heavy_lightfake' in topMVA:
                heavyTopMVA = rTopMVA.copy()
                heavyTopMVA[topMVA] = [el[0] for el in heavyTopMVA[topMVA]]
                lightfakeTopMVA = rTopMVA.copy()
                lightfakeTopMVA[topMVA] = [el[2] for el in lightfakeTopMVA[topMVA]]
            elif 'prompt_heavy_light_fake' in topMVA:
                heavyTopMVA = rTopMVA.copy()
                heavyTopMVA[topMVA] = [el[0] for el in heavyTopMVA[topMVA]]
                lightTopMVA = rTopMVA.copy()
                lightTopMVA[topMVA] = [el[2] for el in lightTopMVA[topMVA]]
                fakeTopMVA = rTopMVA.copy()
                fakeTopMVA[topMVA] = [el[3] for el in fakeTopMVA[topMVA]]
            rTopMVA[topMVA] = [el[sigClass] for el in rTopMVA[topMVA]]

    mvas = topMVAs + ['mvaTTH']
    for mva in mvas:
        print('Evaluating', mva)
        fname = "predict/predict_"+options.flav+"_"+options.year+"_"+mva+".root"
        if sig == '(genPartFlav == 1) & (1)': fname = fname.replace('.root', '_tau.root')
        f = ROOT.TFile(fname, "RECREATE")
        tr = ROOT.TTree("Events", "predict")
        v_score_label_prompt, v_score_label_tau, v_score_label_heavy, v_score_label_light, v_score_label_fake, v_score_label_lightfake = array('f', [-1]), array('f', [-1]), array('f', [-1]), array('f', [-1]), array('f', [-1]), array('f', [-1])
        v_label_prompt, v_label_tau, v_label_heavy, v_label_light, v_label_fake, v_label_lightfake = array('i', [-1]), array('i', [-1]), array('i', [-1]), array('i', [-1]), array('i', [-1]), array('i', [-1])
        tr.Branch("score_label_prompt", v_score_label_prompt, 'score_label_prompt/F')
        tr.Branch("label_prompt", v_label_prompt, 'label_prompt/I')
        if 'prompt_tau_heavy_lightfake' in mva:
            tr.Branch("score_label_tau", v_score_label_tau, 'score_label_tau/F')
            tr.Branch("label_tau", v_label_tau, 'label_tau/I')            
            tr.Branch("score_label_heavy", v_score_label_heavy, 'score_label_heavy/F')
            tr.Branch("label_heavy", v_label_heavy, 'label_heavy/I')            
            tr.Branch("score_label_lightfake", v_score_label_lightfake, 'score_label_lightfake/F')
            tr.Branch("label_lightfake", v_label_lightfake, 'label_lightfake/I')            
        elif 'prompt_heavy_lightfake' in mva:
            tr.Branch("score_label_heavy", v_score_label_heavy, 'score_label_heavy/F')
            tr.Branch("label_heavy", v_label_heavy, 'label_heavy/I')            
            tr.Branch("score_label_lightfake", v_score_label_lightfake, 'score_label_lightfake/F')
            tr.Branch("label_lightfake", v_label_lightfake, 'label_lightfake/I')            
        elif 'prompt_heavy_light_fake' in mva:
            tr.Branch("score_label_heavy", v_score_label_heavy, 'score_label_heavy/F')
            tr.Branch("label_heavy", v_label_heavy, 'label_heavy/I')            
            tr.Branch("score_label_light", v_score_label_light, 'score_label_light/F')
            tr.Branch("label_light", v_label_light, 'label_light/I')            
            tr.Branch("score_label_fake", v_score_label_fake, 'score_label_fake/F')
            tr.Branch("label_fake", v_label_fake, 'label_fake/I')            
        df = ROOT.RDataFrame(trinput)
        flavTopMVA = options.flav.title()+'Selected_'+mva.replace('-', '_')
        if mva != 'mvaTTH': df = df.Define(flavTopMVA, 'auto to_eval = std::string("rTopMVA[") + std::string("mva") + std::string("][") + std::to_string(rdfentry_) + "]"; return float(TPython::Eval(to_eval.c_str()));')
        if 'prompt_tau_heavy_lightfake' in mva:
            flavTopMVA_tau = options.flav.title()+'Selected_'+mva.replace('-', '_')+'_tau'
            df = df.Define(flavTopMVA_tau, 'auto to_eval = std::string("tauTopMVA[") + std::string("mva") + std::string("][") + std::to_string(rdfentry_) + "]"; return float(TPython::Eval(to_eval.c_str()));')            
            flavTopMVA_heavy = options.flav.title()+'Selected_'+mva.replace('-', '_')+'_heavy'
            df = df.Define(flavTopMVA_heavy, 'auto to_eval = std::string("heavyTopMVA[") + std::string("mva") + std::string("][") + std::to_string(rdfentry_) + "]"; return float(TPython::Eval(to_eval.c_str()));')            
            flavTopMVA_lightfake = options.flav.title()+'Selected_'+mva.replace('-', '_')+'_lightfake'
            df = df.Define(flavTopMVA_lightfake, 'auto to_eval = std::string("lightfakeTopMVA[") + std::string("mva") + std::string("][") + std::to_string(rdfentry_) + "]"; return float(TPython::Eval(to_eval.c_str()));')            
        elif 'prompt_heavy_lightfake' in mva:
            flavTopMVA_heavy = options.flav.title()+'Selected_'+mva.replace('-', '_')+'_heavy'
            df = df.Define(flavTopMVA_heavy, 'auto to_eval = std::string("heavyTopMVA[") + std::string("mva") + std::string("][") + std::to_string(rdfentry_) + "]"; return float(TPython::Eval(to_eval.c_str()));')            
            flavTopMVA_lightfake = options.flav.title()+'Selected_'+mva.replace('-', '_')+'_lightfake'
            df = df.Define(flavTopMVA_lightfake, 'auto to_eval = std::string("lightfakeTopMVA[") + std::string("mva") + std::string("][") + std::to_string(rdfentry_) + "]"; return float(TPython::Eval(to_eval.c_str()));')            
        elif 'prompt_heavy_light_fake' in mva:
            flavTopMVA_heavy = options.flav.title()+'Selected_'+mva.replace('-', '_')+'_heavy'
            df = df.Define(flavTopMVA_heavy, 'auto to_eval = std::string("heavyTopMVA[") + std::string("mva") + std::string("][") + std::to_string(rdfentry_) + "]"; return float(TPython::Eval(to_eval.c_str()));')            
            flavTopMVA_light = options.flav.title()+'Selected_'+mva.replace('-', '_')+'_light'
            df = df.Define(flavTopMVA_light, 'auto to_eval = std::string("lightTopMVA[") + std::string("mva") + std::string("][") + std::to_string(rdfentry_) + "]"; return float(TPython::Eval(to_eval.c_str()));')            
            flavTopMVA_fake = options.flav.title()+'Selected_'+mva.replace('-', '_')+'_fake'
            df = df.Define(flavTopMVA_fake, 'auto to_eval = std::string("fakeTopMVA[") + std::string("mva") + std::string("][") + std::to_string(rdfentry_) + "]"; return float(TPython::Eval(to_eval.c_str()));')            
        df, _ = process(df, mva)
        pred_mva = df.AsNumpy([flavTopMVA])[flavTopMVA]
        label_prompt = df.AsNumpy(["label_prompt"])["label_prompt"]
        if 'prompt_tau_heavy_lightfake' in mva:
            pred_mva_tau = df.AsNumpy([flavTopMVA_tau])[flavTopMVA_tau]
            label_tau = df.AsNumpy(["label_tau"])["label_tau"]
            pred_mva_heavy = df.AsNumpy([flavTopMVA_heavy])[flavTopMVA_heavy]
            label_heavy = df.AsNumpy(["label_heavy"])["label_heavy"]
            pred_mva_lightfake = df.AsNumpy([flavTopMVA_lightfake])[flavTopMVA_lightfake]
            label_lightfake = df.AsNumpy(["label_lightfake"])["label_lightfake"]
        elif 'prompt_heavy_lightfake' in mva:
            pred_mva_heavy = df.AsNumpy([flavTopMVA_heavy])[flavTopMVA_heavy]
            label_heavy = df.AsNumpy(["label_heavy"])["label_heavy"]
            pred_mva_lightfake = df.AsNumpy([flavTopMVA_lightfake])[flavTopMVA_lightfake]
            label_lightfake = df.AsNumpy(["label_lightfake"])["label_lightfake"]
        elif 'prompt_heavy_light_fake' in mva:
            pred_mva_heavy = df.AsNumpy([flavTopMVA_heavy])[flavTopMVA_heavy]
            label_heavy = df.AsNumpy(["label_heavy"])["label_heavy"]
            pred_mva_light = df.AsNumpy([flavTopMVA_light])[flavTopMVA_light]
            label_light = df.AsNumpy(["label_light"])["label_light"]
            pred_mva_fake = df.AsNumpy([flavTopMVA_fake])[flavTopMVA_fake]
            label_fake = df.AsNumpy(["label_fake"])["label_fake"]
        for ipred, pred in enumerate(pred_mva):
            if mva == 'mvaTTH': v_score_label_prompt[0] = pred_mva[ipred][0]
            else: v_score_label_prompt[0] = pred_mva[ipred]
            v_label_prompt[0] = label_prompt[ipred]
            if 'prompt_tau_heavy_lightfake' in mva:
                v_score_label_tau[0] = pred_mva_tau[ipred]
                v_label_tau[0] = label_tau[ipred]                
                v_score_label_heavy[0] = pred_mva_heavy[ipred]
                v_label_heavy[0] = label_heavy[ipred]                
                v_score_label_lightfake[0] = pred_mva_lightfake[ipred]
                v_label_lightfake[0] = label_lightfake[ipred]                
            elif 'prompt_heavy_lightfake' in mva:
                v_score_label_heavy[0] = pred_mva_heavy[ipred]
                v_label_heavy[0] = label_heavy[ipred]                
                v_score_label_lightfake[0] = pred_mva_lightfake[ipred]
                v_label_lightfake[0] = label_lightfake[ipred]                
            elif 'prompt_heavy_light_fake' in mva:
                v_score_label_heavy[0] = pred_mva_heavy[ipred]
                v_label_heavy[0] = label_heavy[ipred]                
                v_score_label_light[0] = pred_mva_light[ipred]
                v_label_light[0] = label_light[ipred]                
                v_score_label_fake[0] = pred_mva_fake[ipred]
                v_label_fake[0] = label_fake[ipred]                
            tr.Fill()        
        f.Write()
        f.Close()

if evaluateONNX:

    model, session = [], []
    for m in options.model:
        model.append(onnx.load(m))
        session.append(ort.InferenceSession(m))

    inputs = {}
    for inp in model[0].graph.input:
        shape = str(inp.type.tensor_type.shape.dim)
        inputs[inp.name] = [int(s) for s in shape.split() if s.isdigit()]
#    print(inputs)

    vars = {}
    vars["pf_points"] = ["PF_phi_rel", "PF_eta_rel"]
    if options.flav == "muon":
        vars["pf_features"] = ["PF_pt", "PF_pt_rel_log", "PF_eta_rel", "PF_phi_rel", "PF_charge", "PF_isElectron", "PF_isMuon", "PF_isNeutralHadron", "PF_isPhoton", "PF_isChargedHadron", "PF_fromPV", "PF_puppiWeightNoLep", "PF_hcalFraction", "PF_dzSig", "PF_dxySig"]
    else:
        vars["pf_features"] = ["PF_pt", "PF_pt_rel_log", "PF_eta_rel", "PF_phi_rel", "PF_charge", "PF_isElectron", "PF_isMuon", "PF_isNeutralHadron", "PF_isPhoton", "PF_isChargedHadron", "PF_fromPV", "PF_puppiWeightNoLep", "PF_hcalFraction", "PF_hcalFractionCalib", "PF_dz", "PF_dzSig", "PF_dxy", "PF_dxySig", "PF_trackerLayersWithMeasurement"]        
    vars["pf_mask"] = ["PF_mask"]
    if options.flav == "muon":
#        vars["sv_points"] = ["SV_eta", "SV_phi"]
        vars["sv_points"] = ["SV_eta_rel", "SV_phi_rel"] # FIXME
    else: vars["sv_points"] = ["SV_eta_rel", "SV_phi_rel"]
    if options.flav == "muon": vars["sv_features"] = ["SV_pt", "SV_mass", "SV_phi", "SV_eta", "SV_dlenSig", "SV_dxy", "SV_ndof", "SV_chi2", "SV_cospAngle", "SV_nTracks"]
    else: vars["sv_features"] = ["SV_pt", "SV_pt_rel", "SV_eta_rel", "SV_phi_rel", "SV_mass", "SV_dlenSig", "SV_dxy", "SV_ndof", "SV_chi2", "SV_cospAngle"]
    vars["sv_mask"] = ["SV_mask"]
    if options.flav == 'muon':
        vars["high_level"] = ["MuonSelected_dxy", "MuonSelected_dz", "MuonSelected_sip3d", "MuonSelected_segmentComp", "MuonSelected_LepGood_jetDF", "MuonSelected_LepGood_jetPtRatio", "MuonSelected_LepGood_jetNDauChargedMVASel", "MuonSelected_LepGood_dz", "MuonSelected_LepGood_eta", "MuonSelected_LepGood_miniRelIsoNeutral", "MuonSelected_LepGood_jetPtRelv2", "MuonSelected_LepGood_miniRelIsoCharged", "MuonSelected_LepGood_pfRelIso03_all", "MuonSelected_mvaId"]
    else:
        vars["high_level"] = ["ElectronSelected_dxy_log", "ElectronSelected_dxySig", "ElectronSelected_dz_log", "ElectronSelected_dzSig", "ElectronSelected_sip3d", "ElectronSelected_e_ECAL", "ElectronSelected_lostHits", "ElectronSelected_closeTrackNLayers", "ElectronSelected_deltaetacltrkcalo", "ElectronSelected_dEtaInSeed", "ElectronSelected_hcaloverecal", "ElectronSelected_r9full", "ElectronSelected_e1x5bye5x5", "ElectronSelected_sigmaietaieta", "ElectronSelected_sigmaiphiiphi", "ElectronSelected_supcl_etaWidth", "ElectronSelected_supcl_phiWidth", "ElectronSelected_dr03HcalDepth1TowerSumEt_Rel", "ElectronSelected_fbrem", "ElectronSelected_eoverp", "ElectronSelected_passConversionVeto", "ElectronSelected_jetPtRatio", "ElectronSelected_jetNDauCharged", "ElectronSelected_jetPtRelv2_bylepPt", "ElectronSelected_mvaFall17V2noIso", "ElectronSelected_minisoch", "ElectronSelected_minisonh"]

    dim = {"pf_points": 60, "pf_features": 60, "pf_mask": 60, "sv_points": 5, "sv_features": 5, "sv_mask": 5, "high_level": 1}

    for im in range(len(model)):        
        
        tag = options.model[im].replace('models/model_', '').replace('.onnx', '')
        print('Evaluating', tag)
        f = ROOT.TFile("predict/predict_"+tag+".root", "RECREATE")
##        f = ROOT.TFile("predict/predict_"+tag+"_check_2018.root", "RECREATE")
        tr = ROOT.TTree("Events", "predict")
        v_score_label_prompt, v_score_label_tau, v_score_label_heavy, v_score_label_light, v_score_label_fake, v_score_label_lightfake = array('f', [-1]), array('f', [-1]), array('f', [-1]), array('f', [-1]), array('f', [-1]), array('f', [-1])
        v_label_prompt, v_label_tau, v_label_heavy, v_label_light, v_label_fake, v_label_lightfake = array('i', [-1]), array('i', [-1]), array('i', [-1]), array('i', [-1]), array('i', [-1]), array('i', [-1])
        tr.Branch("score_label_prompt", v_score_label_prompt, 'score_label_prompt/F')
        tr.Branch("label_prompt", v_label_prompt, 'label_prompt/I')
        if 'prompt_tau_heavy_lightfake' in tag or 'tau' in tag or ('cms_multiclass' in tag and 'muon' in tag):
            tr.Branch("score_label_tau", v_score_label_tau, 'score_label_tau/F')
            tr.Branch("label_tau", v_label_tau, 'label_tau/I')            
            tr.Branch("score_label_heavy", v_score_label_heavy, 'score_label_heavy/F')
            tr.Branch("label_heavy", v_label_heavy, 'label_heavy/I')            
            tr.Branch("score_label_lightfake", v_score_label_lightfake, 'score_label_lightfake/F')
            tr.Branch("label_lightfake", v_label_lightfake, 'label_lightfake/I')            
        elif 'prompt_heavy_lightfake' in tag or 'three' in tag:
            tr.Branch("score_label_heavy", v_score_label_heavy, 'score_label_heavy/F')
            tr.Branch("label_heavy", v_label_heavy, 'label_heavy/I')            
            tr.Branch("score_label_lightfake", v_score_label_lightfake, 'score_label_lightfake/F')
            tr.Branch("label_lightfake", v_label_lightfake, 'label_lightfake/I')            
        elif 'prompt_heavy_light_fake' in tag or 'four' in tag or ('cms_multiclass' in tag and 'electron' in tag):
            tr.Branch("score_label_heavy", v_score_label_heavy, 'score_label_heavy/F')
            tr.Branch("label_heavy", v_label_heavy, 'label_heavy/I')            
            tr.Branch("score_label_light", v_score_label_light, 'score_label_light/F')
            tr.Branch("label_light", v_label_light, 'label_light/I')            
            tr.Branch("score_label_fake", v_score_label_fake, 'score_label_fake/F')
            tr.Branch("label_fake", v_label_fake, 'label_fake/I')
        
        df = ROOT.RDataFrame(trinput)
        df, entries = process(df, tag)
        print("Used stats:", entries)

        n_batches = int(entries/float(options.batch_size))

        for ibatch in range(n_batches+1):
    
            event_min = int(options.batch_size) * int(ibatch)
            event_max = event_min + int(options.batch_size)
            print(entries, event_min, event_max)
            if event_max > entries:
                event_max = entries
            dfr = df.Range(event_min, event_max)

            model_inputs = {}

            label_prompt = dfr.AsNumpy(["label_prompt"])["label_prompt"]
            if 'prompt_tau_heavy_lightfake' in tag or 'tau' in tag or ('cms_multiclass' in tag and 'muon' in tag):
                label_tau = dfr.AsNumpy(["label_tau"])["label_tau"]
                label_heavy = dfr.AsNumpy(["label_heavy"])["label_heavy"]
                label_lightfake = dfr.AsNumpy(["label_lightfake"])["label_lightfake"]
            elif 'prompt_heavy_lightfake' in tag or 'three' in tag:
                label_heavy = dfr.AsNumpy(["label_heavy"])["label_heavy"]
                label_lightfake = dfr.AsNumpy(["label_lightfake"])["label_lightfake"]
            elif 'prompt_heavy_light_fake' in tag or 'four' in tag or ('cms_multiclass' in tag and 'electron' in tag):
                label_heavy = dfr.AsNumpy(["label_heavy"])["label_heavy"]
                label_light = dfr.AsNumpy(["label_light"])["label_light"]
                label_fake = dfr.AsNumpy(["label_fake"])["label_fake"]

            for name in inputs.keys():
                v = vars[name]
                if ibatch == 0:
                    if 'cms_multiclass' in tag and name == 'high_level':
                        if options.flav == 'muon':
                            v = vars[name]
                            if 'MuonSelected_LepGood_pfRelIso03_all' in v:
                                v = vars[name]
                                v.remove('MuonSelected_LepGood_pfRelIso03_all')
                    elif name == 'high_level':
                        v = vars[name]
                        if options.flav == 'muon':
                            v.insert(0, 'MuonSelected_LepGood_pt')
                            v.remove('MuonSelected_LepGood_eta')
                            v.insert(1, 'MuonSelected_LepGood_eta')
                            v.remove("MuonSelected_LepGood_dz")
                        else:
                            v.insert(0, 'ElectronSelected_eta')
                            v.insert(0, 'ElectronSelected_pt')
                            v.append('ElectronSelected_jetbtag')
                            v.append('ElectronSelected_pfRelIso03_all')
                            v = ['ElectronSelected_jetPtRelv2' if x == 'ElectronSelected_jetPtRelv2_bylepPt' else x for x in v]
                            v = ['ElectronSelected_mvaNoIso' if x == 'ElectronSelected_mvaFall17V2noIso' else x for x in v]
                    elif 'cms' in tag and name == 'pf_features':
                        v = vars[name]
                        if options.flav == 'muon' and 'multiclass' in tag:
                            v.remove("PF_pt_rel_log")
                if ibatch == 0:
                    print(f"\n>>> FINAL INPUT ORDER for {tag} | Group: {name}")
                    for idx, var_name in enumerate(v):
                        print(f"  [{idx}] {var_name}")
                    print(f">>> TOTAL VARIABLES: {len(v)}\n")

                if name not in ['pf_mask', 'sv_mask']:
                    d = dfr.AsNumpy(v)
                elif name == 'sv_mask':
                    d = ak.ones_like(dfr.AsNumpy(["SV_pt"]))["SV_pt"]
                    d = {"SV_mask": d}
                elif name == 'pf_mask':
                    if options.flav == 'muon': d = ak.ones_like(dfr.AsNumpy(["PF_dzSig"]))["PF_dzSig"]
                    else: d = ak.ones_like(dfr.AsNumpy(["PF_pt_rel_log"]))["PF_pt_rel_log"]
                    d = {"PF_mask": d}
                else: continue
                #        print(name)
                a = []
                for var in v:
                    arr = []
                    for v in d[var]:
                        target = np.zeros((1, dim[name]))
                        #                target[:] = np.nan
                        data = np.array(v)
                        if len(data) > 0:
                            data = np.reshape(data, (-1, len(data)))
                        else:
                            data = np.reshape(data, (-1, 2))
                        data = data[:dim[name], :dim[name]]
                        target[:data.shape[0], :data.shape[1]] = data
                        arr.append(target.ravel())
                    a.append(arr)
                a = [list(t) for t in zip(*a)]
                model_inputs[name] = a

            output_name = session[im].get_outputs()[0].name

            sigClass = 0 # default
            if 'multiclass' in tag and 'cms' in tag and 'muon' in tag: sigClass = 1 # cms muon multiclass trainings
#            elif 'multiclass' in tag: sigClass = 1 # new multiclass trainings
##            pred_onx = session[im].run([output_name], model_inputs)[0][:, sigClass]
            raw_preds = session[im].run([output_name], model_inputs)[0]
            pred_prompt = raw_preds[:, sigClass]

            if 'prompt_tau_heavy_lightfake' in tag or 'tau' in tag:
                pred_tau       = raw_preds[:, 1]
                pred_heavy     = raw_preds[:, 2]
                pred_lightfake = raw_preds[:, 3]
            elif 'prompt_heavy_lightfake' in tag or 'three' in tag:
                pred_heavy     = raw_preds[:, 1]
                pred_lightfake = raw_preds[:, 2]
            elif 'prompt_heavy_light_fake' in tag or 'four' in tag:
                pred_heavy     = raw_preds[:, 1]
                pred_light     = raw_preds[:, 2]
                pred_fake      = raw_preds[:, 3]
            elif ('cms_multiclass' in tag and 'muon' in tag):
                pred_lightfake = raw_preds[:, 0]
                pred_tau       = raw_preds[:, 2]
                pred_heavy     = raw_preds[:, 3]                
            elif ('cms_multiclass' in tag and 'electron' in tag):
                pred_heavy     = raw_preds[:, 1]
                pred_light     = raw_preds[:, 2]
                pred_fake      = raw_preds[:, 3]                
            
            for ipred, pred in enumerate(pred_prompt):
                v_score_label_prompt[0] = pred_prompt[ipred]
                v_label_prompt[0] = label_prompt[ipred]
                if 'prompt_tau_heavy_lightfake' in tag or 'tau' in tag or ('cms_multiclass' in tag and 'muon' in tag):
                    v_score_label_tau[0] = pred_tau[ipred]
                    v_label_tau[0] = label_tau[ipred]                
                    v_score_label_heavy[0] = pred_heavy[ipred]
                    v_label_heavy[0] = label_heavy[ipred]                
                    v_score_label_lightfake[0] = pred_lightfake[ipred]
                    v_label_lightfake[0] = label_lightfake[ipred]                
                elif 'prompt_heavy_lightfake' in tag or 'three' in tag:
                    v_score_label_heavy[0] = pred_heavy[ipred]
                    v_label_heavy[0] = label_heavy[ipred]                
                    v_score_label_lightfake[0] = pred_lightfake[ipred]
                    v_label_lightfake[0] = label_lightfake[ipred]                
                elif 'prompt_heavy_light_fake' in tag or 'four' in tag or ('cms_multiclass' in tag and 'electron' in tag):
                    v_score_label_heavy[0] = pred_heavy[ipred]
                    v_label_heavy[0] = label_heavy[ipred]                
                    v_score_label_light[0] = pred_light[ipred]
                    v_label_light[0] = label_light[ipred]                
                    v_score_label_fake[0] = pred_fake[ipred]
                    v_label_fake[0] = label_fake[ipred]                
                tr.Fill()
        
        f.Write()
        f.Close()
