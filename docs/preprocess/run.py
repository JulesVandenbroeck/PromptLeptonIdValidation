#!/bin/env python3

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

import os, sys, glob
import json
import uproot
import vector
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)
from sklearn.model_selection import train_test_split

eps = 1e-8

# 1GB in bytes
MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024 
file_counter = 0

def get_shard_path(base_path, alg, subset, count):
    return base_path.replace('.root', f'_{subset}{alg}_{count}.root')

HL_vars = None

def _clip(a, a_min, a_max):
    try:
        return np.clip(a, a_min, a_max)
    except ValueError:
        return ak.unflatten(np.clip(ak.flatten(a), a_min, a_max), ak.num(a))

def _pad(a, maxlen, value=0, dtype='float32'):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x

def build_features_and_labels(a, transform_features=True, alg=''):
    
    pf_nan_mask = ~np.isnan(a['PF_dzSig'])

    for branch in a.fields:
        if branch.startswith('PF_'):
            a[branch] = a[branch][pf_nan_mask]

    if options.lepton == 'muon': lepton_pt = ak.to_numpy(ak.flatten(a['MuonSelected_LepGood_pt']))[:, np.newaxis]
    else: lepton_pt = ak.to_numpy(ak.flatten(a['ElectronSelected_pt']))[:, np.newaxis]
    
    # pf_vectors
    a['PF_pt_rel'] = ak.where(lepton_pt > 0, a['PF_pt']/lepton_pt, 0)
    a['PF_px'] = a['PF_pt_rel'] * np.cos(a['PF_phi_rel'])
    a['PF_py'] = a['PF_pt_rel'] * np.sin(a['PF_phi_rel'])
    a['PF_pz'] = a['PF_pt_rel'] * np.sinh(a['PF_eta_rel'])
    a['PF_energy'] = a['PF_pt_rel'] * np.cosh(a['PF_eta_rel'])

    # sv_vectors
    a['SV_pt_rel'] = a['SV_pt'] / lepton_pt
    a['SV_px'] = a['SV_pt_rel'] * np.cos(a['SV_phi_rel'])
    a['SV_py'] = a['SV_pt_rel'] * np.sin(a['SV_phi_rel'])
    a['SV_pz'] = a['SV_pt_rel'] * np.sinh(a['SV_eta_rel'])
    a['SV_energy'] = a['SV_pt_rel'] * np.cosh(a['SV_eta_rel'])
    
    a['PF_mask'] = ak.ones_like(a['PF_pt'])
    a['SV_mask'] = ak.ones_like(a['SV_pt'])

    # pf_features
    a['PF_pt_log'] = np.log(a['PF_pt'] + eps)
    a['PF_dxySig_log'] = np.arcsinh(a['PF_dxySig'])
    a['PF_dzSig_log'] = np.arcsinh(a['PF_dzSig'])

    # sv_features
    a['SV_pt_log'] = np.log(a['SV_pt'] + eps)
    a['SV_dlenSig_log'] = np.log(a['SV_dlenSig'] + eps)
    a['SV_dxy_log'] = np.log(a['SV_dxy'] + eps)
    a['SV_mass_log'] = np.log(a['SV_mass'] + eps)
    if options.lepton == 'electron': a['SV_nTracks'] = a['SV_ntracks']

    # highlevel_features
    if options.lepton == 'muon':        
        a['Lepton_eta'] = a['MuonSelected_LepGood_eta']
        a['Lepton_jetPNet'] = a['MuonSelected_LepGood_jetPNet']
        a['Lepton_pt_log'] = np.log(a['MuonSelected_LepGood_pt'] + eps)
        a['Lepton_jetPtRatio'] = a['MuonSelected_LepGood_jetPtRatio']
        a['Lepton_jetPtRelv2_log'] = np.log(a['MuonSelected_LepGood_jetPtRelv2'] + eps)
        a['Lepton_jetNDauChargedMVASel'] = a['MuonSelected_LepGood_jetNDauChargedMVASel']
        a['Lepton_miniRelIsoCharged_log'] = np.log(a['MuonSelected_LepGood_miniRelIsoCharged'] + eps)
        a['Lepton_miniRelIsoNeutral_log'] = np.log(a['MuonSelected_LepGood_miniRelIsoNeutral'] + eps)
        a['Lepton_pfRelIso03_all_log'] = np.log(a['MuonSelected_LepGood_pfRelIso03_all'] + eps)
        a['Lepton_sip3d'] = a['MuonSelected_sip3d']
        a['Lepton_dxy'] = a['MuonSelected_dxy']
        a['Lepton_dz'] = a['MuonSelected_dz']
        a['Lepton_mvaId'] = a['MuonSelected_mvaId']
        a['Lepton_segmentComp'] = a['MuonSelected_segmentComp']
        a['Lepton_ptError_Rel_log'] = np.log(a['MuonSelected_ptError_Rel'] + eps)
        a['Lepton_calEnergy_had_log'] = np.arcsinh(a['MuonSelected_calEnergy_had'])
        a['Lepton_calEnergy_em_log'] = np.arcsinh(a['MuonSelected_calEnergy_em'])
        a['Lepton_lostPixelHits'] = a['MuonSelected_lostPixelHits']
        a['Lepton_Valid_pixel'] = a['MuonSelected_Valid_pixel']
        a['Lepton_promptMVA'] = a['MuonSelected_mvaTTH']
        a['Lepton_mediumId'] = a['MuonSelected_mediumId']
        a['Lepton_tightId'] = a['MuonSelected_tightId']
        a['Lepton_mediumPromptId'] = a['MuonSelected_mediumPromptId']

        HL_vars_muon = ['Lepton_pt_log', 'Lepton_eta', 'Lepton_dxy', 'Lepton_dz', 'Lepton_sip3d', 'Lepton_segmentComp', 'Lepton_jetPNet', 'Lepton_jetPtRatio', 'Lepton_jetNDauChargedMVASel', 'Lepton_miniRelIsoNeutral_log', 'Lepton_jetPtRelv2_log', 'Lepton_miniRelIsoCharged_log', 'Lepton_pfRelIso03_all_log', 'Lepton_mvaId', 'Lepton_ptError_Rel_log', 'Lepton_calEnergy_had_log', 'Lepton_calEnergy_em_log', 'Lepton_lostPixelHits', 'Lepton_Valid_pixel', 'Lepton_promptMVA', 'Lepton_mediumId', 'Lepton_tightId', 'Lepton_mediumPromptId']
        HL_vars = HL_vars_muon
        
    else:
        a['Lepton_eta'] = a['ElectronSelected_eta']
        a['Lepton_jetPNet'] = a['ElectronSelected_jetbtag_PNet']
        a['Lepton_pt_log'] = np.log(a['ElectronSelected_pt'] + eps)
        a['Lepton_jetPtRatio'] = a['ElectronSelected_jetPtRatio']
        a['Lepton_jetPtRelv2_log'] = np.log(a['ElectronSelected_jetPtRelv2'] + eps)
        a['Lepton_jetNDauChargedMVASel'] = a['ElectronSelected_jetNDauCharged']
        a['Lepton_miniRelIsoCharged_log'] = np.log(a['ElectronSelected_minisoch'] + eps)
        a['Lepton_miniRelIsoNeutral_log'] = np.log(a['ElectronSelected_minisonh'] + eps)
        a['Lepton_pfRelIso03_all_log'] = np.log(a['ElectronSelected_pfRelIso03_all'] + eps)
        a['Lepton_mvaId'] = a['ElectronSelected_mvaNoIso']
        a['Lepton_passConversionVeto'] = a['ElectronSelected_passConversionVeto']
        a['Lepton_eoverp_log'] = np.log(a['ElectronSelected_eoverp'] + eps)
        a['Lepton_fbrem_log'] = np.arcsinh(a['ElectronSelected_fbrem'])
        a['Lepton_dr03HcalDepth1TowerSumEt_Rel'] = a['ElectronSelected_dr03HcalDepth1TowerSumEt_Rel']
        a['Lepton_supcl_phiWidth'] = a['ElectronSelected_supcl_phiWidth']
        a['Lepton_supcl_etaWidth'] = a['ElectronSelected_supcl_etaWidth']
        a['Lepton_sigmaiphiiphi'] = a['ElectronSelected_sigmaiphiiphi']
        a['Lepton_sigmaietaieta'] = a['ElectronSelected_sigmaietaieta']
        a['Lepton_e1x5bye5x5'] = a['ElectronSelected_e1x5bye5x5']
        a['Lepton_r9full'] = a['ElectronSelected_r9full']
        a['Lepton_hcaloverecal_log'] = np.log(a['ElectronSelected_hcaloverecal'] + eps)
        a['Lepton_dEtaInSeed'] = a['ElectronSelected_dEtaInSeed']
        a['Lepton_deltaetacltrkcalo'] = a['ElectronSelected_deltaetacltrkcalo']
        a['Lepton_closeTrackNLayers'] = a['ElectronSelected_closeTrackNLayers']
        a['Lepton_lostHits'] = a['ElectronSelected_lostHits']
        a['Lepton_numberOfValidPixelHits'] = a['ElectronSelected_numberOfValidPixelHits']
        a['Lepton_e_ECAL'] = a['ElectronSelected_e_ECAL']
        a['Lepton_sip3d'] = a['ElectronSelected_sip3d']
        a['Lepton_dxy'] = a['ElectronSelected_dxy']
        a['Lepton_dz'] = a['ElectronSelected_dz']
        a['Lepton_convDist'] = a['ElectronSelected_convDist']
        a['Lepton_convDcot'] = a['ElectronSelected_convDcot']
        a['Lepton_convRadius'] = a['ElectronSelected_convRadius']
        a['Lepton_ptError_Rel_log'] = np.log(a['ElectronSelected_ptError_Rel'] + eps)
        a['Lepton_convVtxFitProb'] = a['ElectronSelected_convVtxFitProb']
        a['Lepton_isGsfCtfConsistent'] = a['ElectronSelected_isGsfCtfConsistent']
        a['Lepton_isGsfScPixConsistent'] = a['ElectronSelected_isGsfScPixConsistent']
        a['Lepton_promptMVA'] = a['ElectronSelected_mvaTTH']

        HL_vars_electron = ['Lepton_pt_log', 'Lepton_eta', 'Lepton_dxy', 'Lepton_dz', 'Lepton_sip3d', 'Lepton_jetPNet', 'Lepton_jetPtRatio', 'Lepton_jetNDauChargedMVASel', 'Lepton_miniRelIsoNeutral_log', 'Lepton_jetPtRelv2_log', 'Lepton_miniRelIsoCharged_log', 'Lepton_pfRelIso03_all_log', 'Lepton_mvaId', 'Lepton_passConversionVeto', 'Lepton_eoverp_log', 'Lepton_fbrem_log', 'Lepton_dr03HcalDepth1TowerSumEt_Rel', 'Lepton_supcl_phiWidth', 'Lepton_supcl_etaWidth', 'Lepton_sigmaiphiiphi', 'Lepton_sigmaietaieta', 'Lepton_e1x5bye5x5', 'Lepton_r9full', 'Lepton_hcaloverecal_log', 'Lepton_dEtaInSeed', 'Lepton_deltaetacltrkcalo', 'Lepton_closeTrackNLayers', 'Lepton_lostHits', 'Lepton_numberOfValidPixelHits', 'Lepton_e_ECAL', 'Lepton_sip3d', 'Lepton_ptError_Rel_log', 'Lepton_convDist', 'Lepton_convDcot', 'Lepton_convRadius', 'Lepton_convVtxFitProb', 'Lepton_isGsfCtfConsistent', 'Lepton_isGsfScPixConsistent', 'Lepton_promptMVA']
        HL_vars = HL_vars_electron
        
    if transform_features:
        for v in ['PF_pt', 'PF_dxySig', 'PF_dzSig', \
                  'SV_pt', 'SV_dlenSig', 'SV_dxy', 'SV_mass']:
            mean, scale = get_robust_constants(a[v+'_log'], use_log=True)
            a[v+'_log'] = (a[v+'_log'] - mean) * scale

        vv = ['Lepton_pt_log', 'Lepton_jetPtRelv2_log', 'Lepton_miniRelIsoCharged_log', 'Lepton_miniRelIsoNeutral_log', 'Lepton_pfRelIso03_all_log']
        for v in vv:
            hl_mean, hl_scale = get_robust_constants(a[v], use_log=False)
            a[v] = (a[v] - hl_mean) * hl_scale                

    feature_list = {
        'PF_vars': ['PF_eta_rel', 'PF_phi_rel', 'PF_pt_log', 'PF_px', 'PF_py', 'PF_pz', 'PF_energy', 'PF_charge', 'PF_isElectron', 'PF_isMuon', 'PF_isNeutralHadron', 'PF_isPhoton', 'PF_isChargedHadron', 'PF_fromPV', 'PF_puppiWeightNoLep', 'PF_hcalFraction', 'PF_dzSig_log', 'PF_dxySig_log', 'PF_trackerLayersWithMeasurement', 'PF_numberOfPixelHits', 'PF_mask'],
        'SV_vars': ['SV_eta_rel', 'SV_phi_rel', 'SV_pt_log', 'SV_px', 'SV_py', 'SV_pz', 'SV_energy', 'SV_mass_log', 'SV_dlenSig_log', 'SV_dxy_log', 'SV_ndof', 'SV_chi2', 'SV_cospAngle', 'SV_nTracks', 'SV_mask'],
        'HL_vars': HL_vars_muon if options.lepton == 'muon' else HL_vars_electron
    }

    out = {}
    if alg == '':
        for n in feature_list['PF_vars']:
            out[n] = _pad(a[n], maxlen=60).to_numpy()
        for n in feature_list['SV_vars']:
            out[n] = _pad(a[n], maxlen=5).to_numpy()
    else:
        for n in feature_list['PF_vars']:
            padded = ak.pad_none(a[n], target=60, clip=True)
            out[n] = ak.to_numpy(ak.fill_none(padded, 0.0))
        for n in feature_list['SV_vars']:
            padded = ak.pad_none(a[n], target=5, clip=True)
            out[n] = ak.to_numpy(ak.fill_none(padded, -1.0))
    for n in feature_list['HL_vars']:
        out[n] = ak.to_numpy(a[n]).astype('float32')

    # binary
    out['label_prompt_binary'] = ak.to_numpy((a['genPartFlav'] == 1) | (a['genPartFlav'] == 15)).astype('int32')
    out['label_nonprompt_binary'] = ak.to_numpy((a['genPartFlav'] != 1) & (a['genPartFlav'] != 15)).astype('int32')
    # prompt-tau-heavy-light-fake
    out['label_prompt_mclass'] = ak.to_numpy((a['genPartFlav'] == 1)).astype('int32')
    out['label_tau_mclass'] = ak.to_numpy((a['genPartFlav'] == 15)).astype('int32')
    out['label_heavy_mclass'] = ak.to_numpy((a['genPartFlav'] == 5)).astype('int32')
    out['label_light_mclass'] = ak.to_numpy((a['genPartFlav'] == 4) | (a['genPartFlav'] == 3)).astype('int32')
    out['label_fake_mclass'] = ak.to_numpy((a['genPartFlav'] != 1) & (a['genPartFlav'] != 15) & (a['genPartFlav'] != 5) & (a['genPartFlav'] != 4) & (a['genPartFlav'] != 3)).astype('int32')
    
    return out, HL_vars

def get_robust_constants(array, use_log=False):
    flat_data = ak.to_numpy(ak.flatten(array, axis=None))
    flat_data = flat_data[np.isfinite(flat_data)]
    
    if len(flat_data) == 0:
        return 0.0, 1.0

    if use_log:
        flat_data = np.log(np.abs(flat_data) + eps)

    mean = np.median(flat_data)
    
    q25, q75 = np.percentile(flat_data, [25, 75])
    iqr = q75 - q25
    
    if iqr > 1e-6:
        scale = 1.0 / iqr
    else:
        std = np.std(flat_data)
        scale = 1.0 / std if std > 1e-6 else 1.0
    
    return mean, scale

def normalize_variable(array, mean=None, scale=None):
    if mean is None or scale is None:
        mean, scale = get_normalization_constants(array)
        
    return (array - mean) * scale, mean, scale

from optparse import OptionParser

def main(argv = None):

    if argv == None:
        argv = sys.argv[1:]

    home = os.getcwd()

    usage = "usage: %prog [options]\n Run preprocessing of the input variables for training"

    parser = OptionParser(usage)
    parser.add_option("--year", default="2022", choices=["2016", "2016APV", "2017", "2018", "2022", "2022EE", "2023", "2023BPix", "2024"], help="year of data taking [default: %default]")
    parser.add_option("--output", default="processed_data.root", help="output file [default: %default]")
    parser.add_option("--lepton", default="muon", choices=["electron", "muon"], help="lepton type [default: %default]")
    parser.add_option("--test", default=0.1, type=float, help="fraction of statistics to be used in test [default: %default]")

    (options, args) = parser.parse_args(sys.argv[1:])

    return options

if __name__ == '__main__':

    options = main()

    if options.year == "2016": year_tag = "RunIISummer20UL16"
    elif options.year == "2016APV": year_tag = "RunIISummer20UL16APV"
    elif options.year == "2017": year_tag = "RunIISummer20UL17"
    elif options.year == "2018": year_tag = "RunIISummer20UL18"
    elif options.year == "2022": year_tag = "Run3Summer22"
    elif options.year == "2022EE": year_tag = "Run3Summer22EE"
    elif options.year == "2023": year_tag = "Run3Summer23"
    elif options.year == "2023BPix": year_tag = "Run3Summer23BPix"
    elif options.year == "2024": year_tag = "Run3Winter24"
    
    finput = "/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/train_sl/"+options.lepton+"_"+year_tag+"*.root"
#    finput = "/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/train/"+options.lepton+"_"+year_tag+"*.root"
    input_path = finput + ":Events"

    for alg in ['', '_bdt']:

        file_counter = 0

        train_path = get_shard_path(options.output, alg, 'train', file_counter)
        test_path = get_shard_path(options.output, alg, 'test', file_counter)

        f_train = uproot.recreate(train_path)
        f_test = uproot.recreate(test_path)
        
        first_chunk = True

        for chunk in uproot.iterate(input_path, step_size="100MB", library="ak"):
            num_events = len(chunk)            
            indices = np.arange(num_events)

            # Split indices (90% train, 10% test)
            idx_train, idx_test = train_test_split(indices, test_size=options.test, random_state=42)
                
            # Process Train and Test subsets in a loop
            for subset_label, idx in [('train', idx_train), ('test', idx_test)]:
                ch = chunk[idx]
                
                # Kinematic selection
                if options.lepton == 'muon':
                    muon_pt = ak.fill_none(ak.firsts(ch['MuonSelected_LepGood_pt']), 0)
                    muon_eta = ak.fill_none(ak.firsts(ch['MuonSelected_LepGood_eta']), 0)        
                    mask = (muon_pt > 5) & (abs(muon_eta) < 2.4)
                else:
                    electron_pt = ak.fill_none(ak.firsts(ch['ElectronSelected_pt']), 0)
                    electron_eta = ak.fill_none(ak.firsts(ch['ElectronSelected_eta']), 0)        
                    mask = (electron_pt > 5) & (abs(electron_eta) < 2.5)

                filtered_chunk = ch[mask]
                if len(filtered_chunk) == 0:
                    continue
        
                processed, HL_vars = build_features_and_labels(filtered_chunk, False, alg)

                # Flatten for BDT if necessary
                data_to_write = processed
                if alg != '':
                    flattened = {}
                    for name, array in processed.items():
                        if array.ndim > 1:
                            for i in range(array.shape[1]):
                                flattened[f"{name}_{i}"] = array[:, i]
                        else:
                            flattened[name] = array
                    data_to_write = flattened

                # Determine which file and initialization flag to use
                target_file = f_train if subset_label == 'train' else f_test

                if first_chunk:
                    branch_dims = {n: (a.dtype, a.shape[1:]) if a.ndim > 1 else a.dtype 
                                       for n, a in data_to_write.items()}
                    f_train.mktree("tree", branch_dims)
                    f_test.mktree("tree", branch_dims)
                    first_chunk = False

                target_file["tree"].extend(data_to_write)

            if os.path.exists(train_path) and os.path.getsize(train_path) > MAX_FILE_SIZE:
                f_train.close()
                f_test.close()

                os.system("mv "+os.getenv("_CONDOR_SCRATCH_DIR")+"/*.root /pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/preprocess_sl/.")
#                os.system("mv "+os.getenv("_CONDOR_SCRATCH_DIR")+"/*.root /pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/preprocess/.")
                
                file_counter += 1
                first_chunk = True # Reset for new file schema initialization
                
                train_path = get_shard_path(options.output, alg, 'train', file_counter)
                test_path = get_shard_path(options.output, alg, 'test', file_counter)
                f_train = uproot.recreate(train_path)
                f_test = uproot.recreate(test_path)
                    
        f_train.close()
        f_test.close()

        os.system("mv "+options.output.replace(".root", "*.root")+" /pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/preprocess_sl/.")
#        os.system("mv "+options.output.replace(".root", "*.root")+" /pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/preprocess/.")

    test_shards = ['/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/preprocess_sl/'+options.lepton+'_'+options.year+'_test_0.root']
#    test_shards = ['/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/preprocess/'+options.lepton+'_'+options.year+'_test_0.root']

    tree_data = uproot.concatenate(
        [f"{s}:tree" for s in test_shards],
        entry_stop=10000,
        library="ak"
    )
    
    os.system('rm -rf pics; mkdir pics')

    stats_output = {}

    variables = {
        "high_level": HL_vars,
        "pf_features": ["PF_pt_log", "PF_charge", "PF_isElectron", "PF_isMuon", "PF_isNeutralHadron", "PF_isPhoton", "PF_isChargedHadron", "PF_fromPV", "PF_puppiWeightNoLep", "PF_hcalFraction", "PF_dzSig_log", "PF_dxySig_log", "PF_trackerLayersWithMeasurement", "PF_numberOfPixelHits"],
        "pf_points": ["PF_eta_rel", "PF_phi_rel"],
        "pf_vectors": ["PF_px", "PF_py", "PF_pz", "PF_energy"],
        "sv_features": ["SV_pt_log", "SV_mass_log", "SV_dlenSig_log", "SV_dxy_log", "SV_ndof", "SV_chi2", "SV_cospAngle", "SV_nTracks"],
        "sv_points": ["SV_eta_rel", "SV_phi_rel"],
        "sv_vectors": ["SV_px", "SV_py", "SV_pz", "SV_energy"],
    }
    label_names = ["label_prompt_binary", "label_nonprompt_binary", "label_all"]

    colors = {
        "label_prompt_binary": "blue",
        "label_nonprompt_binary": "red",
        "label_all": "lightgray"
    }

    labels = tree_data[[l for l in label_names if l != "label_all"]]

    for cl in variables.keys():
        for var in variables[cl]:
            full_raw_data = tree_data[var]
    
            plt.figure(figsize=(8, 6))
    
            for lname in label_names:

                if lname == "label_all": class_ev_mask = ak.ones_like(full_raw_data, dtype=bool)
                else: class_ev_mask = ak.values_astype(labels[lname], bool)
            
                class_data = full_raw_data[class_ev_mask]
        
                if cl in ['pf_points', 'pf_vectors', 'pf_features']:
                    pt_data = tree_data['PF_pt_log'][class_ev_mask]
                    pt_mask = pt_data > 1e-8
                    class_data = class_data[pt_mask]
                
                elif cl in ['sv_points', 'sv_vectors', 'sv_features']:
                    pt_data = tree_data['SV_pt_log'][class_ev_mask]
                    pt_mask = pt_data > 1e-8
                    class_data = class_data[pt_mask]

                plot_data = ak.to_numpy(ak.flatten(class_data, axis=None))

                if len(plot_data) > 0:
                    plt.hist(plot_data, bins=100, histtype='step', density=True, 
                             label=lname.replace("label_", "").replace("_binary", ""), linewidth=1.5, color=colors.get(lname, "gray"))
                    print(f"Class {lname} | {var}: Mean={np.mean(plot_data):.3f}, Std={np.std(plot_data):.3f}")

                    if lname == 'label_all':
                        if var not in stats_output:
                            stats_output[var] = {
                                "mean": round(float(np.mean(plot_data)), 3),
                                "std": round(float(np.std(plot_data)), 3),
                            }

            plt.xlabel(var, fontsize=18)
            plt.ylabel("Normalized to unity", fontsize=18)
            #    plt.yscale('log')
            plt.legend(loc='upper right', fontsize=16)
#            plt.grid(True, alpha=0.3)
    
            import os
            os.makedirs("pics", exist_ok=True)

            plt.tight_layout()
            plt.savefig(f"pics/{cl}_{var}.pdf")
            plt.close()

    with open("preprocess.json", "w") as f:
        json.dump(stats_output, f, indent=4)

