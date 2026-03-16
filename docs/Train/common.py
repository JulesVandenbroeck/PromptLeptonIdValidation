from collections import OrderedDict
import glob
import os, sys

variables = {}

variables['Muon'] = {}
variables['Electron'] = {}

mnames = {'v1': []}
for v in ['TOP-UL', 'TOP_binary', 'TOPMvaId_binary', 'TOPPNet_binary', 'TOPUParT_binary', 'TOP_prompt_tau_heavy_lightfake', 'TOP_prompt_heavy_lightfake', 'TOP_prompt_heavy_light_fake']:
    
    ver = 'v1'
    mname = v
    mnames['v1'].append(mname)
    
    variables['Muon'][mname] = {}
    variables['Muon'][mname]['pt'] = 'MuonSelected_LepGood_pt'
    variables['Muon'][mname]['eta'] = 'lep_abseta'
    variables['Muon'][mname]['trackMultClosestJet'] = 'MuonSelected_LepGood_jetNDauChargedMVASel'
    variables['Muon'][mname]['miniIsoCharged'] = 'MuonSelected_LepGood_miniRelIsoCharged'
    variables['Muon'][mname]['miniIsoNeutral'] = 'MuonSelected_LepGood_miniRelIsoNeutral'
    variables['Muon'][mname]['pTRel'] = 'MuonSelected_LepGood_jetPtRelv2'
    variables['Muon'][mname]['ptRatio'] = 'MuonSelected_LepGood_jetPtRatio'
    variables['Muon'][mname]['relIso'] = 'MuonSelected_LepGood_pfRelIso03_all'
    variables['Muon'][mname]['bTagClosestJet'] = 'MuonSelected_LepGood_jetDF'
    variables['Muon'][mname]['sip3d'] = 'MuonSelected_sip3d'
    variables['Muon'][mname]['dxy'] = 'MuonSelected_dxy'
    variables['Muon'][mname]['dz'] = 'MuonSelected_dz'
    variables['Muon'][mname]['idSeg'] = 'MuonSelected_segmentComp'

    if 'binary' in mname or 'prompt' in mname:
        variables['Muon'][mname]['etaAbs'] = variables['Muon'][mname]['eta']
        variables['Muon'][mname]['dxylog'] = variables['Muon'][mname]['dxy']
        variables['Muon'][mname]['dzlog'] = variables['Muon'][mname]['dz']
        variables['Muon'][mname]['segmentCompatibility'] = variables['Muon'][mname]['idSeg']
        if 'PNet' in mname: variables['Muon'][mname]['bTagClosestJet'] = 'MuonSelected_LepGood_jetPNet'
        elif 'UParT' in mname: variables['Muon'][mname]['bTagClosestJet'] = 'MuonSelected_LepGood_jetUParT'
        elif 'MvaId' in mname: variables['Muon'][mname]['mvaId'] = 'MuonSelected_LepGood_mvaId'
    
    variables['Electron'][mname] = {}
    variables['Electron'][mname]['pt'] = 'ElectronSelected_pt'
    variables['Electron'][mname]['eta'] = 'lep_abseta'
    variables['Electron'][mname]['trackMultClosestJet'] = 'ElectronSelected_jetNDauCharged'
    variables['Electron'][mname]['miniIsoCharged'] = 'ElectronSelected_minisoch'
    variables['Electron'][mname]['miniIsoNeutral'] = 'ElectronSelected_minisonh'
    variables['Electron'][mname]['pTRel'] = 'ElectronSelected_jetPtRelv2'
    variables['Electron'][mname]['ptRatio'] = 'ElectronSelected_jetPtRatio'
    variables['Electron'][mname]['relIso'] = 'ElectronSelected_pfRelIso03_all'
    variables['Electron'][mname]['bTagClosestJet'] = 'ElectronSelected_jetbtag'
    variables['Electron'][mname]['sip3d'] = 'ElectronSelected_sip3d'
    variables['Electron'][mname]['dxy'] = 'ElectronSelected_dxy_log'
    variables['Electron'][mname]['dz'] = 'ElectronSelected_dz_log'
    variables['Electron'][mname]['idSeg'] = 'ElectronSelected_mvaNoIso'
#    variables['Electron'][mname]['idSeg'] = 'ElectronSelected_mvaFall17V2noIso'

    if 'binary' in mname or 'prompt' in mname:
        variables['Electron'][mname]['etaAbs'] = variables['Electron'][mname]['eta']
        variables['Electron'][mname]['dxylog'] = variables['Electron'][mname]['dxy']
        variables['Electron'][mname]['dzlog'] = variables['Electron'][mname]['dz']
        variables['Electron'][mname]['mvaNoIso'] = variables['Electron'][mname]['idSeg']
        if 'PNet' in mname: variables['Electron'][mname]['bTagClosestJet'] = 'ElectronSelected_jetbtag_PNet'
        elif 'UParT' in mname: variables['Electron'][mname]['bTagClosestJet'] = 'ElectronSelected_jetbtag_UParT'
        elif 'Iso' in mname: variables['Electron'][mname]['mvaNoIso'] = 'ElectronSelected_mvaIso'

var = {}
var['Muon'] = {}
var['Electron'] = {}

for lep in ['Muon','Electron']:
    
    for mname in mnames['v1']:
        v = variables[lep][mname]
        var[lep][mname] = OrderedDict([('pt',v['pt']), ('eta',v['eta']), ('trackMultClosestJet',v['trackMultClosestJet']), ('miniIsoCharged',v['miniIsoCharged']), ('miniIsoNeutral',v['miniIsoNeutral']),\
        ('pTRel',v['pTRel']), ('ptRatio',v['ptRatio']), ('relIso',v['relIso']), ('bTagClosestJet',v['bTagClosestJet']), ('sip3d',v['sip3d']), ('dxy',v['dxy']), ('dz',v['dz']), ('idSeg',v['idSeg'])])

def getdataset(year):
    
    fpath = '/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/'
    fsamp = 'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8' if year in ['2016', '2016APV', '2017', '2018'] else 'TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8'
    ftag = 'NanoAODv15_PNet_LeptonID_20251215'
    frun = {'2016': 'RunIISummer20UL16', '2016APV': 'RunIISummer20UL16APV', '2017': 'RunIISummer20UL17', '2018': 'RunIISummer20UL18', \
            '2022': 'Run3Summer22', '2022EE': 'Run3Summer22EE', '2023': 'Run3Summer23', '2023BPix': 'Run3Summer23BPix', '2024': 'Run3Winter24'}
    
    f = fpath+'/'+fsamp+'/'+fsamp+'_'+frun[year]+'_'+ftag+'/*/*/tree*.root'
##    f = fpath+'/'+fsamp+'/'+fsamp+'_RunIISummer20UL18_'+ftag+'/*/*/tree*.root'
##    f = fpath+'/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18_'+ftag+'/*/*/tree*.root'
    flist = glob.glob(f)
    return sorted(flist, key=str.lower)[-10:]    
