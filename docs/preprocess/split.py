#!/bin/env python3

import ROOT

for l in ["electron", "muon"]:
    df = ROOT.RDataFrame("Events", "/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/train/"+l+"_Run3Summer22.root")
    df.Range(1000000).Snapshot("Events", "/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/train/"+l+"_Run3Summer22_small.root")
