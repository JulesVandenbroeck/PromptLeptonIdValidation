#!/bin/env python3

import os, sys, glob

lepton = ['electron', 'muon']
#lepton = ['muon']
year = ['2022']
#dest = '/scratch/gent/vo/002/gvo00240/vsc47410'
dest = '/scratch/gent/vo/002/gvo00240/vsc47410/sl'

#fpath = '/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/preprocess/'
fpath = '/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/preprocess_sl/'
for l in lepton:
    for y in year:
        all_files = glob.glob(fpath+l+'_'+y+'_*.root')
        filtered_files = [f for f in all_files if 'bdt' not in os.path.basename(f)]
        if filtered_files:
            files_str = " ".join(filtered_files)
            cmd = f'scp -o ControlMaster=auto -o ControlPath=~/.ssh/ssh-%r@%h:%p -i ~/.ssh/id_rsa_vsc {files_str} vsc47410@login.hpc.ugent.be:{dest}/.'
            os.system(cmd)
