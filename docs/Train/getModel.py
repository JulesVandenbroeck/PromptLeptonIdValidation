#!/bin/env python3

import os, sys, glob

flav = '*'
year = '2023'
clf = ['binary', 'four', 'three', 'tau']

for cl in clf:
    os.system('scp -i ~/.ssh/id_rsa_vsc vsc47410@login.hpc.ugent.be:/scratch/gent/474/vsc47410/models/*_'+flav+'_'+year+'_'+cl+'.* models/.')
#    cs = glob.glob('models/*four*')
#    for c in cs: os.system('mv '+c+' '+c.replace('four', 'multiclass'))
