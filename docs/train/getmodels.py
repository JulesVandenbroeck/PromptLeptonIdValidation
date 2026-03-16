#!/bin/env python3

import os, sys

os.system("rm -rf models/*")

for l in ['muon']:
    for y in ['2022']:
        for nw in ['particletransformer']:
            os.system("scp -i ~/.ssh/id_rsa_vsc vsc47410@login.hpc.ugent.be:/scratch/gent/474/vsc47410/leptonid/models/model_"+nw+"_"+l+"_"+y+"_mclass_optimal.onnx models/.")
            os.system("scp -i ~/.ssh/id_rsa_vsc vsc47410@login.hpc.ugent.be:/scratch/gent/474/vsc47410/leptonid/models/preprocess_"+nw+"_"+l+"_"+y+"_mclass_optimal.json models/.")
