#!/usr/bin/env python3

import os
import sys
import subprocess
import htcondor

import sys
from optparse import OptionParser

def main(argv = None):

    if argv == None:
        argv = sys.argv[1:]

    home = os.getcwd()

    usage = "usage: %prog [options]\n Script to submit jobs for preprocessing input variables"

    parser = OptionParser(usage)
    parser.add_option("--path", default="jobs", help="output directory [default: %default]")
    parser.add_option("--year", default="2022", help="year of data taking [default: %default]")
    parser.add_option("--cpu", default=8, help="number of cpus [default: %default]")
    parser.add_option("--lepton", default="electron,muon", help="Lepton types [default: %default]")
#    parser.add_option("--lepton", default="muon", help="Lepton types [default: %default]")

    (options, args) = parser.parse_args(sys.argv[1:])

    return options

def job(jname, outname, home, output, lepton, year, storepath, fsh):

    j = "#!/bin/bash\n\n"
    
    j += "echo \"Start: $(/bin/date)\"\n"
    j += "echo \"User: $(/usr/bin/id)\"\n"
    j += "echo \"Node: $(/bin/hostname)\"\n"
    j += "echo \"CPUs: $(/bin/nproc)\"\n"
    j += "echo \"Directory: $(/bin/pwd)\"\n"
    
    j += "cd "+home+"\n"

    j += "cd "+output+"\n"
    j += "python3 "+home+"/./run.py --output ${_CONDOR_SCRATCH_DIR}/"+jname+".root --lepton "+lepton+" --year "+year+"\n"

    with open(fsh, 'w') as f:
        f.write(j)
        
    os.system('chmod u+x '+fsh)

if __name__ == '__main__':

    options = main()

    home = os.getcwd()

    outpath = home+'/'+options.path
    storepath = '/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/preprocess_sl'
#    storepath = '/pnfs/iihe/cms/store/user/kskovpen/PNet_LeptonID/preprocess'
    os.system('rm -rf /tmp/kskovpen; mkdir /tmp/kskovpen')

    if os.path.isdir(outpath):
        os.system('rm -rf '+outpath)

    os.system('mkdir '+outpath)
    
    schedd = htcondor.Schedd()
    
    for ilepton in options.lepton.split(','):
        for iyear in options.year.split(','):
            
            print(ilepton+':', iyear)

            os.system('rm -rf '+storepath+'/'+ilepton+'_'+iyear+'*')
            
            jname = ilepton+'_'+iyear
            os.system('mkdir '+outpath+'/'+jname)
            outname = outpath+'/'+jname+'/'+jname
            outlog = outname+'.log'

            job(jname, outname, home, outpath+'/'+jname+'/', ilepton, iyear, storepath, outname+'.sh')
        
            js = htcondor.Submit({\
                                  "executable": outname+'.sh', \
                                  "request_cpus": options.cpu, \
                                  "output": outname+'.out', \
                                  "error": outname+'.err', \
                                  "log": outname+'.log', \
                                  })
        
            with schedd.transaction() as shd:
                cluster_id = js.queue(shd)
