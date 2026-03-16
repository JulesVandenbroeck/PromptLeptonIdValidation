
mkdir -p out/

config=electron_mclass_part.txt
modelfile=model_particletransformer_electron_2022_mclass.pt
model=python/partileTransformerModel.py
outname=electron_mclass_part

python3 ../../../weaver-core/weaver/train.py -c ${config} -n ${model} -m ${modelfile} --export-onnx out/${outname}.onnx


config=muon_mclass_part.txt
modelfile=model_particletransformer_muon_2022_mclass.pt
model=python/partileTransformerModel.py
outname=muon_mclass_part

python3 ../../../weaver-core/weaver/train.py -c ${config} -n ${model} -m ${modelfile} --export-onnx out/${outname}.onnx