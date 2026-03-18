import onnxruntime as ort
import numpy as np
import yaml
import json
import torch
import argparse
import onnx
import uproot
from weaver.utils.data.config import DataConfig

from model_comparison import evaluate_pytorch_model, evaluate_onnx_model
from partileTransformerModel import CustomParticleTransformer


def torch_model(type):

    pytorch_model_path = "model_particletransformer_" + type + "_2022_mclass.pt"

    # import format for the model input
    if type == "muon":
        with open("muon_mclass_part.txt", "r") as f:
            data_format = yaml.safe_load(f)
    else:
        with open("electron_mclass_part.txt", "r") as f:
            data_format = yaml.safe_load(f)

    kwargs = {
        "pf_input_dim": len(data_format["inputs"]["pf_features"]["vars"])+len(data_format["inputs"]["pf_points"]["vars"]),
        "sv_input_dim": len(data_format["inputs"]["sv_features"]["vars"])+len(data_format["inputs"]["sv_points"]["vars"]),
        "highlevel_dim": len(data_format["inputs"]["high_level"]["vars"]),
        "num_classes": len(data_format["labels"]["value"]),
        "input_dim": 128,
    }
    model = CustomParticleTransformer(**kwargs)
    model.load_state_dict(torch.load(
        pytorch_model_path,  map_location=torch.device('cpu')))

    # Set the model to evaluation mode
    model.eval()

    return model


def onnx_session(type):

    # Path to your ONNX model
    if type == "electron":
        onnx_model_path = "out/electron_mclass_part__softmax.onnx"
    else:
        onnx_model_path = "out/muon_mclass_part__softmax.onnx"

    # Path to your PyTorch model
    pytorch_model_path = "model_particletransformer_" + type + "_2022_mclass.pt"

    # Load the ONNX model
    # Create session options and explicitly set thread counts to avoid
    # ONNXRuntime attempting to set CPU affinity (which can fail in some
    # container/cgroup/cpuset environments).
    sess_options = ort.SessionOptions()
    # Use environment variables if provided, otherwise default to 1
    try:
        sess_options.intra_op_num_threads = int(
            os.environ.get("ORT_INTRA_THREADS", "1"))
    except Exception:
        sess_options.intra_op_num_threads = 1
    try:
        sess_options.inter_op_num_threads = int(
            os.environ.get("ORT_INTER_THREADS", "1"))
    except Exception:
        sess_options.inter_op_num_threads = 1

    # Create the session with the options set. Passing providers here ensures
    # the CPU execution provider is used consistently.
    session = ort.InferenceSession(
        onnx_model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

    return session


def preprocess_data(type, input_file, limit_events=None):
    from weaver.utils.dataset import _preprocess, _read_files

    # import format for the model input
    if type == "muon":
        with open("preprocess_part_muon_2022_mclass.json", "r") as f:
            data_format = json.load(f)
    else:
        with open("preprocess_part_electron_2022_mclass.json", "r") as f:
            data_format = json.load(f)

    sampler_options = {
        "shuffle": False,
        "reweight": False,
        'training': False,
        'up_sample': True,
        'weight_scale': 1,
        'max_resample': 10,
    }

    if type == "muon":
        data_config = DataConfig.load("muon_mclass_part.txt")
    else:
        data_config = DataConfig.load("electron_mclass_part.txt")

    data = {}
    events = uproot.open(input_file)["tree"].arrays(library="np")
    n_events = len(events[list(events.keys())[0]])

    for feature in data_format["input_names"]:
        print(f"Processing feature: {feature}")
        vars = data_format[feature]["var_infos"]
        data[feature] = np.zeros((n_events, len(
            vars), data_format[feature]["var_length"]))
        feature_mask = {
            "pf_features": "PF_mask",
            "pf_points": "PF_mask",
            "sv_features": "SV_mask",
            "sv_points": "SV_mask",
        }
        # Default to all True
        mask = np.ones(
            (n_events, data_format[feature]["var_length"]), dtype=bool)
        if feature in feature_mask:
            mask_name = feature_mask[feature]
            if mask_name in events.keys():
                mask = events[mask_name]

        for i, (var, var_info) in enumerate(vars.items()):
            if var not in events.keys():
                raise ValueError(
                    f"Variable {var} not found in the input file.")

            data[feature][:, i] = events[var]

    # loading data using weaver
    load_branches = data_config.train_load_branches if sampler_options[
        'training'] else data_config.test_load_branches
    table = _read_files([input_file], load_branches, (0, 1), treename=data_config.treename,
                        branch_magic=data_config.branch_magic, file_magic=data_config.file_magic)

    processed_data = _preprocess(table, data_config, sampler_options)
    classes = [events[class_name]
               for class_name in data_format["output_names"]]
    return processed_data, np.transpose(classes), events


def add_roc_curve(classes, outputs, output_file, labels=None, prompt_class=None):

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    plt.figure()
    for i, output in enumerate(outputs):
        if prompt_class is not None:
            # Assuming the first two classes are prompt leptons
            y_true = np.any([classes[:, c] == 1 for c in prompt_class], axis=0)
        else:
            y_true = classes[:, 0] == 1
        # Assuming class 1 is the positive class

        # Sum of prompt lepton scores
        fpr, tpr, _ = roc_curve(y_true, np.sum(
            [output[:, c] for c in prompt_class], axis=0))
        roc_auc = auc(fpr, tpr)
        label = labels[i] if labels is not None else f"Model {i}"
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

    plt.xlim([10**(-4), 1.0])
    plt.ylim([0.7, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xscale('log')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_file)
    plt.close()


def main(type, input_file, output_file):

    model = torch_model(type)
    data, classes, events = preprocess_data(type, input_file)

    # evaluate data in intervals
    torch_output = np.zeros_like(classes, dtype=np.float32)
    onnx_output = np.zeros_like(torch_output)

    import uproot
    weaver_output = uproot.open(
        "preprocess/muon_2022_test_output_20000event_without_softmax.root")["Events"].arrays(library="np")

    weaver_output = np.transpose(
        [weaver_output[c] for c in weaver_output.keys() if c.startswith("score_")])

    features = {
        '_pf_features': "pf_features",
        '_pf_points': "pf_points",
        '_pf_vectors': "pf_vectors",
        '_pf_mask': "pf_mask",
        '_sv_features': "sv_features",
        '_sv_points': "sv_points",
        '_sv_vectors': "sv_vectors",
        '_sv_mask': "sv_mask",
        '_high_level': "high_level",
    }
    batch = 512
    for i in range(0, classes.shape[0], batch):
        print(f"Evaluating events {i} to {i+batch}...")
        batch_data = {features[k]: data[0][k][i:i+batch]
                      for k in data[0] if k in features}
        torch_output[i:i+batch] = evaluate_pytorch_model(model, batch_data)
        batch_data = {k: np.float32(batch_data[k]) for k in batch_data}
        onnx_output[i:i +
                    batch] = evaluate_onnx_model(onnx_session(type), batch_data)

        if i >= 5000:  # Limit to 5k events for faster evaluation
            torch_output = torch_output[:i+batch]
            onnx_output = onnx_output[:i+batch]
            weaver_output = weaver_output[:i+batch]
            classes = classes[:i+batch]
            break

    add_roc_curve(
        classes,
        [torch_output, onnx_output, weaver_output],
        output_file,
        labels=["torch", "onnx", "weaver"],
        # Assuming the first two classes are prompt leptons
        prompt_class=[0, 1],
    )

    return


if __name__ == "__main__":
    # add a CLAs
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["electron", "muon"], default="muon")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    main(
        type=args.type,
        input_file=args.input_file,
        output_file=args.output_file,
    )
