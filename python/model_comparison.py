
import onnxruntime as ort
import numpy as np
import yaml
import torch
import argparse
import onnx

def evaluate_onnx_model(session, data):
    # Run inference
    outputs = session.run(None, data)
    return outputs[0]


def evaluate_pytorch_model(model, data):
    tensor_data = {k: torch.from_numpy(v).float() for k, v in data.items()}
    # Run inference
    with torch.no_grad():
        output = model(**tensor_data)
    return output.numpy()


def dummy_data(type = "muon", length=1):
    # Generate some dummy data for testing

    # import format for the model input
    if type == "muon":
        with open("muon_mclass_part.txt", "r") as f:
            data_format = yaml.safe_load(f)
    else:
        with open("electron_mclass_part.txt", "r") as f:
            data_format = yaml.safe_load(f)

    dummy_data = {feature: np.float32(np.random.rand(
        length,
        len(data_format["inputs"][feature]["vars"]),
        data_format["inputs"][feature]["length"],
    )) for feature in data_format["inputs"]}

    return dummy_data  # Example input shape (1, 10)


def main(type="muon", ntests=100):
    print("ONNX version:", onnx.__version__)
    print("ONNXRuntime version:", ort.__version__)
    print("PyTorch version:", torch.__version__)

    # Path to your ONNX model
    if type == "electron":
        onnx_model_path = "out/electron_mclass_part.onnx"
    else:
        onnx_model_path = "out/muon_mclass_part.onnx"
    
    # Path to your PyTorch model
    pytorch_model_path = "model_particletransformer_" + type + "_2022_mclass.pt"


    # Load the ONNX model
    # Create session options and explicitly set thread counts to avoid
    # ONNXRuntime attempting to set CPU affinity (which can fail in some
    # container/cgroup/cpuset environments).
    sess_options = ort.SessionOptions()
    # Use environment variables if provided, otherwise default to 1
    try:
        sess_options.intra_op_num_threads = int(os.environ.get("ORT_INTRA_THREADS", "1"))
    except Exception:
        sess_options.intra_op_num_threads = 1
    try:
        sess_options.inter_op_num_threads = int(os.environ.get("ORT_INTER_THREADS", "1"))
    except Exception:
        sess_options.inter_op_num_threads = 1

    # Create the session with the options set. Passing providers here ensures
    # the CPU execution provider is used consistently.
    session = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])


    # Load the PyTorch model
    from partileTransformerModel import CustomParticleTransformer


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


    # run the tests 
    tests = ntests
    relative_diff = np.zeros((tests, 5))
    for i in range(tests):
        data = dummy_data(type, length=1)
        # Evaluate ONNX model
        onnx_output = evaluate_onnx_model(session, data)

        # Evaluate PyTorch model
        pytorch_output = evaluate_pytorch_model(model, data)

        relative_diff[i] = (onnx_output - pytorch_output)/onnx_output

    print("Relative Difference:", np.mean(relative_diff, axis=0))


if __name__ == "__main__":
    # add a CLAs
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["electron", "muon"], default="muon")
    parser.add_argument("--ntests", type=int, default=100)
    args = parser.parse_args()

    main(type=args.type, ntests=args.ntests)
