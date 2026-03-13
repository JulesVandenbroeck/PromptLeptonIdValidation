
def evaluate_onnx_model(model_path, data):
    import onnxruntime as ort
    import numpy as np

    # Load the ONNX model
    session = ort.InferenceSession(model_path)

    # Run inference
    outputs = session.run(None, data)

    return outputs[0]


def evaluate_pytorch_model(model_path, data):
    import torch
    from partileTransformerModel import CustomParticleTransformer

    # Load the PyTorch model
    kwargs = {
        "pf_input_dim": 14,
        "sv_input_dim": 10,
        "highlevel_dim": 14,
        "num_classes": 5,
        "input_dim": 128,
    }
    model = CustomParticleTransformer(**kwargs)
    model.load_state_dict(torch.load(
        model_path,  map_location=torch.device('cpu')))

    # Set the model to evaluation mode
    model.eval()

    # Prepare input data

    tensor_data = {k: torch.from_numpy(v).float() for k, v in data.items()}
    # Run inference
    with torch.no_grad():
        output = model(**tensor_data)

    return output.numpy()


def dummy_data(length=1):
    import numpy as np
    import yaml
    # Generate some dummy data for testing

    # import format for the model input
    with open("muon_mclass_part.txt", "r") as f:
        data_format = yaml.safe_load(f)

    dummy_data = {feature: np.float32(np.random.rand(
        length,
        len(data_format["inputs"][feature]["vars"]),
        data_format["inputs"][feature]["length"],
    )) for feature in data_format["inputs"]}

    return dummy_data  # Example input shape (1, 10)


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    print("ONNX version:", onnx.__version__)
    print("ONNXRuntime version:", ort.__version__)
    print("PyTorch version:", torch.__version__)

    # Path to your ONNX model
    onnx_model_path = "model_particletransformer_muon_2022_mclass.onnx"
    # Path to your PyTorch model
    pytorch_model_path = "model_particletransformer_muon_2022_mclass.pt"

    tests = 100
    relative_diff = np.zeros((tests, 5))
    for i in range(tests):
        data = dummy_data(length=1)
        # Evaluate ONNX model
        onnx_output = evaluate_onnx_model(onnx_model_path, data)

        # Evaluate PyTorch model
        pytorch_output = evaluate_pytorch_model(pytorch_model_path, data)

        relative_diff[i] = (onnx_output - pytorch_output)/onnx_output

    print("Relative Difference:", np.mean(relative_diff, axis=0))


if __name__ == "__main__":
    main()
