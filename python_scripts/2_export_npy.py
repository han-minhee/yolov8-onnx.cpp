import onnx
import onnxruntime as ort
import numpy as np
import os
import json
from onnx import helper, shape_inference, external_data_helper
import argparse

def pack_int4(values):
    packed = np.bitwise_or(values[:, 0::2] << 4, values[:, 1::2])
    return packed

def generate_int4_example(shape):
    int4_values = np.random.randint(
        0, 16, size=(shape[0], shape[1] * 2), dtype=np.uint8
    )
    packed_values = pack_int4(int4_values)
    return packed_values.reshape(shape).astype(np.int8)


def generate_example_input(
    graph_input,
    batch_size,
    sequence_length,
    total_sequence_length,
    past_sequence_length,
):
    name = graph_input.name
    tensor_type = graph_input.type.tensor_type
    elem_type = tensor_type.elem_type
    shape = []

    for dim in tensor_type.shape.dim:
        if dim.dim_param == "batch_size":
            shape.append(batch_size)
        elif dim.dim_param == "sequence_length":
            shape.append(sequence_length)
        elif dim.dim_param == "total_sequence_length":
            shape.append(total_sequence_length)
        elif dim.dim_param == "past_sequence_length":
            shape.append(past_sequence_length)
        else:
            shape.append(dim.dim_value if dim.dim_value > 0 else 1)

    np_type_map = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.UINT8: np.uint8,
        onnx.TensorProto.INT8: np.int8,
        onnx.TensorProto.UINT16: np.uint16,
        onnx.TensorProto.INT16: np.int16,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.BOOL: np.bool_,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.DOUBLE: np.float64,
        onnx.TensorProto.UINT32: np.uint32,
        onnx.TensorProto.UINT64: np.uint64,
        onnx.TensorProto.COMPLEX64: np.complex64,
        onnx.TensorProto.COMPLEX128: np.complex128,
        onnx.TensorProto.BFLOAT16: np.float32,  
    }

    if (
        elem_type == onnx.TensorProto.INT8 and shape[-1] % 2 == 0
    ):  
        example_input = generate_int4_example(shape)
    else:
        np_type = np_type_map.get(elem_type, np.float32)
        if "input_ids" in name.lower() or "token_ids" in name.lower():
            example_input = np.random.randint(0, 32064, size=shape).astype(np_type)
        else:
            example_input = np.random.rand(*shape).astype(np_type)

        if np.issubdtype(np_type, np.integer) and "input_ids" not in name.lower():
            example_input = np.random.randint(0, 100, size=shape).astype(np_type)

    return name, example_input


def save_node_information(onnx_model, output_dir):
    nodes_info = []

    for node in onnx_model.graph.node:
        print(f"Node: {node.name}, Type: {node.op_type}")
        node_info = {
            "name": node.name,
            "op_type": node.op_type,
            "attributes": {
                attr.name: str(onnx.helper.get_attribute_value(attr))
                for attr in node.attribute
            },
            "inputs": list(node.input),
            "outputs": list(node.output),
        }
        nodes_info.append(node_info)
        print(f"Node saved\n")

    with open(os.path.join(output_dir, "nodes_info.json"), "w") as f:
        json.dump(nodes_info, f, indent=4)


def save_inputs_and_outputs(session, inputs, output_dir):    
    inputs_dir = os.path.join(output_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    outputs_dir = os.path.join(output_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    outputs = session.run(None, inputs)
    print(f"Inference done")

    
    for input_name, input_data in inputs.items():
        np.save(
            os.path.join(inputs_dir, f"{sanitize_file_name(input_name)}.npy"),
            input_data,
        )

        shape = list(input_data.shape) if input_data is not None else [1]

        tensor_info = {
            "name": input_name,
            "dims": shape,
            "dtype": str(input_data.dtype),
            "size": int(np.prod(shape)),
        }

        print(
            f"Saving input {input_name} with shape {shape} and dtype {input_data.dtype}"
        )

        if input_data is None:
            print(f"Warning: Output {input_name} is None.")
            continue

        
        json_path = os.path.join(inputs_dir, f"{sanitize_file_name(input_name)}.json")
        with open(json_path, "w") as json_file:
            json.dump(tensor_info, json_file, indent=4)
    
    for i, output in enumerate(outputs):
        output_name = session.get_outputs()[i].name
        np.save(
            os.path.join(outputs_dir, f"{sanitize_file_name(output_name)}.npy"), output
        )

        if output is None:
            print(f"Warning: Output {output_name} is None.")
            continue
        shape = list(output.shape) if output.shape is not None else [1]
        
        tensor_info = {
            "name": output_name,
            "dims": shape,
            "dtype": str(output.dtype),
            "size": int(np.prod(shape)),
        }

        print(
            f"Saving output {output_name} with shape {shape} and dtype {output.dtype}"
        )
        json_path = os.path.join(outputs_dir, f"{sanitize_file_name(output_name)}.json")
        with open(json_path, "w") as json_file:
            json.dump(tensor_info, json_file, indent=4)


def unpack_int4_weights(weight_array, expected_size):
    int8_weights = np.frombuffer(weight_array, dtype=np.int8)
    unpacked_weights = np.zeros(expected_size, dtype=np.float32)  

    
    for i in range(int8_weights.size):
        unpacked_weights[2 * i] = (int8_weights[i] >> 4) & 0xF
        unpacked_weights[2 * i + 1] = int8_weights[i] & 0xF

    if unpacked_weights.size != expected_size:
        print(
            f"Shape mismatch during unpacking. Expected {expected_size} elements, but got {unpacked_weights.size}."
        )
        print(f"Original INT8 size: {int8_weights.size}")
        print(f"Unpacked INT4 size: {unpacked_weights.size}")
        return None

    return unpacked_weights


dtype_map = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.UINT8: np.uint8,
    onnx.TensorProto.INT8: np.int8,
    onnx.TensorProto.UINT16: np.uint16,
    onnx.TensorProto.INT16: np.int16,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.DOUBLE: np.float64,
    onnx.TensorProto.UINT32: np.uint32,
    onnx.TensorProto.UINT64: np.uint64,
    onnx.TensorProto.BOOL: np.bool_,
    onnx.TensorProto.COMPLEX64: np.complex64,
    onnx.TensorProto.COMPLEX128: np.complex128,
    onnx.TensorProto.BFLOAT16: np.float32,
}


def onnx_tensor_dtype_to_string(dtype):
    tensorproto_type_string = onnx.helper.tensor_dtype_to_string(dtype)
    truncated = tensorproto_type_string[12:].lower()

    if truncated == "float":
        return "float32"
    elif truncated == "uint8":
        return "uint8"
    elif truncated == "int8":
        return "int8"
    elif truncated == "uint16":
        return "uint16"
    elif truncated == "int16":
        return "int16"
    elif truncated == "int32":
        return "int32"
    elif truncated == "int64":
        return "int64"
    elif truncated == "bool":
        return "bool"
    elif truncated == "float16":
        return "float16"
    elif truncated == "double":
        return "float64"

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def save_weights(onnx_model, output_dir):
    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    for initializer in onnx_model.graph.initializer:

        dtype = dtype_map.get(initializer.data_type, np.float32)
        weight_array = np.frombuffer(initializer.raw_data, dtype=dtype)
        expected_size = np.prod(initializer.dims)

        if (
            dtype == np.int8 and weight_array.size * 2 == expected_size
        ):  
            unpacked_weights = unpack_int4_weights(weight_array, expected_size)
            if unpacked_weights is not None:
                unpacked_weights = unpacked_weights.reshape(
                    [dim for dim in initializer.dims]
                )
                np.save(
                    os.path.join(weights_dir, f"{initializer.name}.npy"),
                    unpacked_weights,
                )
                save_tensor_info(
                    initializer,
                    output_dir,
                    onnx_tensor_dtype_to_string(initializer.data_type),
                )
            else:
                print(f"Skipping weight {initializer.name} due to shape mismatch.")
        elif weight_array.size == expected_size:
            weight_array = weight_array.reshape([dim for dim in initializer.dims])
            np.save(os.path.join(weights_dir, f"{initializer.name}.npy"), weight_array)
            save_tensor_info(
                initializer,
                output_dir,
                onnx_tensor_dtype_to_string(initializer.data_type),
            )
        else:
            print(
                f"Warning: Skipping weight {initializer.name} due to shape mismatch. "
                f"Expected {expected_size} elements, but got {weight_array.size}."
            )
            continue

def save_tensor_info(initializer, output_dir, dtype):
    tensor_info = {
        "name": initializer.name,
        "dims": list(initializer.dims),  
        "dtype": str(dtype),
        "size": int(np.prod(initializer.dims)),
    }

    tensor_info_dir = os.path.join(output_dir, "weights")
    os.makedirs(tensor_info_dir, exist_ok=True)

    json_path = os.path.join(tensor_info_dir, f"{initializer.name}.json")
    with open(json_path, "w") as json_file:
        json.dump(tensor_info, json_file, indent=4)

def sanitize_file_name(name):
    return name.replace("/", ".").replace("\\", ".").replace(":", ".").lstrip(".")

def save_model_with_external_data(model, output_model_path):

    external_data_helper.load_external_data_for_model(
        model, f"{output_model_path}/cache"
    )
    
    onnx.save_model(
        model,
        output_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="./onnx/yolov8n_modified.onnx",
        help="Path to the ONNX model",
    )

    parser.add_argument(
        "--output",
        default="./exports/yolov8n",
        help="Output directory",
    )

    args = parser.parse_args()
    model_path = args.model
    output_path = args.output

    print(f"Original model path: {model_path}")
    print(f"Output directory: {output_path}")

    os.makedirs(output_path, exist_ok=True)

    print(f"Preparing inputs for the model...")
    onnx_model = onnx.load(model_path)

    inputs = {}
    for input_info in onnx_model.graph.input:
        if input_info.name == "images":
            input_data = np.ones((1,3,640,640), dtype=np.float32)
            
            inputs["images"] = input_data

    print(f"Running inference and saving inputs/outputs...")
    session = ort.InferenceSession(model_path)
    save_inputs_and_outputs(session, inputs, output_path)

    print(f"Saving node information")
    save_node_information(onnx_model, output_path)
    
    print(f"Saving weights")
    save_weights(onnx_model, output_path)
    print(
        "Inputs, outputs, node information, weights, and tensor info have been saved to the output directory."
    )

if __name__ == "__main__":
    main()
