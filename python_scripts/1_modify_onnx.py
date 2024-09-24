import onnx
import argparse
def get_output_type_for_node(node):
    type_map = {
        "Conv": onnx.TensorProto.FLOAT,       
        "Relu": onnx.TensorProto.FLOAT,
        "Add": onnx.TensorProto.FLOAT,
        "Mul": onnx.TensorProto.FLOAT,
        "Concat": onnx.TensorProto.FLOAT,
        "Sigmoid": onnx.TensorProto.FLOAT,
        "Shape": onnx.TensorProto.INT64,      
        "Gather": onnx.TensorProto.INT64,     
        "Slice": onnx.TensorProto.INT64,      
    }
    print(f"Using type map for {node.op_type}")
    print(f"Determined type: {type_map.get(node.op_type, None)}")
    return type_map.get(node.op_type, None)

def modify_model_for_intermediate_outputs(model):
    graph = model.graph
    value_info_map = {vi.name: vi for vi in graph.value_info}

    for node in graph.node:
        print(f'Modifying node {node.name}...')
        for i, output in enumerate(node.output):
            
            if output == "":
                output = f"{node.name}_output_{i}"
                node.output[i] = output

            if output not in value_info_map:
                
                output_type = None
                for input_name in node.input:
                    if input_name in value_info_map:
                        output_type = value_info_map[input_name].type.tensor_type.elem_type
                        break

                
                if node.op_type == "Shape":
                    output_type = onnx.TensorProto.INT64  
                elif output_type is None:
                    
                    output_type = get_output_type_for_node(node)

                
                if output_type is None:
                    print(f"Warning: Could not determine the output type for {output}. Using default type float32.")
                    output_type = onnx.TensorProto.FLOAT

                output_value_info = onnx.helper.make_tensor_value_info(output, output_type, None)
                graph.value_info.append(output_value_info)
                value_info_map[output] = output_value_info

            graph.output.append(value_info_map[output])

    if not model.opset_import:
        opset_import = onnx.helper.make_operatorsetid("", 21)
        model.opset_import.append(opset_import)

    if not model.ir_version:
        model.ir_version = onnx.IR_VERSION

    return model



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="./onnx/yolov8n.onnx",
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--output",
        default="./onnx/yolov8n_modified.onnx",
        help="Path to the output ONNX model",
    )

    args = parser.parse_args()
    model_path = args.model
    output_path = args.output

    print(f'Modifying model "{model_path}" and saving it to "{output_path}"...')

    
    model = onnx.load(model_path)

    print(f"Model loaded successfully. Modifying the model...")

    
    modified_model = modify_model_for_intermediate_outputs(model)

    print(f"Model modified successfully. Saving the modified model...")

    
    onnx.save(
        modified_model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
    )

    print(f'Modified model saved to "{output_path}".')

if __name__ == "__main__":
    main()
