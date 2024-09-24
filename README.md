# yolov8-onnx.cpp
[YOLOv8](https://github.com/ultralytics/ultralytics) using [onnx.cpp](https://github.com/han-minhee/onnx.cpp) and [stb image library](https://github.com/nothings/stb), without ONNX Runtime or OpenCV. All the necessary files are included in the repository and you only need the basic build environment and protobuf library on your system.

![Sample inference result](/sample/people1_output.jpg)

## Build and Run
By default, this repository contains test data files, which are not necessary for normal build and run. To clone the repository including the test data, just run:

```bash
git clone --recursive https://github.com/han-minhee/yolov8-onnx.cpp.git
```

However, if you want to clone the repository without the test data files:
```bash
mkdir yolov8-onnx.cpp && cd yolov8-onnx.cpp
git init
git remote add origin https://github.com/han-minhee/yolov8-onnx.cpp.git
git config core.sparseCheckout true
echo '/*' > .git/info/sparse-checkout
echo '!/tests/data/*' >> .git/info/sparse-checkout
git pull origin main
git submodule update --init --recursive
```

### Sample Run
This will build and run the sample inference using `/sample/yolov8n.onnx` and `/sample/people1.jpg`, generate `/sample/people1_output.jpg` and `/sample/people1_output.txt` files.

```bash
mkdir build && cd build
cmake -G Ninja ..
cmake --build .
./yolov8
```

### Full Test
This repo contains gtests and related data. If you want to run the test, you should clone the npy and onnx files too. After build, just run 

```bash
ctest
```

The `session_test` is the comprehensive one that runs the whole session with a fixed input, and compare the results with the results from running the input using ONNX Runtime.

## Python Scripts
The python scripts are used for building the onnx.cpp and this example.

- `0_download_model.py` downloads the YOLOv8 model file and convert it to an ONNX file.
- `1_modify_onnx.py` modifies the onnx file so that we can get the intermediate tensors.
- `2_export_npy.py` runs the model on a demo input, exports the tensor values to npy files, to `./python_scripts/exports`

These files are used to validate the implementation of each operators and the whole model inference result.

## Credits
[YOLOv8](https://github.com/ultralytics/ultralytics)

[stb image library](https://github.com/nothings/stb)
