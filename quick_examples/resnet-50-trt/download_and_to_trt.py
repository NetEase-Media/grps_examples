# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2024/6/19
# Brief  Convert torch resnet50 model to tensorrt model.

import os
import timeit
import urllib.request

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision
import cv2

MODEL_PATH = './data/'
ONNX_MODEL_PATH = MODEL_PATH + 'resnet50.onnx'
TRT_MODEL_PATH = MODEL_PATH + 'resnet50.trt'
MAX_BATCH_SIZE = 128

SYNSET_URL = "".join(
    [
        "https://gist.githubusercontent.com/zhreshold/",
        "4d0b62f3d01426887599d4f7ede23ee5/raw/",
        "596b27d23537e5a1b5751d2b0481ef172f58b539/",
        "imagenet1000_clsid_to_human.txt",
    ]
)
SYNSET_NAME = MODEL_PATH + "imagenet1000_clsid_to_human.txt"


def preprocess_input(img):
    image = np.float32(img) / 255.0
    image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    image = image.transpose((0, 3, 1, 2))
    print('image shape: {}'.format(image.shape))
    return image


def load_data():
    print('Loading image data...')
    img = cv2.imread('./data/tabby.jpeg')
    img = cv2.resize(img, (224, 224))
    img = np.array(img)[np.newaxis, :].astype("float32")
    img = img[:, :, :, ::-1]  # BGR -> RGB
    data = preprocess_input(img)
    return data


def load_torch_model():
    print('Loading pytorch model...')
    model = torchvision.models.resnet50(pretrained=True).eval()
    model.eval()
    print('Loaded pytorch resnet-50 model.')
    return model


def torch_2_onnx(torch_model):
    print('Transfer torch model to onnx model...')
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ["x"]
    output_names = ["495"]
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    torch.onnx.export(torch_model, dummy_input, ONNX_MODEL_PATH, verbose=True,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes={'x': [0], '495': [0]},
                      opset_version=8)


def onnx_2_trt_engine_by_trtexec():
    """convert onnx model to tensorrt engine by trtexec command."""
    print('Apply tensorrt optimizing...')
    save_trt_engine_cmd = 'trtexec --onnx={} --saveEngine={} --minShapes=x:1x3x224x224' \
                          ' --maxShapes=x:{}x3x224x224 --optShapes=x:1x3x224x224 --fp16 --verbose' \
        .format(ONNX_MODEL_PATH, TRT_MODEL_PATH, MAX_BATCH_SIZE)
    os.system(save_trt_engine_cmd)
    return True


class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.device_input = cuda.mem_alloc(1 * 3 * 224 * 224 * 4)
        self.batches = iter([np.ascontiguousarray(load_data())])
        self.cache_file = MODEL_PATH + '/calibrator_cache'

    def get_algorithm(self):
        return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2

    def get_batch(self, names):
        try:
            # Assume self.batches is a generator that provides batch data.
            data = next(self.batches)
            # Assume that self.device_input is a device buffer allocated by the constructor.
            cuda.memcpy_htod(self.device_input, data)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def get_batch_size(self):
        return 1

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def onnx_2_trt_engine_by_api():
    """convert onnx model to tensorrt engine by tensorrt api."""
    logger = trt.Logger(trt.Logger.VERBOSE)

    # parse onnx model
    builder = trt.Builder(logger)
    # print('platform_has_tf32: {}, platform_has_fast_fp16: {}, platform_has_fast_int8: {}'
    #       .format(builder.platform_has_tf32, builder.platform_has_fast_fp16, builder.platform_has_fast_int8))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    # success = parser.parse_from_file(ONNX_MODEL_PATH)
    success = parser.parse(open(ONNX_MODEL_PATH, 'rb').read())
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success:
        print('parse onnx model failed.')
        return False
    print('parse onnx model successfully.')

    print('Creating trt serialized engining...')
    # build trt config
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    # mixed precision
    # config.set_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.FP16)
    # config.set_flag(trt.BuilderFlag.INT8)  # int8 quantization
    # config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
    # # int8 quantization calibrator, provide some representative inputs for trt to calculate activation range
    # # for quantization
    # config.int8_calibrator = MyCalibrator()
    # dynamic shape
    op_profile = builder.create_optimization_profile()
    op_profile.set_shape(network.get_input(0).name, min=trt.Dims([1, 3, 224, 224]), opt=trt.Dims([1, 3, 224, 224]),
                         max=trt.Dims([MAX_BATCH_SIZE, 3, 224, 224]))
    config.add_optimization_profile(op_profile)

    # save trt engine
    serialized_engine = builder.build_engine(network, config)
    with open(TRT_MODEL_PATH, "wb") as f:
        f.write(serialized_engine.serialize())

    return True


def load_trt_engine():
    """load tensorrt engine and create execution context."""
    print('Loading tensorrt engine...')
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    with open(TRT_MODEL_PATH, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()  # create execution context
    for i in range(engine.num_bindings):
        print("----binding {}".format(i))
        print('name: {}'.format(engine.get_binding_name(i)))
        print('shape: {}'.format(engine.get_binding_shape(i)))
        print('dtype: {}'.format(engine.get_binding_dtype(i)))
        print('vec_dim: {}'.format(engine.get_binding_vectorized_dim(i)))
        print('comps: {}'.format(engine.get_binding_components_per_element(i)))
        print('is_shape: {}'.format(engine.is_shape_binding(i)))
        print('is_input: {}'.format(engine.binding_is_input(i)))
        if engine.binding_is_input(i):
            print('get_profile_shape: {}'.format(engine.get_profile_shape(0, i)))

    return engine, context


def trt_malloc(inp_data):
    """malloc tensorrt input and output cpu and gpu memory."""
    h_input = np.array(inp_data)
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(
        MAX_BATCH_SIZE * 3 * 224 * 224 * 4)  # can allocate a larger batch size to reuse of this memory
    d_output = cuda.mem_alloc(MAX_BATCH_SIZE * 1000 * 4)  # can allocate a larger batch size to reuse of this memory
    return h_input, d_input, d_output


def trt_infer(input_idx, h_input, d_input, d_output, trt_ctx, stream):
    """pure-trt infer."""
    batch_size = h_input.shape[0]
    # set true input shape
    trt_ctx.set_binding_shape(input_idx, h_input.shape)
    # copy input data from cpu to gpu
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # execute trt engine
    trt_ctx.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # copy output data from gpu to cpu
    h_output = cuda.pagelocked_empty((batch_size, 1000), dtype=np.float32)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # synchronize stream
    stream.synchronize()
    return h_output


def confirm_output(data, torch_model, trt_engine, trt_ctx):
    """confirm the output of torch and tensorrt model."""
    urllib.request.urlretrieve(SYNSET_URL, SYNSET_NAME)
    with open(SYNSET_NAME) as f:
        synset = eval(f.read())

    input_x = torch.from_numpy(data)
    torch_output = torch_model(input_x)
    torch_top5 = torch.argsort(torch_output, 1).cpu().detach().numpy()[0][-1:-6:-1]
    print("Torch output top-5 id: {}, predict class name: {}".format(torch_top5, synset[torch_top5[0]]))

    input_idx = trt_engine['x']
    # output_idx = trt_engine['495']
    h_input, d_input, d_output = trt_malloc(np.ascontiguousarray(data))
    stream = cuda.Stream()
    trt_out = trt_infer(input_idx, h_input, d_input, d_output, trt_ctx, stream)
    top5_trt = np.argsort(trt_out[0])[-1:-6:-1]
    print("Pure-trt output top-5 id: {}, predict class name: {}".format(top5_trt, synset[top5_trt[0]]))


def compare_infer_speed(data, torch_model, trt_engine, trt_ctx):
    """compare the infer speed of torch and tensorrt model."""
    timing_number = 10
    timing_repeat = 10
    input_x = torch.from_numpy(data)
    torch_speed = (
            np.array(timeit.Timer(lambda: torch_model(input_x.cuda()))
                     .repeat(repeat=timing_repeat, number=timing_number))
            * 1000 / timing_number
    )
    torch_speed = {
        "mean": np.mean(torch_speed),
        "median": np.median(torch_speed),
        "std": np.std(torch_speed),
    }

    input_idx = trt_engine['x']
    h_input, d_input, d_output = trt_malloc(np.ascontiguousarray(data))
    stream = cuda.Stream()
    trt_speed = (
            np.array(timeit.Timer(lambda: trt_infer(input_idx, h_input, d_input, d_output, trt_ctx, stream))
                     .repeat(repeat=timing_repeat, number=timing_number))
            * 1000 / timing_number
    )
    trt_speed = {
        "mean": np.mean(trt_speed),
        "median": np.median(trt_speed),
        "std": np.std(trt_speed),
    }

    print('torch_speed: {}\ntrt_speed:{}'.format(torch_speed, trt_speed))


if __name__ == '__main__':
    data = load_data()  # load image data

    # load torch resnet-50 model
    torch_model = load_torch_model()

    # convert to onnx.
    torch_2_onnx(torch_model)

    # convert onnx to tensorrt engine by trtexec command
    onnx_2_trt_engine_by_trtexec()

    # convert onnx to tensorrt engine by tensorrt api
    # onnx_2_trt_engine_by_api()

    # load tensorrt engine
    trt_engine, trt_ctx = load_trt_engine()

    # confirm the output of torch and tensorrt
    confirm_output(data, torch_model, trt_engine, trt_ctx)

    # compare the infer speed of torch and tensorrt model
    compare_infer_speed(data, torch_model.cuda(), trt_engine, trt_ctx)
