# # --*-- coding:utf-8 --*--
import sys, os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from decode import YoloPro
import argparse
import time

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            #builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_workspace_size = 1 << 30
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx_file is generated with batch size 64. Reshape input to batch size 1
            #network.get_input(0).shape = [1, 3, 608, 608]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def my_get_engine(onnx_file_path, engine_file_path="",max_batchsize=1,fp16_mode=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            #builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_workspace_size = 1 << 30 # 预先分配的工作空间大小,即ICudaEngine执行时GPU最大需要的空间
            builder.max_batch_size = max_batchsize  # 执行时最大可以使用的batchsize
            builder.fp16_mode = fp16_mode

            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx_file is generated with batch size 64. Reshape input to batch size 1
            #network.get_input(0).shape = [1, 3, 608, 608]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            #print(network.num_layers, network.get_layer(network.num_layers - 1).get_output(0).shape)
            # network.mark_output(network.get_layer(network.num_layers -1).get_output(0))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

if __name__=='__main__':
    dir = 'run/onnx_file'
    parser = argparse.ArgumentParser()
    parser.add_argument('-o','--onnx', type=str, default='yolov5s.onnx', help="onnx file")
    parser.add_argument('-i','--inputsize', default=640, type=int, help='input size')
    parser.add_argument('-c', '--conf', default=0.25,  help='confidence threshold')
    parser.add_argument('-n', '--nms_thres', default=0.45,  help='nms threshold')
    parser.add_argument('-d','--data', default='1.avi',help='images source,iamge dir or video')
    args=parser.parse_args()

    args.onnx =os.path.join(dir,args.onnx)
    assert os.path.exists(args.onnx)

    engine_dir= 'run/tensorrt_engines/'
    if not os.path.exists(engine_dir):
        os.makedirs(engine_dir)

    engine_name=args.onnx.split('/')[-1].split('.')[0]+'.trt'
    engine_file_path=engine_dir+engine_name

    yolo_pro=YoloPro(inputsize=args.inputsize, confThreshold=args.conf, nmsThreshold=args.nms_thres, class_file='class.names')
    with my_get_engine(args.onnx, engine_file_path,1,fp16_mode=True) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        if not os.path.isdir(args.data):
            cap = cv2.VideoCapture(args.data)
            infertime = 0
            count = 0
            fps = 0
            if cap.isOpened():
                print('press \'q\' to quit!')
                while True:
                    ret,img=cap.read()
                    input = yolo_pro.get_input(img)
                    # Do inference
                    # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
                    t0 = time.time()
                    inputs[0].host = input
                    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs,stream=stream)
                    t1=time.time()
                    out=trt_outputs[0].reshape((-1,yolo_pro.no))
                    boxs = yolo_pro.decode(img, out)
                    t2=time.time()

                    infertime = infertime * count + t2 - t0
                    count += 1
                    infertime = infertime / count
                    fps = 1 / infertime
                    info = f'FPS:{fps:.1f},infer {(t1 - t0):.3f}s,decode {(t2 - t1):.3f}s'

                    yolo_pro.drawPred(img, boxs)
                    cv2.putText(img, info, (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), thickness=2)
                    cv2.imshow('yolov5', img)
                    if cv2.waitKey(10)==ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
            else:
                print('can not open camera!')

        else:
            assert os.path.exists(args.data)
            imgsave_dir='run/results'
            if not os.path.exists(imgsave_dir):
                os.makedirs(imgsave_dir)
            for f in os.listdir(args.data):
                img = cv2.imread(os.path.join(args.data,f))
                input = yolo_pro.get_input(img)
                # Do inference
                # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
                t0 = time.time()
                inputs[0].host = input
                trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                out=trt_outputs[0].reshape((-1,yolo_pro.no))
                boxs=yolo_pro.decode(img,out)
                print(f, 'done,time',time.time()-t0,'s')
                yolo_pro.drawPred(img,boxs)
                cv2.imwrite(os.path.join(imgsave_dir,f),img)





