#-*-coding:utf-8-*-

#/opt/intel/openvino_2019.3.376/inference_engine/samples/python_samples/object_detection_sample_ssd

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore
from decode import YoloPro
import time

def openvino_simple_infer(args):
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)
    ie = IECore()
    # 检查是否支持所有层
    supported_layers = ie.query_network(net, 'CPU')
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                  format(args.device, ', '.join(not_supported_layers)))
        log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                  "or --cpu_extension command line argument")
        sys.exit(1)
    # -----------------------------------------------------------------------------------------------------
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    print('input:', input_blob, 'output:', out_blob)
    n, c, h, w = net.inputs[input_blob].shape
    images = np.ndarray(shape=(n, c, h, w))
    exec_net = ie.load_network(network=net, device_name='GPU')
    log.info("Creating infer request and starting inference")
    res = exec_net.infer(inputs={input_blob: images})  # 字典，输出层名字和相应
    print(res[out_blob].shape)

#python openvino_acc.py
if __name__ == '__main__':
    # 1)由onnx生成ir文件
    # cd /opt/intel/openvino/deployment_tools/model_optimizer
    # --keep_shape_ops参数方便进行动态尺寸输入
    # python mo.py --input_model /home/zyk/ml_projects/yolov5_InferenceAcc/run/onnx_file/yolov5s.onnx --output_dir=/home/zyk/ml_projects/yolov5_InferenceAcc/run/openvino_file --data_type=FP16 --keep_shape_ops

    #2）inference
    dir='run/openvino_file'
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument("-m", "--model", default='yolov5s.xml',
                      help="Path to an .xml file with a trained model.", type=str)
    parser.add_argument('-c', '--conf', default=0.25, help='confidence threshold')
    parser.add_argument('-n', '--nms_thres', default=0.45, help='nms threshold')
    parser.add_argument('-d','--data', default='1.avi',help='images source,iamge dir or video')

    args=parser.parse_args()

    # openvino_simple_infer()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))

    model_xml=os.path.join(dir,model_xml)
    model_bin=os.path.join(dir,model_bin)

    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    # 检查是否支持所有层
    device = 'CPU'
    supported_layers = ie.query_network(net, device)
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                  format(args.device, ', '.join(not_supported_layers)))
        log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                  "or --cpu_extension command line argument")
        sys.exit(1)
    # -----------------------------------------------------------------------------------------------------
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    print('input:', input_blob, 'output:', out_blob)
    n, c, h, w = net.inputs[input_blob].shape
    #https: // blog.csdn.net / github_28260175 / article / details / 107128484
    #net.reshape({input_blob: (1, 3, 480, 640)})#focus模块的存在导致无法修改输入？
    exec_net = ie.load_network(network=net, device_name=device)
    log.info("Creating infer request and starting inference")

    yolo_pro = YoloPro(inputsize=max(h,w), confThreshold=args.conf,
                       nmsThreshold=args.nms_thres, class_file='class.names',dynamic=True)

    if not os.path.isdir(args.data):
        cap = cv2.VideoCapture(args.data)
        infertime = 0
        count = 0
        fps = 0
        if cap.isOpened():
            print('press \'q\' to quit!')
            while True:
                ret, img = cap.read()
                input = yolo_pro.get_input(img)

                # Do inference
                t0 = time.time()
                res = exec_net.infer(inputs={input_blob: input})  # 字典，输出层名字和相应
                t1 = time.time()
                out = res[out_blob].reshape((-1, yolo_pro.no))
                boxs = yolo_pro.decode(img, out)
                t2 = time.time()

                infertime = infertime * count + t2 - t0
                count += 1
                infertime = infertime / count
                fps = 1 / infertime
                info = f'FPS:{fps:.1f},infer {(t1 - t0):.3f}s,decode {(t2 - t1):.3f}s'

                yolo_pro.drawPred(img, boxs)
                cv2.putText(img, info, (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), thickness=2)
                cv2.imshow('yolov5', img)
                if cv2.waitKey(10) == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break

        else:
            print('can not open video!')

    else:
        assert os.path.exists(args.data)
        imgsave_dir = 'run/results'
        if not os.path.exists(imgsave_dir):
            os.makedirs(imgsave_dir)
        for f in os.listdir(args.data):
            img = cv2.imread(os.path.join(args.data, f))
            input = yolo_pro.get_input(img)
            # Do inference
            # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
            t0 = time.time()
            res = exec_net.infer(inputs={input_blob: input})  # 字典，输出层名字和相应
            out = res[out_blob].reshape((-1, yolo_pro.no))
            boxs = yolo_pro.decode(img, out)
            print(f, 'done,time', time.time() - t0, 's')
            yolo_pro.drawPred(img, boxs)
            cv2.imwrite(os.path.join(imgsave_dir, f), img)