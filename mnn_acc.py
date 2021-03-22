#-*-coding:utf-8-*-
import MNN
import argparse
import os
import cv2
from decode import YoloPro
import time
import numpy as np

#onnx -> mnn
def onnx_to_mnn(onnxfile,mnnfile):
    #cmd='mnnconvert -h'
    cmd=f'mnnconvert -f ONNX  --modelFile {onnxfile} --MNNModel {mnnfile} --fp16 True --bizCode MNN'
    os.system(cmd)

def inference_demo():
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter("run/mnn_file/yolov5s.mnn")
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    image = np.random.rand(640,640,3)
    #cv2 read as bgr format
    image = image[..., ::-1]
    #change to rgb format
    image = cv2.resize(image, (640, 640))
    #resize to mobile_net tensor size
    image = image - (103.94, 116.78, 123.68)
    image = image * (0.017, 0.017, 0.017)
    #preprocess it
    image = image.transpose((2, 0, 1))
    #change numpy data type as np.float32 to match tensor's format
    image = image.astype(np.float32)
    #cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    tmp_input = MNN.Tensor((1, 3, 640, 640), MNN.Halide_Type_Float,\
                    image, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    #constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    tmp_output = MNN.Tensor((25200, 85), MNN.Halide_Type_Float, np.ones([25200, 85]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
    output_tensor.copyToHostTensor(tmp_output)
    # print("expect 983")
    # print("output belong to class: {}".format(np.argmax(tmp_output.getData())))
    output_data = np.array(output_tensor.getData()).reshape((25200, 85))
    print(output_data.shape)

    #https://blog.csdn.net/weixin_45250844/article/details/113824512
    # output_tensor.printTensorData()
    # output_data = np.array(output_tensor.getData()).reshape(-1,85)
    # print(output_data.shape)



if __name__=='__main__':
    rundir = 'run/onnx_file'
    mnndir='run/mnn_file'
    if not os.path.exists(mnndir):
        os.makedirs(mnndir)

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, default='infer', choices=['trans','infer'],help="get mnn file or mnn infer")
    parser.add_argument('-o','--onnx', type=str, default='yolov5s.onnx', help="onnx_file file")
    parser.add_argument('-c', '--conf', default=0.25,  help='confidence threshold')
    parser.add_argument('-n', '--nms_thres', default=0.45,  help='nms threshold')
    parser.add_argument('-d','--data', default='1.avi',help='images source,iamge dir or video')
    args=parser.parse_args()

    mnnfile=args.onnx.split('.')[0]+'.mnn'
    mnnfile=os.path.join(mnndir,mnnfile)
    args.onnx = os.path.join(rundir, args.onnx)
    assert os.path.exists(args.onnx)
    if args.type=='trans':
        onnx_to_mnn(args.onnx,mnnfile)
    else:
        #inference_demo()
        # mnn,
        interpreter = MNN.Interpreter(mnnfile)
        session = interpreter.createSession()
        input_tensor = interpreter.getSessionInput(session)
        b,c,h,w=input_tensor.getShape()
        yolo_pro = YoloPro(inputsize=h, confThreshold=args.conf, nmsThreshold=args.nms_thres,
                           class_file='class.names')

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

                    #inferrence
                    t0 = time.time()
                    tmp_input = MNN.Tensor((1, 3, h, w), MNN.Halide_Type_Float, input, MNN.Tensor_DimensionType_Caffe)
                    input_tensor.copyFrom(tmp_input)
                    interpreter.runSession(session)
                    output_tensor = interpreter.getSessionOutput(session)
                    output_tensor.printTensorData()
                    out = np.array(output_tensor.getData()).reshape(-1,85)
                    t1 = time.time()
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
                src = cv2.imread(os.path.join(args.data, f))
                t0 = time.time()
                input = yolo_pro.get_input(src)
                # inferrence
                tmp_input = MNN.Tensor((1, 3, h, w), MNN.Halide_Type_Float, \
                                       input, MNN.Tensor_DimensionType_Caffe)
                input_tensor.copyFrom(tmp_input)
                interpreter.runSession(session)
                output_tensor = interpreter.getSessionOutput(session)
                output_tensor.printTensorData()
                out = np.array(output_tensor.getData()).reshape(-1, 85)

                boxs = yolo_pro.decode(src, out)
                print(f, 'done,time', time.time() - t0, 's')
                yolo_pro.drawPred(src, boxs)
                cv2.imwrite(os.path.join(imgsave_dir, f), src)




