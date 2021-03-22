#-*-coding:utf-8-*-
import sys, os
import argparse
import cv2
import numpy as np
from decode import YoloPro
import time

if __name__=='__main__':
    rundir = 'run/onnx_file'

    parser = argparse.ArgumentParser()
    parser.add_argument('-o','--onnx', type=str, default='yolov5s.onnx', help="onnx_file file")
    parser.add_argument('-i','--inputsize', default=640, type=int, help='input size')
    parser.add_argument('-c', '--conf', default=0.25,  help='confidence threshold')
    parser.add_argument('-n', '--nms_thres', default=0.45,  help='nms threshold')
    parser.add_argument('-d','--data', default='1.avi',help='images source,iamge dir or video')
    args=parser.parse_args()

    args.onnx = os.path.join(rundir, args.onnx)
    assert os.path.exists(args.onnx)
    net = cv2.dnn.readNet(args.onnx)

    yolo_pro=YoloPro(inputsize=args.inputsize, confThreshold=args.conf, nmsThreshold=args.nms_thres, class_file='class.names')

    if not os.path.isdir(args.data):
        cap=cv2.VideoCapture(args.data)
        infertime=0
        count=0
        fps=0
        if cap.isOpened():
            print('press \'q\' to quit!')
            while True:
                ret,img=cap.read()
                input = yolo_pro.get_input(img)

                # Do inference
                t0 = time.time()
                net.setInput(input)
                out = net.forward(net.getUnconnectedOutLayersNames())[0]
                t1=time.time()
                boxs = yolo_pro.decode(img, out)
                t2=time.time()

                infertime=infertime*count+t2-t0
                count+=1
                infertime=infertime/count
                fps=1/infertime
                info = f'FPS:{fps:.1f},infer {(t1 - t0):.3f}s,decode {(t2 - t1):.3f}s'

                yolo_pro.drawPred(img, boxs)
                cv2.putText(img,info,(0,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),thickness=2)
                cv2.imshow('yolov5',img)
                if cv2.waitKey(10)==ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
        else:
            print('can not open video!')
        print(fps)
    else:
        assert os.path.exists(args.data)
        imgsave_dir='run/results'
        if not os.path.exists(imgsave_dir):
            os.makedirs(imgsave_dir)

        for f in os.listdir(args.data):
            src = cv2.imread(os.path.join(args.data,f))
            t0=time.time()
            input = yolo_pro.get_input(src)
            # Do inference
            # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
            net.setInput(input)
            out = net.forward(net.getUnconnectedOutLayersNames())[0]
            boxs=yolo_pro.decode(src,out)
            print(f, 'done,time',time.time()-t0,'s')
            yolo_pro.drawPred(src,boxs)
            cv2.imwrite(os.path.join(imgsave_dir,f),src)