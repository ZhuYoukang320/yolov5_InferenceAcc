#-*-coding:utf-8-*-

import cv2
import numpy as np
import time


class YoloPro():
    def __init__(self,inputsize,confThreshold=0.5, nmsThreshold=0.5,class_file='class.names',dynamic=False):
        with open(class_file, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(self.classes))]
        num_classes = len(self.classes)
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.stride = np.array([8., 16., 32.])

        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5
        self.grid = [np.zeros(1)] * self.nl
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.inputsize=inputsize
        self.inpWidth, self.inpHeight=inputsize,inputsize

        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.dynamic=dynamic

    def get_input(self,img):
        h, w, c = img.shape
        r = min(self.inputsize / h, self.inputsize / w)
        rh, rw = int(round(h * r)), int(round(w * r))

        if self.dynamic:
            self.inpWidth = (rw + 31) // 32 * 32
            self.inpHeight = (rh + 31) // 32 * 32

        keep_ratio_img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)[...,::-1]

        top = (self.inpHeight - rh) // 2
        bottom = self.inpHeight - rh - top
        left = (self.inpWidth - rw) // 2
        right = self.inpWidth - rw - left
        #记录
        self.top=top
        self.left=left
        self.r=r

        trans = cv2.copyMakeBorder(keep_ratio_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=114)
        input = np.ascontiguousarray((trans / 255.).transpose(2, 0, 1)[np.newaxis]).astype(np.float32)

        return input

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        grid=np.stack((xv, yv), 2)
        return grid.reshape(-1,2).astype(np.float32)

    def drawPred(self,frame,outs):
        # Draw a bounding box.
        for classId,conf,[left,top,w,h] in outs:
            right=left+w
            bottom=top+h
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=3)

            label = '%.2f' % conf
            label = '%s:%s' % (self.classes[classId], label)
            # Display the label at the top of the bounding box
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=3)
            top = max(top, labelSize[1])
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=3)

    def decode(self, srcimg,outs):
        t0 = time.time()
        # inference output
        #outs = 1 / (1 + np.exp(-outs)) #sigmoid
        row_ind = 0
        for i in range(self.nl):
            h, w = int(self.inpHeight/self.stride[i]), int(self.inpWidth/self.stride[i])
            length = int(self.na * h * w)#按每个尺度网格大小进行解析

            if self.grid[i].shape!= (h*w,2):
                self.grid[i] = self._make_grid(w, h)

            outs[row_ind:row_ind+length, 0:2] = (outs[row_ind:row_ind+length, 0:2] * 2. - 0.5 + np.tile(self.grid[i],(self.na, 1))) * int(self.stride[i])
            outs[row_ind:row_ind+length, 2:4] = (outs[row_ind:row_ind+length, 2:4] * 2) ** 2 * np.repeat(self.anchor_grid[i],h*w, axis=0)
            row_ind += length

        # 类别置信度计算方式，后面代码不再用objThreshold判断
        outs[:, 5:] *= outs[:, 4:5]  # conf = obj_conf * cls_conf
        t1 = time.time()

        classIds = []
        confidences = []
        boxes = []
        # 修改4,类别使用的sigmoid激活，允许多类别的存在
        offsetboxes = []  # 偏移框
        i, j = np.nonzero(outs[:, 5:] > self.confThreshold)
        outs = np.concatenate((outs[:, :4][i], outs[i, j + 5, None], j[:, None]), 1)  # (n.6)(x,y,w,h,conf,cls)
        maxwh = 4096
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        for detection in outs:
            confidence = detection[4]
            clsid = int(detection[5])
            if confidence > self.confThreshold:
                # 对应原图坐标
                center_x = (detection[0]-self.left)/self.r
                center_y = (detection[1]-self.top)/self.r
                width = int(round(detection[2]/self.r))
                height = int(round(detection[3]/self.r))
                left = int(round(center_x - width / 2))
                top = int(round(center_y - height / 2))
                classIds.append(clsid)
                confidences.append(float(confidence))
                offsetboxes.append([left + maxwh * clsid, top + maxwh * clsid, width, height])  # 按类别偏移
                # 对应原图坐标
                boxes.append([left,top,width,height])

        n = len(offsetboxes)  # number of boxes
        if not n:  # no boxes
            return []
        elif n > max_nms:  # excess boxes
            sort_idx=np.argsort(confidences)[::-1]
            offsetboxes = (np.array(offsetboxes)[sort_idx]).tolist()[:max_nms]
            confidences = (np.array(confidences)[sort_idx]).tolist()[:max_nms]
            classIds = (np.array(classIds)[sort_idx]).tolist()[:max_nms]
            boxes = (np.array(boxes)[sort_idx]).tolist()[:max_nms]


        t2 = time.time()
        indices = cv2.dnn.NMSBoxes(offsetboxes, confidences, 0, self.nmsThreshold)
        retboxs=[]
        for i in indices:
            i =i[0]
            box = boxes[i]
            retboxs.append([classIds[i],confidences[i],box])
        t3 = time.time()
        #print('%.3f,%.3f,%.3f'%(t1-t0,t2-t1,t3-t2))
        return retboxs

if __name__=='__main__':
    img=cv2.imread('images/zidane.jpg')
    h,w,c=img.shape
    s=640
    r=min(s/h,s/w)
    rh,rw=int(round(h*r)),int(round(w*r))
    ih=(rh+31)//32*32
    iw=(rw+31)//32*32
    keep_ratio_img = cv2.resize(img, (rw,rh), interpolation=cv2.INTER_LINEAR)

    top=(ih-rh)//2
    bottom=ih-rh-top
    left=(iw-rw)//2
    right=iw-rw-left
    trans=cv2.copyMakeBorder(keep_ratio_img,top,bottom,left,right,cv2.BORDER_CONSTANT, value=(114,114,114))
    print((w, h), (rw, rh), (iw, ih),(trans.shape[1],trans.shape[0]))
    cv2.imshow('input', trans)
    cv2.waitKey(0)