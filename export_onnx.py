#-*-coding:utf-8-*-
import yaml
from common import *
import os
import argparse

class FocusContract(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(FocusContract, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        N, C, H, W = x.size()
        x = x.view(N, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 5, 3, 1, 2, 4).contiguous()   #different with source,but same as torch.cat operation
        x = x.view(N, C * 4, H // 2, W // 2)
        #torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.conv(x)

class MyDetect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(MyDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs,c,h, w = x[i].shape
            x[i] = x[i].view(self.na, self.no, h, w).permute(0, 2, 3, 1).contiguous()
            x[i] = x[i].view(-1, self.no)

        x=torch.cat(x, 0)
        x=x.sigmoid()
        return x

class MyYoloV5(nn.Module):
    def __init__(self,nc=80,anchors=()):
        super().__init__()
        #修改每个组件参数从而得到s，m，l，x各类模型
        self.m0=FocusContract(c1=3,c2=32,k=3)
        #self.m0 = Focus(c1=3, c2=32, k=3)

        self.m1=Conv(c1=32,c2=64,k=3,s=2)
        self.m2=C3(c1=64,c2=64,n=1)

        self.m3 = Conv(c1=64, c2=128, k=3, s=2)
        self.m4 = C3(c1=128, c2=128, n=3)

        self.m5 = Conv(c1=128, c2=256, k=3, s=2)
        self.m6 = C3(c1=256, c2=256, n=3)

        self.m7=Conv(c1=256, c2=512, k=3, s=2)
        self.m8=SPP(c1=512, c2=512, k=[5, 9, 13])
        self.m9=C3(c1=512,c2=512,n=1,shortcut=False)

        self.m10=Conv(c1=512, c2=256, k=1, s=1)
        self.m11=nn.Upsample(None,scale_factor=2,mode='nearest')
        self.m12=Concat(dimension=1)
        self.m13=C3(c1=512,c2=256,n=1,shortcut=False)

        self.m14 = Conv(c1=256, c2=128, k=1, s=1)
        self.m15 = nn.Upsample(None, scale_factor=2, mode='nearest')
        self.m16 = Concat(dimension=1)
        self.m17 = C3(c1=256, c2=128, n=1, shortcut=False)

        self.m18 = Conv(c1=128, c2=128, k=3, s=2)
        self.m19 = Concat(dimension=1)
        self.m20 = C3(c1=256, c2=256, n=1, shortcut=False)

        self.m21 = Conv(c1=256, c2=256, k=3, s=2)
        self.m22 = Concat(dimension=1)
        self.m23 = C3(c1=512, c2=512, n=1, shortcut=False)

        self.m24=MyDetect(nc,anchors,[128, 256, 512])

    def forward(self,x):
        x=self.m0(x)

        x=self.m1(x)
        x=self.m2(x)

        x = self.m3(x)
        d3 = self.m4(x)#4

        x = self.m5(d3)
        d4 = self.m6(x)#6

        x=self.m7(d4)
        x=self.m8(x)
        x=self.m9(x)

        d4_2=self.m10(x)#10
        x=self.m11(d4_2)
        x=self.m12((x,d4))
        x=self.m13(x)

        d3_2 = self.m14(x)#14
        x = self.m15(d3_2)
        x = self.m16((x, d3))
        d3_out = self.m17(x)#17

        x=self.m18(d3_out)
        x=self.m19((x,d3_2))
        d4_out=self.m20(x)#20

        x = self.m21(d4_out)
        x = self.m22((x,d4_2))
        d5_out = self.m23(x)#23

        x=self.m24([d3_out,d4_out,d5_out])
        return x

    def load_pretrained(self,weight='models/yolov5s_param.pth',issource=True):
        #issource: weight file is yolov5's weight file
        src_dict = torch.load(weight)
        d=self.state_dict()
        for (k,v) in d.items():
            i=k.split('.')[0][1:]+'.'
            m_name=k.split('.')[0]+'.'
            src_k=k.replace(m_name,'model.'+i)
            assert d[k].shape==src_dict[src_k].shape
            d[k]=src_dict[src_k]
        self.load_state_dict(d)
        if issource:
            ln=0
            print('deal bn\'s parameters...')
            for m in self.modules():
                if type(m) is Conv and hasattr(m, 'bn'):
                    ln+=1
                    m.bn.eps = 0.001
                    m.bn.momentum = 0.03
            print(f'deal {ln} bn\'s parameters...')

def export_onnx_myyolov5():
    onnxfile = 'yolov5s.onnx'
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    model = MyYoloV5(nc=80, anchors=anchors).eval()
    # Update model,suport onnx_file
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()

    model.load_pretrained()
    device = torch.device('cuda:0')
    model = model.eval().to(device)
    input = torch.randn(1, 3, 640, 640).to(device)
    #torch.onnx.export(model, input, onnxfile, verbose=True)
    torch.onnx.export(model, input, onnxfile, verbose=False, input_names=['images'],output_names=['output'],
                      opset_version=12)
    print('done')

# 根据源码进行了简化
# 1）更换Focus，detect模块，focus要正常导出onnx需使用contract函数，代替切片操作，v5源码contract那里有一点点问题
# 2）读取v5作者给出预训练模型后bn的eps，momentum参数要修改，否则结果不一致
# 3）fuse合并卷积和bn的操作，要不要基本没啥区别

class YoloV5(nn.Module):
    def __init__(self, cfg='yolov5s.yaml'):  # model, input channels, number of classes
        super().__init__()
        ch=3
        with open(cfg) as f:
            self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        self.model, self.save = self.parse_model(self.yaml, ch=[ch])  # model, savelist

    def forward(self, x):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def parse_model(self,d, ch):  # model_dict, input_channels(3)
        anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
        no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
            m = eval(m) if isinstance(m, str) else m  # eval strings
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                except:
                    pass

            n = max(round(n * gd), 1) if n > 1 else n  # depth gain
            if m in [Conv, Bottleneck, SPP, DWConv,MixConv2d, Focus,CrossConv, BottleneckCSP, C3]:
                # repalce Focus to support onnx_file
                if m is Focus:
                    m=FocusContract
                c1, c2 = ch[f], args[0]
                c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3]:
                    args.insert(2, n)
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum([ch[x if x < 0 else x + 1] for x in f])
            elif m is Detect:
                #return nn.Sequential(*layers), sorted(save)
                m=MyDetect
                args.append([ch[x + 1] for x in f])
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
            elif m is Contract:
                m=FocusContract
                c2 = ch[f if f < 0 else f + 1] * args[0] ** 2
            elif m is Expand:
                c2 = ch[f if f < 0 else f + 1] // args[0] ** 2
            else:
                c2 = ch[f if f < 0 else f + 1]

            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            np = sum([x.numel() for x in m_.parameters()])  # number params
            m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
            #print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        return self

def export_onnx(inputsize=640,fuse=True,weight='models/yolov5s_param.pth',cfg='models/yolov5s.yaml',
                onnxfile='models/yolov5s.onnx_file',issource=False):
    assert os.path.exists(weight) and os.path.exists(cfg)
    model=YoloV5(cfg).eval()

    src_dict = torch.load(weight)
    d = model.state_dict()

    for (k, v) in d.items():
        assert d[k].shape == src_dict[k].shape
        d[k] = src_dict[k]
    model.load_state_dict(d)

    # eps momentum is different,bn need eps correct value,only need if load yolov5s~x pretained model
    if issource:
        print('deal bn\'s parameters')
        for m in model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.bn.eps = 0.001
                m.bn.momentum = 0.03
    if fuse:
        model=model.fuse()

    img=torch.rand(1,3,inputsize,inputsize,dtype=torch.float32)
    # Update model,suport onnx_file
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    out=model(img)
    #torch.onnx_file.export(model.cuda(), img.cuda(), onnxfile, verbose=True, opset_version=12,example_outputs=out)

    # opset_version 为稳定版本tensorrt 解析onnx 才不报错
    torch.onnx.export(model.cuda(), img.cuda(), onnxfile, verbose=False, example_outputs=out,input_names=['images'],output_names=['output'],
                        #opset_version=12,
                        #dynamic_axes={'images':[2,3]}
                      )
    print('done')

if __name__=='__main__':
    #export_onnx_myyolov5()

    model_dir='models'
    rundir='run/onnx_file'
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='yolov5s_param.pth')
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml')
    parser.add_argument('--inputsize', default=640, type=int)
    parser.add_argument('--issource', default=True, type=bool,
                        help='is yolov5 pretrained model?')  # you own model set it False
    args = parser.parse_args()
    onnx_file = args.cfg.split('.')[0] + '.onnx'
    onnx_file=os.path.join(rundir,onnx_file)

    args.weight=os.path.join(model_dir,args.weight)
    args.cfg = os.path.join(model_dir, args.cfg)
    export_onnx(args.inputsize, False, args.weight, args.cfg, onnx_file, args.issource)

