[toc]

这是yolov5的几个加速实现，包括opencv dnn调用、openvino加速、tensorrt加速。也使用下mnn，ncnn（没在移动端尝试）

> ubuntu16.4
>
> python 3.7
>
> cuda10.2+cudnn7.6.5
>
> pytorch 1.7.1
>
> openvino_2020.3.341(最后一个版本支持ubuntu16.4)
>
> tensorrt 7.0.0.11
>
> 

# 生成yolo5 onnx文件

## 生成yolov5参数文件

下载yolov5源码，自行编写代码另存yolov5纯网络参数文件，放置到models目录下

## 转换onnx

假设上一步生成的文件为yolov5s_param.pth，终端执行

> python export_onnx.py --weight yolov5s_param.pth --cfg yolov5s.yaml --inputsize=512

执行完毕后onnx文件保存在run/onnx_file目录下

# dnn 调用

--data 参数传递测试图片路径或者视频文件

> python cv_dnn_infer.py --inputsize 512 --onnx yolov5s.onnx 
> python cv_dnn_infer.py --inputsize 512 --onnx yolov5s.onnx --data images
>
> python cv_dnn_infer.py --inputsize 512 --onnx yolov5s.onnx --data 1.avi

# openvino加速

根据onnx自行生成openvino需要的xml、bin等文件，在run目录下创建openvino_file文件下，把生成的xml、bin、mapping文件放到此处

> python openvino_acc.py -m yolov5s.xml -d 1.avi

# tensorrt加速

第一次执行会在run目录下创建trt文件，时间耗时会比较久

> python tensorrt_acc.py -o yolov5s.onnx -i 512 -d 1.avi

# mnn

转换模型，run/mnn_file下生成mnn文件

> python mnn_acc.py -t  trans -o 'yolov5s.onnx'
>
> python mnn_acc.py -t  infer -d 1.avi



# ncnn

ncnn编译后有yolov5的加速示例程序，也有链接提供了yolov5的ncnn模型文件。其中ncnn文件的生成在这篇文章中有介绍[链接](https://zhuanlan.zhihu.com/p/275989233)。不过我跑官方的yolov5例程的时候，发现结果跟yolov5 pytorch源码不太一致，因此按链接的步骤重新生成ncnn模型跑一次。下面记录过程（大部分跟知乎链接介绍的一致）。

## yolov5源码下生成onnx文件

> ```text
> python models/export.py --weights yolov5s.pt --img 640 --batch 1
> python -m onnxsim yolov5s.onnx yolov5s-sim.onnx
> ```

## onnx转ncnn

> ```text
> ./onnx2ncnn yolov5s-sim.onnx yolov5s.param yolov5s.bin
> ```

## 修改param文件

找准输入输出 blob 名字，用一个自定义层 YoloV5Focus 连接

param 开头第二行，layer_count 要对应修改，但 blob_count 只需确保大于等于实际数量即可

这时我也发现我生成的param文件中，跟ncnn提供的param文件不一致，比如激活函数换成了swish。



替换后用 ncnnoptimize 过一遍模型，顺便转为 fp16 存储减小模型体积（最后一个参数0 fp32,1 fp16）

> ```text
> ./ncnnoptimize yolov5s.param yolov5s.bin yolov5s-opt.param yolov5s-opt.bin 1
> ```

但是我这样做后增加的YoloV5Focus和input又不见了，要从yolov5s.param复制过来。

## reshape层修改

打开yolov5s.param文件，搜索reshape层，相关参数0=6400/1600/400，都改为0=-1

## 重写一个yolov5示例

在ncnn/examples下新建yolov5_zyk.cpp,把官方例程yolov5.cpp文件复制过来，然后做一些修改。主要是根据param文件修改输出层号,我的cpp改动的层号如下。

> ex.extract("417", out);
>
> ...
>
> ex.extract("437", out);

ncnn/examples下修改CMakeLists.txt文件，增加生成yolov5_zyk可执行文件，重新编译ncnn。

## 测试

bin、param文件放到build目录下，终端执行

> ./examples/yolov5_zyk ../images/dog.jpg

# 测试结果

测试模型yolov5s,动态尺寸是按yolov5的输入方式，输入大小为Nx640或640xN,N为小于640的一个32的公倍数。输入为640x640的是因为改变输入大小时dnn、openvino等暂时无法处理focus模块做到动态尺寸输入。我测试的视频，类型为640动态尺寸就是640x480。类型为640x480的测试是重新生成ONNX测试得到的，可与640动态尺寸做同级比较。

cpu

|    类型    |    输入     |  语言  | FPS  | 模型文件大小 |
| :--------: | :---------: | :----: | :--: | :----------: |
|  pytorch   | 640动态尺寸 | python | 10.2 | 14.8M(fp16)  |
| opencv dnn |   640*640   | python | 4.1  | 29.1M(fp32)  |
|  openvino  |   640*640   | python | 8.8  | 14.8M(fp16)  |
|  openvino  |   640*480   | python |  12  | 14.8M(fp16)  |
|    mnn     |   640*640   | python | 4.6  | 14.6M(fp16)  |
|    ncnn    | 640动态尺寸 |  c++   | 8.2  | 14.6M(fp16)  |

gpu

|             类型             |    输入     |  语言  |  FPS  | 模型文件大小 |
| :--------------------------: | :---------: | :----: | :---: | :----------: |
|         pytorch_fp16         | 640动态尺度 | python | 32.2  | 14.8M(fp16)  |
| pytorch_fp16_cudnn_benchmark | 640动态尺度 | python | 111.1 | 14.8M(fp16)  |
|        tensorrt_fp16         |   640×640   | python | 61.7  | 19.8M(fp16)  |



测试模型yolov5x

cpu

|  类型   |    输入     |  语言  | FPS  |
| :-----: | :---------: | :----: | :--: |
| pytorch | 640动态尺寸 | python | 1.9  |

gpu

|             类型             |    输入     |  语言  | FPS  |
| :--------------------------: | :---------: | :----: | :--: |
|         pytorch_fp16         | 640动态尺寸 | python | 3.2  |
| pytorch_fp16_cudnn_benchmark | 640动态尺寸 | python | 16.8 |
|        tensorrt_fp16         |   640*640   | python | 25.5 |

