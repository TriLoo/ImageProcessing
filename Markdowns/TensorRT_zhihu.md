# TensorRT 高性能深度学习推理引擎

## 推理与训练的不同

* 推理过程中网络全职固定下来：

  * 可以进行计算图优化

  * 输入输出固定、可以进行Memory优化

* 推理过程中的Batch Size非常小

* 推理过程可以使用低精度数值，如FP16, INT8等

## TensorRT使用流程

预训练好的FP32的模型和网络，将模型通过Parser等方式输入到TensorRT中，TensorRT可以生成一个Serialization，也就是说将输入串流到内存或文件中，形成一个优化好的Engine，执行的时候可以调取他来执行推理。输出的文件被称为PLAN.

## TensorRT支持的两种输入

* Parser，即模型解析器。

* API, 可以添加一个Convolution或Pooling

  * 对于不支持的网络层

    使用Customer layer的功能。构建用户自定义曾需要告诉TensorRT该层的*连接关系*和*实现方式*。

    支持C++和Python.

    支持的Parser: Caffe, Uff(这个是NV定义的网络模型的一种文件结构， 支持Tensorflow), ONNX(Facebook主导的开源的可交换的各个框架都可以输出的).

## TensorRT所涉及的优化

首先是模型解析，之后Engine会进行优化，优化好的Engine可以序列化到内存或文件，读的时候需要反序列化，将其变成Engine以供使用。然后在执行的时候创建Context，主要是预先分配资源，Engien加Context就可以做推理了。

具体的优化方法：

* 合并网络层。

    TensorRT是存在Kernel调用的。也可以合并不同的卷积计算。

* Kernel可以不同大小的Batch Size和问题的复杂程度，选择最合适的算法，TensorRT预先写了很多GPU实现，有一个自动选择的过程。

* 不同的Batch Size会做Tuning.

* 支持多种设备：比如P4、V100以及嵌入式设备。

  * Convolution, Baise, ReLU会进行合并：CBR。

## TensorRT高级特性介绍

### 插件支持

* TensorRT支持插件。如添加自定义层

  * 需要冲在一个IPlugin的基类，生成自己的IPlugin的实现，告诉GPU或TensorRT需要做什么操作，要稿件的IPlugin是什么样子。

  * 要将插件放到合适的位置，在这里是添加到网络中。

* 低精度数值的支持

  * 支持FP16, INT8等：分别由P100, V100和P4, P40分别支持。

* Python接口和更多的框架支持

### 用户自定义层

使用用户自定义层主要分为两个步骤：

* 创建使用IPlugin接口创建用户自定义层

* 将创建的用户自定义层添加到网络中

IPlugin接口中需要被重载的函数有以下几类：

* 确定输出

  * *int getNbOutput()* 获取输出的数目

  * *Dims getOutputDimensions(int index, const Dims\* input, int nbInputDims)* 得到整个输出的维度信息。

* 层配置

  * *void configure()* 构建推理

  * 这个阶段确定的的恭喜是作为运行时作为插件参数来存储、序列化、反序列化的

* 资源管理

  * *void terminate()* 销毁资源。

  * *void initialize()* 初始化资源。

* 执行

  * *void enqueue()* 定义用户层的操作

* 序列化和反序列化

    这个过程是将层的参数写入二进制文件中。

  * *size_t getSerializationSize()* 获取序列大小

  * *void serialize()* 将层的参数序列化到缓存中

  * *PluginSample()* 从缓存中将层参数反序列化

* 从Caffe Parser添加Plugin

  * *Parsernvinfer1::IPlugin\* createPlugin()* 实现 *nvcaffeparse1::IPlugin* 接口，然后传递工厂实例到 *ICaffeParser::parse()*， Caffe的Parse才能识别。

* 运行时创建插件

  * 通过 *IPlugin\* createPlugin()* 实现 *nvinfer1::IPlugin* 接口，传递工厂实例到 *IInferRuntime::deserializeCudaEngine()*

## 一个具体的例子： YOLO-v2

省略。

## 总结TensorRT的优点

* TensorRT是一个高性能的深度学习推理的优化器和运行的引擎

* TensorRT支持Plugin, 对于不支持的层，可以通过Plugin来支持自定义的创建的层

* TensorRT支持低精度数值，来之加速