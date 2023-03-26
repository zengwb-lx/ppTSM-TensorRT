# 视频分类算法ppTSM使用记录

[TOC]
## 简介
- 视频分类与图像分类相似，均属于识别任务，对于给定的输入视频，视频分类模型需要输出其预测的标签类别。如果标签都是行为类别，则该任务也常被称为行为识别。与图像分类不同的是，视频分类往往需要利用多帧图像之间的时序信息。PP-TSM是PaddleVideo自研的实用产业级视频分类模型，在实现前沿算法的基础上，考虑精度和速度的平衡，进行模型瘦身和精度优化，使其可能满足产业落地需求。
## 一. 资源准备
- TSM论文：https://arxiv.org/pdf/1811.08383.pdf
- TSM作者github：https://github.com/mit-han-lab/temporal-shift-module
- mmaction2实现：https://github.com/open-mmlab/mmaction2
- [mmaction2版本] 视频分类(一) TSM：Temporal Shift Module for Efficient Video - Understanding 原理及代码讲解：https://www.jianshu.com/p/22317230210d
- PaddleVideo实现ppTSM：https://github.com/PaddlePaddle/PaddleVideo
- pp飞浆模型库的训练流程：https://www.paddlepaddle.org.cn/modelbasedetail/tsm
- pp飞浆项目库的讲解：https://aistudio.baidu.com/aistudio/projectdetail/3628596
- CSDN讲解：https://blog.csdn.net/u012193416/article/details/127977221

**这里本人主要使用了PaddleVideo的ppTSM进行开发**

## 二. 模型训练
1. 数据集准备:
按照ucf101格式做好自己的数据就行了,ucf101下载链接: https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-tsm.md
2. 训练命令行:
`python -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_pptsm main.py --validate -c ./configs/recognition/pptsm/v2/pptsm_lcnet_ucf101_16frames_uniform.yaml --amp`
3. 模型导出:
`python tools/export_model.py -c configs/recognition/pptsm/v2/pptsm_lcnet_ucf101_16frames_uniform.yaml -p output/ppTSMv2/ppTSMv2_best.pdparams -o inference/PPTSMv2`
4. 模型预测,验证模型准确率: 
`python tools/predict.py --input_file data/example.avi --config configs/recognition/pptsm/v2/pptsm_lcnet_k400_16frames_uniform.yaml --model_file weights/ppTSMv2.pdmodel --params_file weights/ppTSMv2.pdiparams --use_gpu=True`
5. paddle2onnx:
`paddle2onnx --model_dir=./weights/ucf101 --model_filename=ppTSMv2.pdmodel --params_filename=ppTSMv2.pdiparams --save_file=./weights/ucf101/ppTSMv2/pptsmv2.onnx --opset_version=11 --enable_onnx_checker=True`

- 以上5个步骤的参考链接: https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-tsm.md

## 三. 算法部署
### 方案一 paddle_inference:
这个paddle_inference后端支持tensorrt和onnx, CPU端和GPU端
https://github.com/PaddlePaddle/PaddleVideo/tree/develop/deploy/cpp_infer
### 方案二 tensorrt(本文重点)
**本人选用的是tensorrt,因为其他算法是用tensorrt部署的,如果再加一个paddle_inference会显得太过于杂乱了,详细实现见本人GitHub:**

## 四. 精度对齐
为了保证模型转换后精度无损,需要进行paddle, onnx, tensorrt三方推理结果比对:详细可比对predict.py, onnx_infer.py, C++输出结果.


## 五. 总结
之前一直用的是传统的图像处理算法(opencv的高斯混合模型,帧间差法)做视频分类识别,改用深度学习的方法后准确率的通用性提高了很多,但是需要更多的训练数据来增加算法的鲁棒性.