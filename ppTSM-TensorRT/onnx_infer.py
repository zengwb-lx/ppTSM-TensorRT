import argparse
import os, shutil
import sys
from os import path as osp
import onnx
import onnxruntime as ort
import numpy as np
from typing import List
import paddle.nn.functional as F
from onnx_utils import *

def modify_onnx():
    onnx_path = './ppTSM-sim.onnx'
    onnx_model = onnx.load(onnx_path) #加载onnx模型
    graph = onnx_model.graph
    old_nodes = graph.node
    new_nodes = old_nodes[1:] #去掉data,sub,mul前三个节点
    del onnx_model.graph.node[:] # 删除当前onnx模型的所有node
    onnx_model.graph.node.extend(new_nodes) # extend新的节点
    conv0_node = onnx_model.graph.node[0]
    conv0_node.input[0] = 'data_batch_0' #给第一层的卷积节点设置输入的data节点
    # onnx_model.graph.input[0].type.tensor_type.shape.dim = 4
    
    onnx_model.graph.input[0].type.tensor_type.shape.dim.pop()
    print(onnx_model.graph.input[0].type.tensor_type.shape.dim)
    # onnx_model.graph.input[0].type.tensor_type.shape.dim = ['dim_value': 16, 'dim_value': 3, 'dim_value': 512, 'dim_value': 682]
    # onnx_model.graph.input[0].type.tensor_type.shape.dim = [16, 3,512, 682]
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 16
    onnx_model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 3
    onnx_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 512
    onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 682
    # graph = onnx_model.graph
    # print(graph.node)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, "./ppTSM-sim-m.onnx")

def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser("PaddleVideo Inference model script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/recognition/pptsm/v2/pptsm_lcnet_k400_16frames_uniform.yaml',
                        help='config file path')
    parser.add_argument("-i", "--input_file", type=str, default='data/example.avi', help="input file path")
    parser.add_argument("--onnx_file", type=str, default='weights/ppTSM-sim.onnx', help="onnx model file path")

    # params for onnx predict
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_gpu",
                        type=str2bool,
                        default=False,
                        help="set to False when using onnx")
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--enable_benchmark",
                        type=str2bool,
                        default=False,
                        help="set to False when using onnx")
    parser.add_argument("--cpu_threads", type=int, default=4)

    return parser.parse_args()

def create_onnx_predictor(args, cfg=None):
    onnx_file = args.onnx_file
    config = ort.SessionOptions()
    if args.use_gpu:
        raise ValueError(
            "onnx inference now only supports cpu! please set `use_gpu` to False."
        )
    else:
        config.intra_op_num_threads = args.cpu_threads
        if args.ir_optim:
            config.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    predictor = ort.InferenceSession(onnx_file, sess_options=config)
    return config, predictor


def parse_file_paths(input_path: str) -> list:
    if osp.isfile(input_path):
        files = [
            input_path,
        ]
    else:
        files = os.listdir(input_path)
        files = [
            file for file in files
            if (file.endswith(".avi") or file.endswith(".mp4"))
        ]
        files = [osp.join(input_path, file) for file in files]
    return files

class ppTSM_Inference_helper():
    def __init__(self,
                 num_seg=16,
                 seg_len=1,
                 short_size=500,
                 target_size=448,
                 top_k=1):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.short_size = short_size
        self.target_size = target_size
        self.top_k = top_k

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        results = {'filename': input_file}
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        ops = [
            VideoDecoder(backend="decord"),
            Sampler(self.num_seg, self.seg_len, valid_mode=True),
            Scale(self.short_size, backend='cv2'),
            # CenterCrop(self.target_size),
            Image2Array(),
            Normalization(img_mean, img_std)
        ]
        for op in ops:
            results = op(results)

        res = np.expand_dims(results['imgs'], axis=0).copy()
        return [res]

    def preprocess_batch(self, file_list: List[str]) -> List[np.ndarray]:
        """preprocess for file list

        Args:
            file_list (List[str]): file pathes in an list, [path1, path2, ...].

        Returns:
            List[np.ndarray]: batched inputs data, [data_batch[0], data_batch[1], ...].
        """
        batched_inputs = []
        for file in file_list:
            inputs = self.preprocess(file)
            batched_inputs.append(inputs)
        batched_inputs = [
            np.concatenate([item[i] for item in batched_inputs])
            for i in range(len(batched_inputs[0]))
        ]
        self.input_file = file_list
        # print('batched_inputs[0].shape: ', batched_inputs[0].shape)
        # in1 = np.full((1, 16, 3, 512, 682), 1.1,dtype=np.float32)
        # batched_inputs[0] = in1
        # with open('/mnt/data/DcVideoData/train/videos/wuchibie1.txt', 'w') as f:
        #     for i in range(batched_inputs[0].shape[1]):
        #         for j in range(batched_inputs[0].shape[2]):
        #             for k in range(batched_inputs[0].shape[3]):
        #                 for h in range(batched_inputs[0].shape[4]):
        #                     print("pixel: ", batched_inputs[0][0][i][j][k][h])
        #                     f.write(str(batched_inputs[0][0][i][j][k][h]) + '\n')
        return batched_inputs

    def postprocess(self,
                    output: np.ndarray,
                    print_output: bool = True,
                    return_result: bool = False):
        """postprocess

        Args:
            output (np.ndarray): batched output scores, shape of (batch_size, class_num).
            print_output (bool, optional): whether to print result. Defaults to True.
        """
        if not isinstance(self.input_file, list):
            self.input_file = [
                self.input_file,
            ]
        output = output[0]  # [B, num_cls]
        N = len(self.input_file)
        if output.shape[0] != N:
            output = output.reshape([N] + [output.shape[0] // N] +
                                    list(output.shape[1:]))  # [N, T, C]
            output = output.mean(axis=1)  # [N, C]
        output = F.softmax(paddle.to_tensor(output), axis=-1).numpy()
        results_list = []
        for i in range(N):
            classes = np.argpartition(output[i], -self.top_k)[-self.top_k:]
            classes = classes[np.argsort(-output[i, classes])]
            scores = output[i, classes]
            topk_class = classes[:self.top_k]
            topk_scores = scores[:self.top_k]
            result = {
                "video_id": self.input_file[i],
                "topk_class": topk_class,
                "topk_scores": topk_scores
            }
            results_list.append(result)
            if print_output:
                print("Current video file: {0}".format(self.input_file[i]))
                print("\ttop-{0} class: {1}".format(self.top_k, topk_class))
                print("\ttop-{0} score: {1}".format(self.top_k, topk_scores))
            # if topk_class[0] == 1:
            #     print("Current video file: {0}".format(self.input_file[i]))
            #     print("\ttop-{0} class: {1}".format(self.top_k, topk_class))
            #     print("\ttop-{0} score: {1}".format(self.top_k, topk_scores))
            #     shutil.move(str(self.input_file[i]), '/mnt/data/DcVideoData/train/videos/wuchibie0/')
        if return_result:
            return results_list

def main():
    args = parse_args()
    # cfg = get_config(args.config, show=False)

    model_name = "ppTSM"

    print(f"Inference model({model_name})...")
    InferenceHelper = ppTSM_Inference_helper(num_seg=16)  # build_inference_helper(cfg.INFERENCE)
    inference_config, predictor = create_onnx_predictor(args)
    # get input_tensor and output_tensor
    input_names = predictor.get_inputs()[0].name
    output_names = predictor.get_outputs()[0].name
    # print(input_names, output_names)

    # get the absolute file path(s) to be processed
    files = parse_file_paths(args.input_file)
    # Inferencing process
    batch_num = args.batch_size
    for st_idx in range(0, len(files), batch_num):
        ed_idx = min(st_idx + batch_num, len(files))

        # Pre process batched input
        batched_inputs = InferenceHelper.preprocess_batch(files[st_idx:ed_idx])
        # print("batched_inputs: ", batched_inputs[0].shape)
        # print(len(batched_inputs), batched_inputs[0].shape)

        batched_outputs = predictor.run(
            output_names=[output_names],
            input_feed={input_names: batched_inputs[0]})
        # print("batched_outputs: ", batched_outputs)
        InferenceHelper.postprocess(batched_outputs, not args.enable_benchmark)



if __name__ == "__main__":
    main()
    # modify_onnx()