# 本文件负责：根据配置初始化运行环境，构建并加载 EdgeConnect 模型，
# 在训练（MODE=1）、测试（MODE=2）与评估（MODE=3）三种模式下运行。
# 通过设定 CUDA 显卡、随机种子与 OpenCV 线程，保证可复现与稳定性。
import os                    # 操作系统路径与环境变量
import cv2                   # OpenCV，用于图像处理与多线程配置
import random                # Python 内置随机数库
import numpy as np           # 数值计算库
import torch                 # PyTorch 深度学习框架
import argparse              # 命令行参数解析
from shutil import copyfile  # 文件复制，用于生成默认配置
from src.config import Config            # 配置读取与封装
from src.edge_connect import EdgeConnect # EdgeConnect 模型主体


def main(mode=None):
    r"""starts the model
    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    # 读取配置；若未传入 mode，则由命令行参数与配置文件决定运行模式
    config = load_config(mode)


    # 指定可见的 CUDA 显卡编号，避免占用全部显卡
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # 初始化运行设备；若支持 CUDA 则使用 GPU，并开启 cuDNN 自动调优
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # 启用 cuDNN 自动调优以提升卷积性能
    else:
        config.DEVICE = torch.device("cpu")

    # 设置 OpenCV 运行线程为 1，避免与 PyTorch DataLoader 竞争导致死锁
    cv2.setNumThreads(0)

    # 固定随机种子，确保实验可复现（PyTorch / NumPy / Python）
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.autograd.set_detect_anomaly(True)

    # 构建模型并加载权重/状态（如存在）
    model = EdgeConnect(config)
    model.load()

    # 训练模式
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        model.train()

    # 测试模式
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    # 评估模式
    else:
        print('\nstart eval...\n')
        model.eval()


def load_config(mode=None):
    r"""loads model config
    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    # 构建命令行解析器：支持通用参数与测试专用参数
    parser = argparse.ArgumentParser()
    # 检查点目录；默认 ./checkpoints，可用 --path 或 --checkpoints 指定
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    # 模型类型：1 边缘模型，2 修复模型，3 边缘+修复组合，4 联合模型
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')

    # 若处于测试模式，允许传入输入图像/掩码/边缘/输出目录
    if mode == 2:
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
        parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
        parser.add_argument('--output', type=str, help='path to the output directory')

    # 解析参数后拼接配置文件路径
    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # 若检查点目录不存在则创建
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # 若配置文件不存在则从模板复制一份
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # 加载配置文件为 Config 对象
    config = Config(config_path)

    # 训练模式：强制 MODE=1；允许通过 --model 指定子模型
    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    # 测试模式：MODE=2；默认 MODEL=3（边缘-修复组合）；禁用固定输入尺寸
    elif mode == 2:
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else 3
        config.INPUT_SIZE = 0

        # 覆盖测试文件列表（输入图像路径或目录）
        if args.input is not None:
            config.TEST_FLIST = args.input

        # 覆盖掩码文件列表（路径或目录）
        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        # 覆盖边缘文件列表（路径或目录）
        if args.edge is not None:
            config.TEST_EDGE_FLIST = args.edge

        # 指定输出目录
        if args.output is not None:
            config.RESULTS = args.output

    # 评估模式：MODE=3；默认 MODEL=3（边缘-修复组合）
    elif mode == 3:
        config.MODE = 3
        config.MODEL = args.model if args.model is not None else 3

    return config


if __name__ == "__main__":
    # 作为脚本执行时，调用主入口
    main()
