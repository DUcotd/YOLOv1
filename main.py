import argparse


def get_args():
    parser = argparse.ArgumentParser(description="加载数据与训练模型")
    parser.add_argument("--voc_dir", type=str, default="./voc", help="VOC数据集目录")
    parser.add_argument("--batch_size", type=int, default=16, help="批大小")
    parser.add_argument("--load_checkpoint", default="istrue", help="是否加载预训练模型")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数")
    parser.add_argument("--pin_memory", type=bool, default=True, help="是否使用固定内存")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    