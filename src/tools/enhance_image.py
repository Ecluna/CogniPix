import argparse
import os
import sys
from pathlib import Path
import torch

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parents[2]  # 获取项目根目录
sys.path.append(str(ROOT_DIR))

from src.models.super_resolution import ImageEnhancer

def main():
    parser = argparse.ArgumentParser(description='图片清晰度提升工具')
    parser.add_argument('input', help='输入图片路径或目录')
    parser.add_argument('--output', '-o', help='输出路径（可选）')
    parser.add_argument('--device', choices=['cuda', 'cpu'], 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用的设备（可选，默认使用GPU如果可用）')
    
    args = parser.parse_args()
    
    try:
        # 初始化增强器
        enhancer = ImageEnhancer(device=args.device)
        
        # 判断输入是文件还是目录
        if os.path.isfile(args.input):
            # 处理单个文件
            output_path = args.output if args.output else None
            result_path = enhancer.enhance_image(args.input, output_path)
            print(f'处理完成，结果保存在: {result_path}')
        
        elif os.path.isdir(args.input):
            # 处理目录
            output_dir = args.output if args.output else os.path.join(args.input, 'enhanced')
            enhancer.enhance_directory(args.input, output_dir)
            print(f'处理完成，结果保存在: {output_dir}')
        
        else:
            print('错误：输入路径不存在')
            
    except KeyboardInterrupt:
        print("\n处理已取消")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 