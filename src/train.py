import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import os

from models.watermark_remover import WatermarkRemover
from data.dataset import WatermarkDataset

def train(config_path='configs/config.yaml'):
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(config['training']['device'])
    
    # 创建数据加载器
    train_dataset = WatermarkDataset(config['data']['train_path'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    # 初始化模型
    model = WatermarkRemover().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # 训练循环
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}') as pbar:
            for watermarked, clean in pbar:
                watermarked = watermarked.to(device)
                clean = clean.to(device)
                
                optimizer.zero_grad()
                output = model(watermarked)
                loss = criterion(output, clean)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = 'checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train() 