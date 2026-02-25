import matplotlib.pyplot as plt
import os
import numpy as np

class LossVisualizer:
    """用于可视化和保存训练/验证损失的类"""
    
    def __init__(self, save_dir):
        """
        初始化损失可视化器
        Args:
            save_dir: 保存图表的目录路径
        """
        self.save_dir = save_dir
        self.train_losses = []  # 存储训练损失
        self.val_losses = []    # 存储验证损失
        self.epochs = []        # 存储轮次
        
        # 如果图表目录不存在则创建
        self.plot_dir = os.path.join(save_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)

        # 初始化损失记录文件
        self.loss_file_path = os.path.join(self.plot_dir, 'loss.txt')
        with open(self.loss_file_path, 'w') as f:
            f.write("Epoch\tTrain_Loss\tVal_Loss\n")
        
    def update(self, epoch, train_loss, val_loss):
        """
        更新损失历史记录
        Args:
            epoch: 当前轮次
            train_loss: 训练损失值
            val_loss: 验证损失值(可选)
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

        # 同步更新到loss.txt文件
        with open(self.loss_file_path, 'a') as f:
            if val_loss is not None:
                f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")
            else:
                f.write(f"{epoch}\t{train_loss:.6f}\tN/A\n")
            
    def plot_losses(self):
        """
        创建并保存损失曲线图
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.epochs, self.train_losses, label='train_loss', color='blue')
        if self.val_losses:
            plt.plot(self.epochs, self.val_losses, label='val_loss', color='red')
        
        plt.xlabel('epoch')
        plt.ylabel('loss value')
        plt.title('loss curves')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plot_path = os.path.join(self.plot_dir, 'loss_plot.png')
        plt.savefig(plot_path)
        plt.close()
        
    def save_loss_history(self):
        """
        将损失历史记录保存为numpy文件
        """
        history = {
            'epochs': np.array(self.epochs),
            'train_losses': np.array(self.train_losses),
            'val_losses': np.array(self.val_losses)
        }
        history_path = os.path.join(self.plot_dir, 'loss_history.npz')
        np.savez(history_path, **history)