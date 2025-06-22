import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 配置 Matplotlib 字体以支持中文
rcParams['font.sans-serif'] = ['SimHei']  # 使用系统中的黑体字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_board(board, title="棋盘状态", ax=None):
    size = board.shape[0]
    if ax is None:
        fig, ax = plt.subplots(figsize=(size + 1, size + 1))
    ax.set_title(title)
    ax.set_xticks(np.arange(size + 1))
    ax.set_yticks(np.arange(size + 1))
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.grid(color="black", linestyle="-", linewidth=1)
    ax.invert_yaxis()

    # 在格子中标记 X 和 O
    for i in range(size):
        for j in range(size):
            if board[i, j] == 1:
                ax.text(j + 0.5, i + 0.5, "X", ha="center", va="center", fontsize=48 / size, color="black", fontweight="bold")
            elif board[i, j] == 2:
                ax.text(j + 0.5, i + 0.5, "O", ha="center", va="center", fontsize=48 / size, color="black", fontweight="bold")
    
    if ax is None:
        plt.tight_layout()
        plt.show()
    return ax

def plot_action(board, title="执行动作", ax=None):
    size = board.shape[0]
    if ax is None:
        fig, ax = plt.subplots(figsize=(size + 1, size + 1))
    ax.set_title(title)
    ax.set_xticks(np.arange(size + 1))
    ax.set_yticks(np.arange(size + 1))
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.grid(color="black", linestyle="-", linewidth=1)
    ax.invert_yaxis()

    # 在格子中标记 X 和 O
    for i in range(size):
        for j in range(size):
            if board[i, j] == 1:
                ax.text(j + 0.5, i + 0.5, "X", ha="center", va="center", fontsize=48 / size, color="black", fontweight="bold")
            elif board[i, j] == 2:
                ax.text(j + 0.5, i + 0.5, "O", ha="center", va="center", fontsize=48 / size, color="black", fontweight="bold")
    
    if ax is None:
        plt.tight_layout()
        plt.show()
    return ax

def hidden_state(board, title="隐状态", ax=None):
    # 获取输入数据的维度并转换为numpy数组
    if hasattr(board, 'detach'):
        board_array = board.detach().cpu().numpy()
    else:
        board_array = board
    
    # 处理 batch 维度
    if len(board_array.shape) > 1:
        board_array = board_array[0]
        
    # 确保是一维数组并重塑为正方形
    board_array = board_array.flatten()
    size = int(np.sqrt(len(board_array)))
    board_array = board_array.reshape(size, size)
    
    # 使用现有的 ax 绘制热力图
    im = ax.imshow(board_array, cmap='viridis')
    
    # 设置标题和坐标轴
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax

def action_prob(board, title="动作概率", ax=None):
    # 获取输入数据的维度并转换为numpy数组
    if hasattr(board, 'detach'):
        board_array = board.detach().cpu().numpy()
    else:
        board_array = board

    # 处理一维数据
    if len(board_array.shape) == 1:
        length = board_array.shape[0]
        size = int(np.sqrt(length))  # 计算合适的边长
        if size * size != length:
            raise ValueError(f"输入长度 {length} 无法转换为完美的正方形")
        board_array = board_array.reshape(size, size)  # 重塑为二维正方形
    
    size = board_array.shape[0]
    if ax is None:
        fig, ax = plt.subplots(figsize=(size + 1, size + 1))
    
    im = ax.imshow(board_array, cmap='viridis')
    
    ax.set_title(title)
    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    if ax is None:
        plt.tight_layout()
        plt.show()
    return ax

def value(value_tensor, title="价值", ax=None):
    # 获取输入数据并转换为标量
    if hasattr(value_tensor, 'detach'):
        value_num = value_tensor.detach().cpu().numpy().item()
    else:
        value_num = float(value_tensor)

    if ax is None:
        fig, ax = plt.subplots(figsize=(2, 2))
    
    # 清除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # 设置标题
    ax.set_title(title)
    
    # 在中心显示数值
    ax.text(0.5, 0.5, f'{value_num:.3f}', 
            ha='center', 
            va='center',
            fontsize=12,
            transform=ax.transAxes)
    
    if ax is None:
        plt.tight_layout()
        plt.show()
    return ax


def draw_all(v1, v2, v3,
             p1, p2, p3,
             h1, h2, h3,
             a1, a2,
             s1, s2, s3,
             fig=None):
    """在固定的画布上更新所有图像
    
    Args:
        v1, v2, v3: 价值
        p1, p2, p3: 策略
        h1, h2, h3: 隐状态
        a1, a2: 动作
        s1, s2, s3: 状态
        fig: matplotlib图形对象
    """
    if fig is None:
        return
    
    axes = fig.axes
    if not axes:
        return
    
    # 清除所有子图内容
    for ax in axes:
        ax.clear()
    
    # 更新价值
    value(v1, title="v1", ax=axes[0])
    value(v2, title="v2", ax=axes[2])
    value(v3, title="v3", ax=axes[4])
    
    # 更新策略
    plot_action(p1, title="p1", ax=axes[5])
    plot_action(p2, title="p2", ax=axes[7])
    plot_action(p3, title="p3", ax=axes[9])
    
    # 更新隐状态
    hidden_state(h1, title="h1", ax=axes[10])
    hidden_state(h2, title="h2", ax=axes[12])
    hidden_state(h3, title="h3", ax=axes[14])
    
    # 更新动作
    plot_action(a1, title="a1", ax=axes[16])
    plot_action(a2, title="a2", ax=axes[18])
    
    # 更新状态
    plot_board(s1, title="s1", ax=axes[20])
    plot_board(s2, title="s2", ax=axes[22])
    plot_board(s3, title="s3", ax=axes[24])