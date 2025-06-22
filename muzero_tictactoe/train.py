import draw
import numpy as np
import torch
import math
import settings
import rules
import network
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt



# state1 = np.zeros((settings.config['board_size'], settings.config['board_size']), dtype=np.int8)
# state2 = np.zeros((settings.config['board_size'], settings.config['board_size']), dtype=np.int8)
# state3 = np.zeros((settings.config['board_size'], settings.config['board_size']), dtype=np.int8)

# action_onehot1 = np.zeros((settings.config['board_size'], settings.config['board_size']))
# action_onehot2 = np.zeros((settings.config['board_size'], settings.config['board_size']))

representation_net = network.RepresentationNetwork(settings.config['board_size'] * settings.config['board_size'], 64).to(torch.device("cpu") )
prediction_net=network.PredictionNetwork().to(torch.device("cpu"))
dynamics_net = network.DynamicsNetwork().to(torch.device("cpu"))

# #通过棋盘状态state1获取隐状态hidden_state1
# hidden_state1 = representation_net(state1)
# #通过隐状态hidden_state1获取隐状态action_prob1，value1
# action_prob1, value1 = prediction_net(hidden_state1)
# #通过隐状态hidden_state1和动作概率action_prob1获取隐状态hidden_state2和奖励reward1
# hidden_state2, reward1 = dynamics_net(hidden_state1, action_onehot1)
# #通过隐状态hidden_state2获取隐状态action_prob2，value2
# action_prob2, value2 = prediction_net(hidden_state2)
# #通过隐状态hidden_state2和动作概率action_prob2获取隐状态hidden_state3和奖励reward2
# hidden_state3, reward2 = dynamics_net(hidden_state2, action_onehot2)
# #通过隐状态hidden_state3获取隐状态action_prob3，value3
# action_prob3, value3 = prediction_net(hidden_state3)

# #绘制图像
# draw.draw_all(value1,value2,value3,action_prob1,action_prob2,action_prob3,hidden_state1,hidden_state2,hidden_state3,action_onehot1,action_onehot2,state1,state2,state3)


def train_step(state, next_state, action, reward, optimizer):
    # 计算预测值
    hidden_state = representation_net(state)
    action_prob, value = prediction_net(hidden_state)
    next_hidden_state, predicted_reward = dynamics_net(hidden_state, action)
    
    # 调整 reward 的维度以匹配预测值
    reward = reward.view(-1, 1)
    
    # 计算损失
    value_loss = F.mse_loss(value, reward)
    reward_loss = F.mse_loss(predicted_reward, reward)
    
    total_loss = value_loss + reward_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

def select_action(action_prob):
    """根据动作概率选择动作"""
    # 将动作概率转换为numpy数组
    probs = action_prob.detach().numpy().reshape(-1)
    
    # 过滤掉非法动作（已经有棋子的位置）
    valid_moves = np.where(probs > 0)[0]
    
    if len(valid_moves) == 0:
        return None
    
    # 选择概率最高的动作
    action_index = valid_moves[np.argmax(probs[valid_moves])]
    
    # 转换为二维坐标
    row = action_index // settings.config['board_size']
    col = action_index % settings.config['board_size']
    
    # 创建one-hot动作向量
    action = np.zeros((settings.config['board_size'], settings.config['board_size']))
    action[row, col] = 1
    
    return action




def train_loop(num_episodes=100):
    # 初始化优化器
    optimizer = torch.optim.Adam([
        {'params': representation_net.parameters()},
        {'params': prediction_net.parameters()},
        {'params': dynamics_net.parameters()}
    ], lr=0.001)
    
    for episode in range(num_episodes):
        # 初始化状态
        state1 = np.zeros((settings.config['board_size'], settings.config['board_size']), dtype=np.int8)
        action_onehot1 = np.zeros((settings.config['board_size'], settings.config['board_size']))
        action_onehot2 = np.zeros((settings.config['board_size'], settings.config['board_size']))
        
        # 前向传播
        hidden_state1 = representation_net(state1)
        action_prob1, value1 = prediction_net(hidden_state1)
        hidden_state2, reward1 = dynamics_net(hidden_state1, action_onehot1)
        action_prob2, value2 = prediction_net(hidden_state2)
        hidden_state3, reward2 = dynamics_net(hidden_state2, action_onehot2)
        action_prob3, value3 = prediction_net(hidden_state3)
        
        # 计算损失并更新网络
        loss = train_step(state1, state1, action_onehot1, torch.tensor([0.0]), optimizer)
        
        # 每10轮显示一次图像
        if episode % 1 == 0:
                print(f"Episode {episode}, Loss: {loss}")
                
                # 清除所有子图的内容
                for ax in fig.axes:
                    ax.clear()
                
                # 重新绘制所有内容
                draw.draw_all(
                    value1, value2, value3,
                    action_prob1, action_prob2, action_prob3,
                    hidden_state1, hidden_state2, hidden_state3,
                    action_onehot1, action_onehot2,
                    state1, state1, state1,
                    fig=fig  # 传入固定的图形对象
                )
                
                # 更新显示
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(1)
        
        # 每25轮保存一次模型
        if episode % 25 == 0:
            save_models()

def save_models():
    # 确保 models 目录存在
    os.makedirs('models', exist_ok=True)
    
    torch.save(representation_net.state_dict(), 'models/representation_net.pth')
    torch.save(prediction_net.state_dict(), 'models/prediction_net.pth')
    torch.save(dynamics_net.state_dict(), 'models/dynamics_net.pth')

def load_models():
    representation_net.load_state_dict(torch.load('models/representation_net.pth'))
    prediction_net.load_state_dict(torch.load('models/prediction_net.pth'))
    dynamics_net.load_state_dict(torch.load('models/dynamics_net.pth'))
    
if __name__ == "__main__":

    # 初始化优化器
    optimizer = torch.optim.Adam([
        {'params': representation_net.parameters()},
        {'params': prediction_net.parameters()},
        {'params': dynamics_net.parameters()}
    ], lr=0.001)

    # 创建一个固定的图形和子图
    plt.ion()
    fig = plt.figure(figsize=(10, 8))

    # 创建所有需要的子图
    gs = fig.add_gridspec(5, 5)
    ax00 = fig.add_subplot(gs[0, 0], aspect='equal')
    ax01 = fig.add_subplot(gs[0, 1], aspect='equal')
    ax02 = fig.add_subplot(gs[0, 2], aspect='equal')
    ax03 = fig.add_subplot(gs[0, 3], aspect='equal')
    ax04 = fig.add_subplot(gs[0, 4], aspect='equal')
    ax10 = fig.add_subplot(gs[1, 0], aspect='equal')
    ax11 = fig.add_subplot(gs[1, 1], aspect='equal')
    ax12 = fig.add_subplot(gs[1, 2], aspect='equal')
    ax13 = fig.add_subplot(gs[1, 3], aspect='equal')
    ax14 = fig.add_subplot(gs[1, 4], aspect='equal')
    ax20 = fig.add_subplot(gs[2, 0], aspect='equal')
    ax21 = fig.add_subplot(gs[2, 1], aspect='equal')
    ax22 = fig.add_subplot(gs[2, 2], aspect='equal')
    ax23 = fig.add_subplot(gs[2, 3], aspect='equal')
    ax24 = fig.add_subplot(gs[2, 4], aspect='equal')
    ax30 = fig.add_subplot(gs[3, 0], aspect='equal')
    ax31 = fig.add_subplot(gs[3, 1], aspect='equal')
    ax32 = fig.add_subplot(gs[3, 2], aspect='equal')
    ax33 = fig.add_subplot(gs[3, 3], aspect='equal')
    ax34 = fig.add_subplot(gs[3, 4], aspect='equal')
    ax40 = fig.add_subplot(gs[4, 0], aspect='equal')
    ax41 = fig.add_subplot(gs[4, 1], aspect='equal')
    ax42 = fig.add_subplot(gs[4, 2], aspect='equal')
    ax43 = fig.add_subplot(gs[4, 3], aspect='equal')
    ax44 = fig.add_subplot(gs[4, 4], aspect='equal')

    # 强制更新布局
    plt.tight_layout()
    
    for episode in range(100):
        # 初始化状态
        state1 = np.zeros((settings.config['board_size'], settings.config['board_size']), dtype=np.int8)
        action_onehot1 = np.zeros((settings.config['board_size'], settings.config['board_size']))
        action_onehot2 = np.zeros((settings.config['board_size'], settings.config['board_size']))
        
        # 前向传播
        hidden_state1 = representation_net(state1)
        action_prob1, value1 = prediction_net(hidden_state1)
        hidden_state2, reward1 = dynamics_net(hidden_state1, action_onehot1)
        action_prob2, value2 = prediction_net(hidden_state2)
        hidden_state3, reward2 = dynamics_net(hidden_state2, action_onehot2)
        action_prob3, value3 = prediction_net(hidden_state3)
        
        # 计算损失并更新网络
        loss = train_step(state1, state1, action_onehot1, torch.tensor([0.0]), optimizer)
        
        # 每10轮显示一次图像
        if episode % 1 == 0:
                print(f"Episode {episode}, Loss: {loss}")
                                
                # 重新绘制所有内容
                draw.draw_all(
                    value1, value2, value3,
                    action_prob1, action_prob2, action_prob3,
                    hidden_state1, hidden_state2, hidden_state3,
                    action_onehot1, action_onehot2,
                    state1, state1, state1,
                    fig=fig  # 传入固定的图形对象
                )
                
                # 更新显示
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(1)
        
        # 每25轮保存一次模型
        if episode % 25 == 0:
            save_models()
           
    plt.ioff()
    save_models()
    plt.close(fig)  # 关闭图形

