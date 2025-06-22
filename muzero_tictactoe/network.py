import torch
import torch.nn as nn
import torch.nn.functional as F

class RepresentationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
    
    def forward(self, board):
        # 将输入转换为 PyTorch tensor
        if not isinstance(board, torch.Tensor):
            x = torch.FloatTensor(board)
        else:
            x = board
            
        # 添加 batch 维度（如果需要）
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        # 添加 channel 维度（如果需要）
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        # 重塑张量
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        return self.fc(x)
    
class PredictionNetwork(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, output_size=9):
        super().__init__()
        # 动作预测网络
        self.action_fc1 = nn.Linear(input_size, hidden_size)
        self.action_fc2 = nn.Linear(hidden_size, output_size)
        
        # 价值预测网络
        self.value_fc1 = nn.Linear(input_size, hidden_size)
        self.value_fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_state):
        # 动作预测
        action = F.relu(self.action_fc1(hidden_state))
        action_logits = self.action_fc2(action)
        action_prob = F.softmax(action_logits, dim=1)
        
        # 价值预测
        value = F.relu(self.value_fc1(hidden_state))
        value = self.value_fc2(value)
        
        return action_prob, value


class DynamicsNetwork(nn.Module):
    def __init__(self, input_size=73, hidden_size=64, output_size=64):
        super().__init__()
        self.state_fc1 = nn.Linear(input_size, hidden_size)
        self.state_fc2 = nn.Linear(hidden_size, output_size)
        
        self.reward_fc1 = nn.Linear(input_size, hidden_size)
        self.reward_fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_state, action):
        # 将输入转换为 PyTorch tensor
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action)
        
        # 添加 batch 维度（如果需要）
        if len(action.shape) == 2:
            action = action.unsqueeze(0)
        
        # 展平 action
        batch_size = hidden_state.size(0)
        action = action.view(batch_size, -1)
        
        # 确保 hidden_state 维度正确
        hidden_state = hidden_state.view(batch_size, -1)
        
        # 将隐状态和动作连接
        x = torch.cat([hidden_state, action], dim=1)
        
        # 预测下一个状态
        next_state = F.relu(self.state_fc1(x))
        next_state = self.state_fc2(next_state)
        
        # 预测奖励
        reward = F.relu(self.reward_fc1(x))
        reward = self.reward_fc2(reward)
        
        return next_state, reward
    
    