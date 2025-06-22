# 配置参数
config = {
    # 游戏环境
    'board_size': 3,
    'num_to_win': 3, #三子连珠

    # 神经网络
    'hidden_dim': 64,
    
    # MCTS
    'num_simulations': 50, # 模拟次数

    # 训练
    'num_games': 500, # 自对弈游戏数量
    'batch_size_train': 128, # 训练批次大小
    'train_epochs_per_game': 1, # 每个游戏训练多少个 epoch
    'num_unroll_steps': 3, # MuZero 训练时展开步数
    'replay_buffer_capacity': 900, # 经验回放缓冲区容量
    'learning_rate': 1e-4,
}