import numpy as np

# =========================
# 通用连珠检测函数
# =========================
def check_winner(board: np.ndarray, n: int) -> int:
    """
    检查棋盘上是否有连续n个相同棋子的玩家获胜。
    board: 2D numpy数组，1代表player1，2代表player2，0代表空
    n: 连珠数
    返回：1（player1胜），2（player2胜），0（无人获胜）
    """
    H, W = board.shape
    directions = [(0,1), (1,0), (1,1), (1,-1)]  # 右, 下, 右下, 左下

    for r in range(H):
        for c in range(W):
            player = board[r, c]
            if player == 0:
                continue
            for dr, dc in directions:
                count = 1
                nr, nc = r + dr, c + dc
                while 0 <= nr < H and 0 <= nc < W and board[nr, nc] == player:
                    count += 1
                    if count == n:
                        return int(player)
                    nr += dr
                    nc += dc
    return 0