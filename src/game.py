import numpy as np
import os
from numba import njit

BOARD_SIZE = 9
N_POINTS = 81

# プリ計算テーブルの読み込み
# ファイルがない場合はエラーを出す
if not os.path.exists("patterns.npy"):
    raise FileNotFoundError("Run 'python precompute.py' first!")

PATTERNS_TABLE = np.load("patterns.npy") # Shape: (81, MAX_PATTERNS, 2)

class Board:
    def __init__(self):
        # 0: 空き, 1: 自分(黒), -1: 相手(白)
        self.board = np.zeros(N_POINTS, dtype=np.int8)
        
        # 高速判定用のBitboard (自分, 相手)
        # Numbaに渡すため、[low_64, high_64] の形(長さ2の配列)で管理してもいいが、
        # ここではPython管理用に巨大整数を持ち、渡すときに分割する
        self.bb_me = 0 
        self.bb_opp = 0
        
        self.turn = 1
        self.move_count = 0

    def copy(self):
        b = Board()
        b.board = self.board.copy()
        b.bb_me = self.bb_me
        b.bb_opp = self.bb_opp
        b.turn = self.turn
        b.move_count = self.move_count
        return b

    def valid_moves(self):
        return np.where(self.board == 0)[0]

    def step(self, action):
        if self.board[action] != 0:
            raise ValueError(f"Invalid move: {action}")
        
        self.board[action] = self.turn
        
        # Bitboard更新
        mask = 1 << action
        if self.turn == 1:
            self.bb_me |= mask
            current_bb = self.bb_me
        else:
            self.bb_opp |= mask
            current_bb = self.bb_opp
            
        self.move_count += 1
        
        # 勝敗判定 (高速化)
        # Pythonの巨大整数を 64bit x 2 に分割してNumbaへ
        bb_lo = current_bb & 0xFFFFFFFFFFFFFFFF
        bb_hi = current_bb >> 64
        
        is_win = check_winner_numba(bb_lo, bb_hi, action, PATTERNS_TABLE)
        
        done = is_win or (self.move_count >= N_POINTS)
        reward = 1.0 if is_win else 0.0
        
        self.turn *= -1
        return self.board, reward, done

    def get_feature(self):
        feat = np.zeros((3, 9, 9), dtype=np.float32)
        me = (self.board == self.turn).reshape(9, 9)
        opp = (self.board == -self.turn).reshape(9, 9)
        feat[0] = me
        feat[1] = opp
        feat[2] = 1.0 if self.turn == 1 else 0.0
        return feat

@njit(fastmath=True)
def check_winner_numba(bb_lo, bb_hi, last_move, patterns_table):
    """
    bb_lo: Bitboardの下位64bit (uint64)
    bb_hi: Bitboardの上位64bit (uint64)
    last_move: 最後に打った位置 (int)
    patterns_table: プリ計算テーブル (uint64 array)
    """
    # この点に関係するパターンだけを取得
    relevant_patterns = patterns_table[last_move]
    
    # ループ (numbaが最適化してくれる)
    for i in range(len(relevant_patterns)):
        pat_lo = relevant_patterns[i][0]
        pat_hi = relevant_patterns[i][1]
        
        # パディング(0,0)に到達したら終了
        if pat_lo == 0 and pat_hi == 0:
            break
            
        # 包含判定: (Board & Pattern) == Pattern
        # 上位・下位それぞれでチェック
        if (bb_lo & pat_lo) == pat_lo:
            if (bb_hi & pat_hi) == pat_hi:
                return True
            
    return False