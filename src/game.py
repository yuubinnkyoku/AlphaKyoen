import numpy as np
import os
from numba import njit

BOARD_SIZE = 9
N_POINTS = 81

# プリ計算テーブルの読み込み
if not os.path.exists("patterns.npy"):
    raise FileNotFoundError("Run 'python src/precompute.py' first!")

# uint64として読み込むことを確実にする
PATTERNS_TABLE = np.load("patterns.npy").astype(np.uint64)

class Board:
    def __init__(self):
        self.board = np.zeros(N_POINTS, dtype=np.int8)
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
        # actionがnumpy.int64などの場合があるので、標準のintに変換
        action = int(action)
        
        if self.board[action] != 0:
            raise ValueError(f"Invalid move: {action}")
        
        self.board[action] = self.turn
        
        mask = 1 << action
        if self.turn == 1:
            self.bb_me |= mask
            current_bb = self.bb_me
        else:
            self.bb_opp |= mask
            current_bb = self.bb_opp
            
        self.move_count += 1
        
        # 相手の石と自分の石を合わせた全体の盤面で共円を判定する
        all_bb = self.bb_me | self.bb_opp
        
        # --- 修正ポイント：明示的に numpy.uint64 に変換する ---
        # 0xFFFFFFFFFFFFFFFF は 64bitの最大値
        bb_lo = np.uint64(all_bb & 0xFFFFFFFFFFFFFFFF)
        bb_hi = np.uint64(all_bb >> 64)
        
        # actionも明示的にintにキャストして渡す
        is_win = check_winner_numba(bb_lo, bb_hi, action, PATTERNS_TABLE)
        # --------------------------------------------------
        
        done = is_win or (self.move_count >= N_POINTS)
        reward = -1.0 if is_win else 0.0
        
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
    # last_moveに対応するパターン群を取り出す
    relevant_patterns = patterns_table[last_move]
    
    for i in range(relevant_patterns.shape[0]):
        pat_lo = relevant_patterns[i, 0]
        pat_hi = relevant_patterns[i, 1]
        
        # パディング（0, 0）に到達したら終了
        if pat_lo == np.uint64(0) and pat_hi == np.uint64(0):
            break
            
        # ビット演算でチェック
        if (bb_lo & pat_lo) == pat_lo:
            if (bb_hi & pat_hi) == pat_hi:
                return True
            
    return False