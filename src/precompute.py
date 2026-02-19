import numpy as np
import itertools
from tqdm import tqdm
import os

# 設定
BOARD_SIZE = 9
OUTPUT_FILE = "patterns.npy"

def get_coords(idx):
    """インデックス(0-80)を(x, y)座標に変換"""
    return idx % BOARD_SIZE, idx // BOARD_SIZE

def det_4x4(mat):
    """
    4x4行列の行列式を計算する（整数演算用）
    NumPyのdetは浮動小数点になるため、あえて自作して厳密性を保つ
    """
    # 余因子展開で計算 (再帰なしのハードコードで高速化)
    m = mat
    
    # 0行目で展開
    # 3x3行列式の手計算公式: a(ei−fh)−b(di−fg)+c(dh−eg)
    def d3(a, b, c, d, e, f, g, h, i):
        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

    return (
        m[0][0] * d3(m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], m[3][3]) -
        m[0][1] * d3(m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], m[3][3]) +
        m[0][2] * d3(m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], m[3][3]) -
        m[0][3] * d3(m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], m[3][2])
    )

def main():
    print(f"Generating Kyoen patterns for {BOARD_SIZE}x{BOARD_SIZE}...")

    # 1. 各点のパラメータ [x^2+y^2, x, y, 1] を事前計算
    points_params = []
    for i in range(BOARD_SIZE * BOARD_SIZE):
        x, y = get_coords(i)
        points_params.append([x**2 + y**2, x, y, 1])

    valid_patterns = []
    
    # 2. 全ての4点の組み合わせを探索 (81C4 = 1,663,740通り)
    # tqdmで進捗を表示
    all_combinations = itertools.combinations(range(BOARD_SIZE * BOARD_SIZE), 4)
    
    for p1, p2, p3, p4 in tqdm(all_combinations, total=1663740):
        # 行列を作成
        mat = [
            points_params[p1],
            points_params[p2],
            points_params[p3],
            points_params[p4]
        ]
        
        # 行列式が0なら共円（または共線）
        if det_4x4(mat) == 0:
            # ビットマスクを作成 (2^p1 + 2^p2 + 2^p3 + 2^p4)
            mask = (1 << p1) | (1 << p2) | (1 << p3) | (1 << p4)
            valid_patterns.append(mask)

    print(f"Found {len(valid_patterns)} valid patterns.")

    # 3. ルックアップテーブルの作成
    # index: 最後に打った石の位置 (0-80)
    # value: その石を含むパターン一覧
    # Numbaで高速アクセスするため、パディングして固定長の2D配列にする
    
    # まず、各点ごとのパターンリストを作成
    patterns_by_point = [[] for _ in range(BOARD_SIZE * BOARD_SIZE)]
    
    for mask in valid_patterns:
        # このパターンに含まれる4つの点すべてに登録
        temp = mask
        for i in range(BOARD_SIZE * BOARD_SIZE):
            if (temp >> i) & 1:
                patterns_by_point[i].append(mask)
    
    # 最大パターン数を調べる（配列サイズ決定のため）
    max_len = max(len(lst) for lst in patterns_by_point)
    print(f"Max patterns per point: {max_len}")
    
    # NumPy配列化 (パディング部分は0で埋める)
    # BitboardはPythonでは多倍長整数だが、Numbaに渡すために
    # ここでは便宜上 uint64 を2つ使うか...いや、
    # シンプルに「Pythonのint」のリストをNumba Typed Listにする手もあるが、
    # 配列アクセスの爆速化のためには「3つのuint64」等に分割するのが定石。
    # しかし、Python3.10 + Numbaなら実は int64に入りきらない値も扱える場合があるが、
    # 安全策として「2つのint64 (低位64bit, 高位64bit)」に分割して保存する手法をとる。
    
    # ...と思いましたが、実装簡易化のため、Bitboard (int) をそのまま保存できるか？
    # Numbaは大きな整数定数を扱えますが、配列としては int64 が限界です。
    # 81bitは int64 (63bit) に入りません。
    
    # ★戦略変更★
    # 9x9=81bit なので、1つのパターンを [part1(64bit), part2(64bit)] の2つの整数で表現します。
    
    table_shape = (BOARD_SIZE * BOARD_SIZE, max_len, 2)
    lookup_table = np.zeros(table_shape, dtype=np.uint64)
    
    for i in range(81):
        patterns = patterns_by_point[i]
        for j, mask in enumerate(patterns):
            # 下位64bit
            lower = mask & 0xFFFFFFFFFFFFFFFF
            # 上位64bit (残りのビット)
            upper = mask >> 64
            
            lookup_table[i, j, 0] = lower
            lookup_table[i, j, 1] = upper
            
    # 保存
    np.save(OUTPUT_FILE, lookup_table)
    print(f"Saved lookup table to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()