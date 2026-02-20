import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from collections import deque
from game import Board
from model import SmallResNet

# 設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAMES_PER_EPOCH = 50  # 1回の学習のために自己対戦する回数
BATCH_SIZE = 64
LR = 0.0002           # 学習率を少し下げて安定化
SAVE_DIR = "models"
MAX_MEMORY = 50000    # リプレイバッファの最大サイズ
TRAIN_EPOCHS = 3      # 1回のデータ生成に対する学習エポック数


def self_play_batch(net, n_games=GAMES_PER_EPOCH):
    """全ゲームを同時進行させ、バッチ推論でデータを生成する。
    
    従来の逐次実行 (n_games × n_steps 回の単独推論) と異なり、
    各ステップで全アクティブゲームをバッチとしてまとめて推論するため
    GPU/CPU ともに大幅に高速化される。
    """
    net.eval()

    boards = [Board() for _ in range(n_games)]
    histories = [[] for _ in range(n_games)]
    active = list(range(n_games))
    dataset = []

    while active:
        # アクティブな全ゲームの特徴量を一括収集してバッチ推論
        features = np.array([boards[i].get_feature() for i in active], dtype=np.float32)
        feature_tensor = torch.from_numpy(features).to(DEVICE)

        with torch.no_grad():
            log_pi, _ = net(feature_tensor)
            pi = torch.exp(log_pi).cpu().numpy()  # (n_active, 81)

        next_active = []
        for batch_idx, game_idx in enumerate(active):
            board = boards[game_idx]
            valid_moves = board.valid_moves()

            pi_valid = pi[batch_idx][valid_moves]
            pi_sum = pi_valid.sum()
            if pi_sum > 0:
                pi_valid /= pi_sum
            else:
                pi_valid = np.ones_like(pi_valid) / len(pi_valid)

            # 探索を促進するためにディリクレノイズを追加 (AlphaZeroの定石)
            # 序盤の手数 (例: 30手未満) のみノイズを強くするなどの工夫も可能だが、
            # ここでは常に一定のノイズを混ぜて多様な局面を経験させる
            noise = np.random.dirichlet([0.3] * len(valid_moves))
            pi_valid = 0.75 * pi_valid + 0.25 * noise
            pi_valid /= pi_valid.sum()

            action_idx = np.random.choice(len(valid_moves), p=pi_valid)
            action = valid_moves[action_idx]

            # features[batch_idx] は numpy のビューだが、base array への参照が
            # histories に残るため、次のループで features が再代入されても安全
            histories[game_idx].append((features[batch_idx], action, board.turn))

            _, reward, done = board.step(action)

            if done:
                final_winner = board.turn * -1 if reward > 0 else 0
                for feat, act, turn in histories[game_idx]:
                    v = 1.0 if turn == final_winner else (-1.0 if final_winner != 0 else 0.0)
                    pi_target = np.zeros(81, dtype=np.float32)
                    pi_target[act] = 1.0
                    dataset.append((feat, pi_target, np.float32(v)))
            else:
                next_active.append(game_idx)

        active = next_active

    return dataset


def train():
    print(f"Using Device: {DEVICE}")
    net = SmallResNet().to(DEVICE)

    # torch.compile で推論・学習を JIT 高速化 (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        net = torch.compile(net)

    optimizer = optim.Adam(net.parameters(), lr=LR)

    # TF32 を有効化 (Ampere 以降の GPU で行列積を高速化)
    torch.set_float32_matmul_precision("high")

    # Mixed Precision (CUDA のみ): Tensor Core を活用して約 2x 高速化
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    memory = deque(maxlen=MAX_MEMORY)

    epoch = 0
    try:
        while True:
            epoch += 1

            # 1. 自己対戦 (全ゲーム同時進行・バッチ推論)
            print(f"Epoch {epoch}: Self-playing ({GAMES_PER_EPOCH} games, batched)...")
            dataset = self_play_batch(net, GAMES_PER_EPOCH)
            memory.extend(dataset)

            # 2. 学習 (Update)
            print(f"Training on {len(memory)} samples in memory...")
            net.train()

            # 一括 NumPy 変換 → permutation シャッフル → テンソル化 (コピーなし)
            memory_list = list(memory)
            feats_all = np.array([d[0] for d in memory_list], dtype=np.float32)
            pis_all   = np.array([d[1] for d in memory_list], dtype=np.float32)
            vs_all    = np.array([d[2] for d in memory_list], dtype=np.float32)

            for train_epoch in range(TRAIN_EPOCHS):
                perm = np.random.permutation(len(memory_list))
                feats_all_shuffled = feats_all[perm]
                pis_all_shuffled   = pis_all[perm]
                vs_all_shuffled    = vs_all[perm]

                feats_t = torch.from_numpy(feats_all_shuffled)
                pis_t   = torch.from_numpy(pis_all_shuffled)
                vs_t    = torch.from_numpy(vs_all_shuffled).unsqueeze(1)

                total_loss = 0.0
                n_batches  = 0

                for i in range(0, len(memory_list), BATCH_SIZE):
                    feats = feats_t[i:i+BATCH_SIZE].to(DEVICE, non_blocking=True)
                    pis   = pis_t[i:i+BATCH_SIZE].to(DEVICE, non_blocking=True)
                    vs    = vs_t[i:i+BATCH_SIZE].to(DEVICE, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            out_pi, out_v = net(feats)
                            loss_pi = -torch.sum(vs * pis * out_pi) / feats.size(0)
                            loss_v  = F.mse_loss(out_v, vs)
                            loss    = loss_pi + loss_v
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        out_pi, out_v = net(feats)
                        loss_pi = -torch.sum(vs * pis * out_pi) / feats.size(0)
                        loss_v  = F.mse_loss(out_v, vs)
                        loss    = loss_pi + loss_v
                        loss.backward()
                        optimizer.step()

                    total_loss += loss.item()
                    n_batches  += 1

                print(f"  Train Epoch {train_epoch+1}/{TRAIN_EPOCHS} - Loss: {total_loss / n_batches:.4f}")

            # モデル保存 (torch.compile されている場合は _orig_mod を参照)
            if epoch % 10 == 0:
                save_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pth")
                state_dict = getattr(net, "_orig_mod", net).state_dict()
                torch.save(state_dict, save_path)
                print(f"Model saved to {save_path}")

    except KeyboardInterrupt:
        print("\nInterrupting... Saving current model...")
        save_path = os.path.join(SAVE_DIR, "model_interrupted.pth")
        state_dict = getattr(net, "_orig_mod", net).state_dict()
        torch.save(state_dict, save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()