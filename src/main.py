import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from collections import deque
from game import Board
from model import ResNet
from mcts import mcts_search_with_policy

# 設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAMES_PER_EPOCH = 50  # 1回の学習のために自己対戦する回数
BATCH_SIZE = 64
LR = 0.0002           # 学習率を少し下げて安定化
SAVE_DIR = "models"
MAX_MEMORY = 50000    # リプレイバッファの最大サイズ
TRAIN_EPOCHS = 3      # 1回のデータ生成に対する学習エポック数
MCTS_SIMULATIONS = 128
TEMP_THRESHOLD = 16


class TrainingPolicyValueNet:
    """Adapter to use the current training net as MCTS evaluator."""

    def __init__(self, net, device=None):
        self.net = net
        self.device = device or next(net.parameters()).device

    @torch.no_grad()
    def predict(self, board):
        x = torch.from_numpy(board.get_feature()).unsqueeze(0).to(self.device)
        log_pi, value = self.net(x)
        policy = torch.exp(log_pi)[0].cpu().numpy()
        return policy, float(value.item())


def self_play_alpha_zero(net, n_games=GAMES_PER_EPOCH, num_simulations=MCTS_SIMULATIONS):
    """Generate (state, pi, z) from AlphaZero-style MCTS self-play."""
    net.eval()
    evaluator = TrainingPolicyValueNet(net)

    dataset = []

    for _ in range(n_games):
        board = Board()
        history = []

        while True:
            # AlphaZero: high temperature in opening, near-greedy later.
            temperature = 1.0 if board.move_count < TEMP_THRESHOLD else 1e-8
            move, pi_target = mcts_search_with_policy(
                root_board=board,
                policy_value_net=evaluator,
                num_simulations=num_simulations,
                add_root_noise=True,
                temperature=temperature,
            )

            history.append((board.get_feature(), pi_target, board.turn))
            _, reward, done = board.step(move)

            if done:
                final_winner = board.turn if reward < 0 else 0
                for feat, pi, turn in history:
                    z = 1.0 if turn == final_winner else (-1.0 if final_winner != 0 else 0.0)
                    dataset.append((feat, pi.astype(np.float32), np.float32(z)))
                break

    return dataset


def train():
    print(f"Using Device: {DEVICE}")
    net = ResNet().to(DEVICE)

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
    os.makedirs(SAVE_DIR, exist_ok=True)

    epoch = 0
    try:
        while True:
            epoch += 1

            # 1. AlphaZero形式の自己対戦 (MCTS訪問回数分布を教師に利用)
            print(f"Epoch {epoch}: Self-playing ({GAMES_PER_EPOCH} games, MCTS sims={MCTS_SIMULATIONS})...")
            dataset = self_play_alpha_zero(net, GAMES_PER_EPOCH, MCTS_SIMULATIONS)
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
                            # AlphaZero policy loss: cross entropy with MCTS visit distribution pi.
                            loss_pi = -torch.sum(pis * out_pi, dim=1).mean()
                            loss_v  = F.mse_loss(out_v, vs)
                            loss    = loss_pi + loss_v
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        out_pi, out_v = net(feats)
                        loss_pi = -torch.sum(pis * out_pi, dim=1).mean()
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