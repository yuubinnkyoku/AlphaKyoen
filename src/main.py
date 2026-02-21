import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state
import numpy as np
import os
from collections import deque
from game import Board
from model import ResNet
from mcts import mcts_search_with_policy

# 設定
GAMES_PER_EPOCH = 50  # 1回の学習のために自己対戦する回数
BATCH_SIZE = 64
LR = 0.0002           # 学習率を少し下げて安定化
SAVE_DIR = "models"
MAX_MEMORY = 50000    # リプレイバッファの最大サイズ
TRAIN_EPOCHS = 3      # 1回のデータ生成に対する学習エポック数
MCTS_SIMULATIONS = 128
TEMP_THRESHOLD = 16

class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict

@jax.jit
def predict_step(state, x):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    log_pi, value = ResNet().apply(variables, x, train=False)
    return log_pi, value

class TrainingPolicyValueNet:
    """Adapter to use the current training net as MCTS evaluator."""
    def __init__(self, state):
        self.state = state

    def predict(self, board):
        x = jnp.array(board.get_feature()).reshape(1, 3, 9, 9)
        log_pi, value = predict_step(self.state, x)
        policy = jnp.exp(log_pi)[0]
        return np.array(policy), float(value[0, 0])

def self_play_alpha_zero(state, n_games=GAMES_PER_EPOCH, num_simulations=MCTS_SIMULATIONS):
    """Generate (state, pi, z) from AlphaZero-style MCTS self-play."""
    evaluator = TrainingPolicyValueNet(state)

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

@jax.jit
def train_step(state, feats, pis, vs):
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        (out_pi, out_v), new_model_state = ResNet().apply(
            variables, feats, train=True, mutable=['batch_stats']
        )
        loss_pi = -jnp.sum(pis * out_pi, axis=1).mean()
        loss_v = jnp.mean((out_v - vs) ** 2)
        loss = loss_pi + loss_v
        return loss, new_model_state

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, new_model_state), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state['batch_stats'])
    return state, loss

def train():
    print(f"Using JAX backend: {jax.default_backend()}")
    
    net = ResNet()
    dummy_x = jnp.zeros((1, 3, 9, 9), dtype=jnp.float32)
    key = jax.random.PRNGKey(42)
    variables = net.init(key, dummy_x, train=False)

    tx = optax.adam(learning_rate=LR)
    state = TrainState.create(
        apply_fn=net.apply,
        params=variables['params'],
        tx=tx,
        batch_stats=variables['batch_stats'],
    )

    memory = deque(maxlen=MAX_MEMORY)
    os.makedirs(SAVE_DIR, exist_ok=True)

    epoch = 0
    try:
        while True:
            epoch += 1

            # 1. AlphaZero形式の自己対戦 (MCTS訪問回数分布を教師に利用)
            print(f"Epoch {epoch}: Self-playing ({GAMES_PER_EPOCH} games, MCTS sims={MCTS_SIMULATIONS})...")
            dataset = self_play_alpha_zero(state, GAMES_PER_EPOCH, MCTS_SIMULATIONS)
            memory.extend(dataset)

            # 2. 学習 (Update)
            print(f"Training on {len(memory)} samples in memory...")

            memory_list = list(memory)
            feats_all = np.array([d[0] for d in memory_list], dtype=np.float32)
            pis_all   = np.array([d[1] for d in memory_list], dtype=np.float32)
            vs_all    = np.array([d[2] for d in memory_list], dtype=np.float32)

            for train_epoch in range(TRAIN_EPOCHS):
                perm = np.random.permutation(len(memory_list))
                feats_all_shuffled = feats_all[perm]
                pis_all_shuffled   = pis_all[perm]
                vs_all_shuffled    = vs_all[perm]

                total_loss = 0.0
                n_batches  = 0

                for i in range(0, len(memory_list), BATCH_SIZE):
                    feats = feats_all_shuffled[i:i+BATCH_SIZE]
                    pis   = pis_all_shuffled[i:i+BATCH_SIZE]
                    vs    = vs_all_shuffled[i:i+BATCH_SIZE].reshape(-1, 1)

                    state, loss = train_step(state, feats, pis, vs)

                    total_loss += float(loss)
                    n_batches  += 1

                print(f"  Train Epoch {train_epoch+1}/{TRAIN_EPOCHS} - Loss: {total_loss / n_batches:.4f}")

            # モデル保存
            if epoch % 10 == 0:
                save_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch}.msgpack")
                with open(save_path, "wb") as f:
                    f.write(flax.serialization.to_bytes({'params': state.params, 'batch_stats': state.batch_stats}))
                print(f"Model saved to {save_path}")

    except KeyboardInterrupt:
        print("\nInterrupting... Saving current model...")
        save_path = os.path.join(SAVE_DIR, "model_interrupted.msgpack")
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes({'params': state.params, 'batch_stats': state.batch_stats}))
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()