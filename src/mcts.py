import curses
import math
import os
import random
import time

import numpy as np
import jax
import jax.numpy as jnp
import flax

from game import Board
from model import ResNet

def find_latest_model_path(save_dir="models"):
    if not os.path.isdir(save_dir):
        return None

    epoch_models = []
    for filename in os.listdir(save_dir):
        if not filename.startswith("model_epoch_") or not filename.endswith(".msgpack"):
            continue
        try:
            epoch = int(filename[len("model_epoch_") : -len(".msgpack")])
            epoch_models.append((epoch, os.path.join(save_dir, filename)))
        except ValueError:
            continue

    if epoch_models:
        epoch_models.sort(key=lambda x: x[0])
        return epoch_models[-1][1]

    interrupted = os.path.join(save_dir, "model_interrupted.msgpack")
    if os.path.exists(interrupted):
        return interrupted

    return None

@jax.jit
def predict_step(params, batch_stats, x):
    variables = {'params': params, 'batch_stats': batch_stats}
    log_pi, value = ResNet().apply(variables, x, train=False)
    return log_pi, value

class PolicyValueNet:
    def __init__(self, model_path=None):
        self.net = ResNet()
        dummy_x = jnp.zeros((1, 3, 9, 9), dtype=jnp.float32)
        variables = self.net.init(jax.random.PRNGKey(0), dummy_x, train=False)
        self.params = variables['params']
        self.batch_stats = variables['batch_stats']

        if model_path is None:
            model_path = find_latest_model_path()

        if model_path is not None and os.path.exists(model_path):
            with open(model_path, "rb") as f:
                loaded_vars = flax.serialization.from_bytes(variables, f.read())
                self.params = loaded_vars['params']
                self.batch_stats = loaded_vars['batch_stats']
            print(f"Loaded model: {model_path}")
        else:
            print("No checkpoint found. Using randomly initialized ResNet.")

    def predict(self, board):
        x = jnp.array(board.get_feature()).reshape(1, 3, 9, 9)
        log_pi, value = predict_step(self.params, self.batch_stats, x)
        policy = jnp.exp(log_pi)[0]
        return np.array(policy), float(value[0, 0])

class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior=0.0, done=False, winner=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.done = done
        self.winner = winner

        self.value_sum = 0.0
        self.visit_count = 0
        self.children = {}

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct=1.5):
        best_score = -float("inf")
        best_child = None

        sqrt_parent_visits = math.sqrt(self.visit_count + 1)
        for child in self.children.values():
            # child.value() is from child.board.turn perspective, so invert for parent perspective.
            q = -child.value()
            u = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            score = q + u

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

def terminal_value(node):
    if node.winner == 0:
        return 0.0
    return 1.0 if node.winner == node.board.turn else -1.0

def expand_node(node, policy):
    valid_moves = node.board.valid_moves()
    if len(valid_moves) == 0:
        return

    priors = policy[valid_moves]
    prior_sum = np.sum(priors)
    if prior_sum > 0:
        priors = priors / prior_sum
    else:
        priors = np.ones_like(priors, dtype=np.float32) / len(priors)

    for move, prior in zip(valid_moves, priors):
        next_board = node.board.copy()
        _, reward, done = next_board.step(move)

        if done and reward < 0:
            winner = next_board.turn
        else:
            winner = 0

        child = MCTSNode(
            board=next_board,
            parent=node,
            move=int(move),
            prior=float(prior),
            done=done,
            winner=winner,
        )
        node.children[int(move)] = child

def backpropagate(search_path, value):
    for node in reversed(search_path):
        node.visit_count += 1
        node.value_sum += value
        value = -value

def _run_mcts(
    root_board,
    policy_value_net,
    num_simulations=800,
    c_puct=1.5,
    dirichlet_alpha=0.3,
    dirichlet_eps=0.25,
    add_root_noise=False,
):
    valid_root_moves = root_board.valid_moves()
    if len(valid_root_moves) == 0:
        raise ValueError("No valid moves available for MCTS search.")

    root = MCTSNode(root_board.copy())
    root_policy, _ = policy_value_net.predict(root.board)
    expand_node(root, root_policy)

    # Add Dirichlet noise at root for self-play exploration.
    if add_root_noise and root.children:
        moves = list(root.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(moves))
        for idx, move in enumerate(moves):
            child = root.children[move]
            child.prior = (1.0 - dirichlet_eps) * child.prior + dirichlet_eps * float(noise[idx])

    for _ in range(num_simulations):
        node = root
        search_path = [node]

        while node.children:
            node = node.select_child(c_puct=c_puct)
            search_path.append(node)
            if node.done:
                break

        if node.done:
            value = terminal_value(node)
        else:
            policy, value = policy_value_net.predict(node.board)
            expand_node(node, policy)

        backpropagate(search_path, value)

    return root

def _visits_to_policy(root, temperature=1.0):
    pi = np.zeros(81, dtype=np.float32)
    if not root.children:
        return pi

    moves = np.array(list(root.children.keys()), dtype=np.int64)
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float32)

    if temperature <= 1e-6:
        best = int(np.argmax(visits))
        probs = np.zeros_like(visits, dtype=np.float32)
        probs[best] = 1.0
    else:
        inv_t = 1.0 / temperature
        probs = np.power(visits, inv_t).astype(np.float32)
        s = probs.sum()
        if s > 0:
            probs /= s
        else:
            probs = np.ones_like(probs, dtype=np.float32) / len(probs)

    pi[moves] = probs
    return pi

def mcts_search_with_policy(
    root_board,
    policy_value_net,
    num_simulations=800,
    c_puct=1.5,
    dirichlet_alpha=0.3,
    dirichlet_eps=0.25,
    add_root_noise=False,
    temperature=1.0,
):
    root = _run_mcts(
        root_board=root_board,
        policy_value_net=policy_value_net,
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_eps=dirichlet_eps,
        add_root_noise=add_root_noise,
    )

    pi = _visits_to_policy(root, temperature=temperature)
    move = int(np.random.choice(np.arange(81), p=pi))
    return move, pi

def mcts_search(
    root_board,
    policy_value_net,
    num_simulations=800,
    c_puct=1.5,
):
    move, _ = mcts_search_with_policy(
        root_board=root_board,
        policy_value_net=policy_value_net,
        num_simulations=num_simulations,
        c_puct=c_puct,
        add_root_noise=False,
        temperature=1e-8,
    )
    return move

def draw_board(stdscr, board, cursor_r, cursor_c, message=""):
    stdscr.clear()
    stdscr.addstr(0, 0, "=== AlphaKyoen AlphaZero MCTS ===")

    symbols = {1: "O", -1: "X", 0: "."}

    for r in range(9):
        for c in range(9):
            idx = r * 9 + c
            sym = symbols[board.board[idx]]

            if r == cursor_r and c == cursor_c:
                stdscr.addstr(r + 2, c * 2 + 2, sym, curses.A_REVERSE)
            else:
                stdscr.addstr(r + 2, c * 2 + 2, sym)

    stdscr.addstr(12, 0, message)
    stdscr.refresh()

def play_game_curses(stdscr):
    curses.curs_set(0)
    stdscr.clear()

    stdscr.addstr(0, 0, "=== AlphaKyoen AlphaZero MCTS ===")
    stdscr.addstr(2, 0, "Do you want to play first? (y/n): ")
    stdscr.refresh()

    while True:
        key = stdscr.getch()
        if key in [ord("y"), ord("Y")]:
            player_turn = 1
            break
        if key in [ord("n"), ord("N")]:
            player_turn = -1
            break

    player_symbol = "O" if player_turn == 1 else "X"
    ai_symbol = "X" if player_turn == 1 else "O"

    board = Board()
    cursor_r, cursor_c = 4, 4
    policy_value_net = PolicyValueNet()

    while True:
        if board.turn == player_turn:
            while True:
                draw_board(
                    stdscr,
                    board,
                    cursor_r,
                    cursor_c,
                    f"Your turn ({player_symbol}). Use Arrow keys to move, Enter to place.",
                )
                key = stdscr.getch()

                if key == curses.KEY_UP and cursor_r > 0:
                    cursor_r -= 1
                elif key == curses.KEY_DOWN and cursor_r < 8:
                    cursor_r += 1
                elif key == curses.KEY_LEFT and cursor_c > 0:
                    cursor_c -= 1
                elif key == curses.KEY_RIGHT and cursor_c < 8:
                    cursor_c += 1
                elif key in [curses.KEY_ENTER, 10, 13]:
                    move = cursor_r * 9 + cursor_c
                    if move in board.valid_moves():
                        break
                    draw_board(stdscr, board, cursor_r, cursor_c, "Invalid move! Press any key...")
                    stdscr.getch()

            _, reward, done = board.step(move)
            if done:
                msg = "You lose! Press any key to exit." if reward < 0 else "Draw! Press any key to exit."
                draw_board(stdscr, board, -1, -1, msg)
                stdscr.getch()
                break
        else:
            draw_board(stdscr, board, -1, -1, f"AI ({ai_symbol}) is thinking...")
            start_time = time.time()
            move = mcts_search(board, policy_value_net, num_simulations=1200)
            elapsed = time.time() - start_time

            _, reward, done = board.step(move)
            if done:
                if reward < 0:
                    msg = f"AI played {move // 9} {move % 9} ({elapsed:.2f}s). AI loses! Press any key to exit."
                else:
                    msg = "Draw! Press any key to exit."
                draw_board(stdscr, board, -1, -1, msg)
                stdscr.getch()
                break

def ai_vs_random(num_games=10, mcts_simulations=800):
    policy_value_net = PolicyValueNet()

    mcts_wins = 0
    random_wins = 0
    draws = 0

    for i in range(num_games):
        board = Board()
        mcts_turn = 1 if i % 2 == 0 else -1

        while True:
            if board.turn == mcts_turn:
                move = mcts_search(board, policy_value_net, num_simulations=mcts_simulations)
            else:
                move = random.choice(board.valid_moves())

            _, reward, done = board.step(move)

            if done:
                if reward < 0:
                    if board.turn * -1 == mcts_turn:
                        random_wins += 1
                    else:
                        mcts_wins += 1
                else:
                    draws += 1
                break

    print(f"NN-MCTS vs Random: NN-MCTS {mcts_wins} - {random_wins} Random (Draws: {draws})")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        ai_vs_random(10, 800)
    else:
        curses.wrapper(play_game_curses)
