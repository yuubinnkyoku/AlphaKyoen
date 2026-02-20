import math
import random
import time
import curses
import numpy as np
from game import Board

class MCTSNode:
    def __init__(self, board, parent=None, move=None, done=False, winner=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.done = done
        self.winner = winner
        
        if self.done:
            self.untried_moves = []
        else:
            self.untried_moves = list(board.valid_moves())
            
    def uct_select_child(self, c_puct=1.414):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            if child.visits == 0:
                return child
            exploit = child.wins / child.visits
            explore = math.sqrt(math.log(self.visits) / child.visits)
            score = exploit + c_puct * explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def update(self, result):
        self.visits += 1
        self.wins += result

def playout(board):
    current_board = board.copy()
    
    while True:
        valid_moves = current_board.valid_moves()
        if len(valid_moves) == 0:
            return 0 # Draw
            
        move = random.choice(valid_moves)
        turn_current = current_board.turn
        _, reward, done = current_board.step(move)
        
        if done:
            if reward > 0:
                return turn_current
            else:
                return 0

def mcts_search(root_board, num_simulations=1000):
    root = MCTSNode(root_board)
    
    for _ in range(num_simulations):
        node = root
        
        # 1. Selection
        while not node.done and len(node.untried_moves) == 0 and len(node.children) > 0:
            node = node.uct_select_child()
            
        # 2. Expansion
        if not node.done and len(node.untried_moves) > 0:
            move = random.choice(node.untried_moves)
            node.untried_moves.remove(move)
            
            next_board = node.board.copy()
            turn_played = next_board.turn
            _, reward, done = next_board.step(move)
            
            winner = turn_played if (done and reward > 0) else 0
            
            child_node = MCTSNode(next_board, parent=node, move=move, done=done, winner=winner)
            node.children.append(child_node)
            node = child_node
            
            if done:
                sim_winner = winner
            else:
                sim_winner = playout(next_board)
        else:
            sim_winner = node.winner
            
        # 3. Backpropagation
        current = node
        while current is not None:
            if current.parent is not None:
                turn_played = current.parent.board.turn
                if sim_winner == turn_played:
                    result = 1.0
                elif sim_winner == -turn_played:
                    result = 0.0
                else:
                    result = 0.5
            else:
                result = 0.5
                
            current.update(result)
            current = current.parent
            
    if not root.children:
        return random.choice(root_board.valid_moves())
        
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.move

def draw_board(stdscr, board, cursor_r, cursor_c, message=""):
    stdscr.clear()
    stdscr.addstr(0, 0, "=== AlphaKyoen MCTS ===")
    
    symbols = {1: 'O', -1: 'X', 0: '.'}
    
    # 盤面の描画
    for r in range(9):
        for c in range(9):
            idx = r * 9 + c
            sym = symbols[board.board[idx]]
            
            # カーソル位置はハイライト
            if r == cursor_r and c == cursor_c:
                stdscr.addstr(r + 2, c * 2 + 2, sym, curses.A_REVERSE)
            else:
                stdscr.addstr(r + 2, c * 2 + 2, sym)
                
    stdscr.addstr(12, 0, message)
    stdscr.refresh()

def play_game_curses(stdscr):
    curses.curs_set(0) # カーソルを隠す
    stdscr.clear()
    
    stdscr.addstr(0, 0, "=== AlphaKyoen MCTS ===")
    stdscr.addstr(2, 0, "Do you want to play first? (y/n): ")
    stdscr.refresh()
    
    while True:
        key = stdscr.getch()
        if key in [ord('y'), ord('Y')]:
            player_turn = 1
            break
        elif key in [ord('n'), ord('N')]:
            player_turn = -1
            break
            
    player_symbol = 'O' if player_turn == 1 else 'X'
    ai_symbol = 'X' if player_turn == 1 else 'O'
    
    board = Board()
    cursor_r, cursor_c = 4, 4
    
    while True:
        if board.turn == player_turn:
            # Player's turn
            while True:
                draw_board(stdscr, board, cursor_r, cursor_c, f"Your turn ({player_symbol}). Use Arrow keys to move, Enter to place.")
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
                    else:
                        draw_board(stdscr, board, cursor_r, cursor_c, "Invalid move! Press any key...")
                        stdscr.getch()
                        
            _, reward, done = board.step(move)
            if done:
                msg = "You win! Press any key to exit." if reward > 0 else "Draw! Press any key to exit."
                draw_board(stdscr, board, -1, -1, msg)
                stdscr.getch()
                break
        else:
            # AI's turn
            draw_board(stdscr, board, -1, -1, f"AI ({ai_symbol}) is thinking...")
            start_time = time.time()
            move = mcts_search(board, num_simulations=3000)
            elapsed = time.time() - start_time
            
            _, reward, done = board.step(move)
            if done:
                msg = f"AI played {move//9} {move%9} ({elapsed:.2f}s). AI wins! Press any key to exit." if reward > 0 else "Draw! Press any key to exit."
                draw_board(stdscr, board, -1, -1, msg)
                stdscr.getch()
                break

def ai_vs_random(num_games=10, mcts_simulations=1000):
    mcts_wins = 0
    random_wins = 0
    draws = 0
    
    for i in range(num_games):
        board = Board()
        mcts_turn = 1 if i % 2 == 0 else -1
        
        while True:
            if board.turn == mcts_turn:
                move = mcts_search(board, num_simulations=mcts_simulations)
            else:
                move = random.choice(board.valid_moves())
                
            _, reward, done = board.step(move)
            
            if done:
                if reward > 0:
                    if board.turn * -1 == mcts_turn:
                        mcts_wins += 1
                    else:
                        random_wins += 1
                else:
                    draws += 1
                break
                
    print(f"MCTS vs Random: MCTS {mcts_wins} - {random_wins} Random (Draws: {draws})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        ai_vs_random(10, 1000)
    else:
        curses.wrapper(play_game_curses)
