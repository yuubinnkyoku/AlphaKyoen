from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import GameState, HintRequest, MoveRequest, health, post_ai_move, post_hint, post_move


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-ai", action="store_true", help="Include /api/ai_move in smoke test.")
    args = parser.parse_args()

    state = GameState(board=[0] * 81, turn=1)

    h = health()
    print("health:", h)

    hints = post_hint(HintRequest(state=state))
    print("hint_count:", len(hints.hints))

    move_resp = post_move(MoveRequest(state=state, move=40))
    print("player_move:", move_resp.move)
    print("move_count:", move_resp.move_count)
    print("done_after_player:", move_resp.done)

    if args.with_ai:
        ai_state = GameState(board=move_resp.board, turn=move_resp.turn)
        ai_resp = post_ai_move(HintRequest(state=ai_state))
        print("ai_move:", ai_resp.move)
        print("move_count_after_ai:", ai_resp.move_count)
        print("done_after_ai:", ai_resp.done)
    else:
        print("ai_move: skipped (run with --with-ai)")


if __name__ == "__main__":
    main()
