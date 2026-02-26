from __future__ import annotations

import os
import sys
from threading import Lock
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Allow imports from src/ when running `uvicorn app:app` at repository root.
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from game import Board  # noqa: E402
from mcts import PolicyValueNet, mcts_search  # noqa: E402


class GameState(BaseModel):
    board: List[int] = Field(..., min_length=81, max_length=81)
    turn: int = Field(..., description="1 for O turn, -1 for X turn")


class MoveRequest(BaseModel):
    state: GameState
    move: int = Field(..., ge=0, le=80)


class HintRequest(BaseModel):
    state: GameState


class MoveResponse(BaseModel):
    board: List[int]
    turn: int
    move_count: int
    done: bool
    reward: float
    kyoen_points: List[int]
    move: int


class HintResponse(BaseModel):
    hints: List[int]


app = FastAPI(title="AlphaKyoen API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_lock = Lock()
_policy_model: PolicyValueNet | None = None


@app.get("/")
def root() -> dict:
    return {
        "service": "AlphaKyoen API",
        "message": "Backend is running. This Space serves API endpoints.",
        "health": "/api/health",
        "docs": "/docs",
        "frontend": "https://yuubinnkyoku.github.io/AlphaKyoen/",
    }


def _state_to_board(state: GameState) -> Board:
    arr = np.array(state.board, dtype=np.int8)
    if not np.all(np.isin(arr, np.array([-1, 0, 1], dtype=np.int8))):
        raise HTTPException(status_code=400, detail="Board values must be -1, 0, or 1.")

    if state.turn not in (-1, 1):
        raise HTTPException(status_code=400, detail="Turn must be 1 or -1.")

    b = Board()
    b.board = arr
    b.turn = int(state.turn)
    b.move_count = int(np.count_nonzero(arr))

    bb_me = 0
    bb_opp = 0
    for idx, val in enumerate(arr.tolist()):
        if val == 1:
            bb_me |= 1 << idx
        elif val == -1:
            bb_opp |= 1 << idx

    b.bb_me = bb_me
    b.bb_opp = bb_opp
    return b


def _board_to_response(board: Board, move: int, done: bool, reward: float, kyoen_points: List[int]) -> MoveResponse:
    return MoveResponse(
        board=board.board.astype(int).tolist(),
        turn=int(board.turn),
        move_count=int(board.move_count),
        done=bool(done),
        reward=float(reward),
        kyoen_points=[int(x) for x in kyoen_points],
        move=int(move),
    )


def _get_policy_model() -> PolicyValueNet:
    global _policy_model
    with _model_lock:
        if _policy_model is None:
            _policy_model = PolicyValueNet()
    return _policy_model


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/move", response_model=MoveResponse)
def post_move(req: MoveRequest) -> MoveResponse:
    board = _state_to_board(req.state)

    if board.board[req.move] != 0:
        raise HTTPException(status_code=400, detail="Invalid move: already occupied.")

    _, reward, done, info = board.step(req.move, return_details=True)
    return _board_to_response(board, req.move, done, reward, info["kyoen_points"])


@app.post("/api/ai_move", response_model=MoveResponse)
def post_ai_move(req: HintRequest) -> MoveResponse:
    board = _state_to_board(req.state)

    valid = board.valid_moves()
    if len(valid) == 0:
        raise HTTPException(status_code=400, detail="No valid moves available.")

    model = _get_policy_model()
    ai_move = mcts_search(board, model, num_simulations=400)
    _, reward, done, info = board.step(ai_move, return_details=True)
    return _board_to_response(board, ai_move, done, reward, info["kyoen_points"])


@app.post("/api/hint", response_model=HintResponse)
def post_hint(req: HintRequest) -> HintResponse:
    board = _state_to_board(req.state)
    return HintResponse(hints=board.get_hint_moves())
