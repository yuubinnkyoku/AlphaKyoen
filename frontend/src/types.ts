export type Turn = 1 | -1;

export interface GameState {
  board: number[];
  turn: Turn;
}

export interface MoveResponse {
  board: number[];
  turn: Turn;
  move_count: number;
  done: boolean;
  reward: number;
  kyoen_points: number[];
  move: number;
}
