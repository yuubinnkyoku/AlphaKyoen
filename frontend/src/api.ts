import type { GameState, MoveResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const message = await res.text();
    throw new Error(message || `Request failed: ${res.status}`);
  }

  return (await res.json()) as T;
}

export function playerMove(state: GameState, move: number): Promise<MoveResponse> {
  return postJson<MoveResponse>("/api/move", { state, move });
}

export function aiMove(state: GameState): Promise<MoveResponse> {
  return postJson<MoveResponse>("/api/ai_move", { state });
}

export async function getHints(state: GameState): Promise<number[]> {
  const data = await postJson<{ hints: number[] }>("/api/hint", { state });
  return data.hints;
}
