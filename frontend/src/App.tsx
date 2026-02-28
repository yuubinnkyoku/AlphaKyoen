import { TFunction } from "i18next";
import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { aiMove, getHints, playerMove } from "./api";
import type { GameState, MoveResponse, Turn } from "./types";

const SIZE = 9;

function emptyState(): GameState {
  return {
    board: Array.from({ length: SIZE * SIZE }, () => 0),
    turn: 1,
  };
}

function idxToLabel(idx: number): string {
  const row = Math.floor(idx / SIZE) + 1;
  const col = (idx % SIZE) + 1;
  return `${row},${col}`;
}

function getCircumcircle(p1: [number, number], p2: [number, number], p3: [number, number]) {
  const [x1, y1] = p1;
  const [x2, y2] = p2;
  const [x3, y3] = p3;
  const d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
  if (Math.abs(d) < 1e-6) return null;
  const h1 = x1 * x1 + y1 * y1;
  const h2 = x2 * x2 + y2 * y2;
  const h3 = x3 * x3 + y3 * y3;
  const cx = (h1 * (y2 - y3) + h2 * (y3 - y1) + h3 * (y1 - y2)) / d;
  const cy = (h1 * (x3 - x2) + h2 * (x1 - x3) + h3 * (x2 - x1)) / d;
  const r = Math.sqrt((cx - x1) ** 2 + (cy - y1) ** 2);
  return { cx, cy, r };
}

function getStatusText(
  t: TFunction,
  resp: MoveResponse | null,
  currentTurn: Turn,
  humanSide: Turn,
  isAiThinking: boolean
): string {
  if (isAiThinking) return t("thinking");
  if (!resp || !resp.done) return currentTurn === humanSide ? t("yourTurn") : t("aiTurn");

  return resp.turn === humanSide ? t("winYou") : t("winAi");
}

export default function App() {
  const { t, i18n } = useTranslation();
  const [state, setState] = useState<GameState>(emptyState);
  const [humanSide, setHumanSide] = useState<Turn>(1);
  const [isAiThinking, setIsAiThinking] = useState(false);
  const [error, setError] = useState<string>("");
  const [done, setDone] = useState(false);
  const [lastResult, setLastResult] = useState<MoveResponse | null>(null);
  const [showHints, setShowHints] = useState(false);
  const [hintMoves, setHintMoves] = useState<number[]>([]);
  const [showResult, setShowResult] = useState(false);
  const prevBoard = useRef<number[]>([]);

  const hintSet = useMemo(() => new Set(hintMoves), [hintMoves]);
  const resultSet = useMemo(() => new Set(lastResult?.kyoen_points ?? []), [lastResult]);
  const newStones = useMemo(() => {
    const s = new Set<number>();
    state.board.forEach((cell, i) => {
      if (cell !== 0 && (prevBoard.current[i] ?? 0) === 0) s.add(i);
    });
    return s;
  }, [state.board]);

  useEffect(() => {
    prevBoard.current = [...state.board];
  }, [state.board]);

  const changeLanguage = (lng: string) => {
    void i18n.changeLanguage(lng);
  };

  async function refreshHints(nextState: GameState, enabled = showHints, gameDone = done): Promise<void> {
    if (!enabled || gameDone) {
      setHintMoves([]);
      return;
    }
    const hints = await getHints(nextState);
    setHintMoves(hints);
  }

  async function runAiTurn(nextState: GameState): Promise<void> {
    setIsAiThinking(true);
    try {
      const aiResp = await aiMove(nextState);
      const aiState: GameState = { board: aiResp.board, turn: aiResp.turn };
      setState(aiState);
      setLastResult(aiResp);
      setDone(aiResp.done);
      await refreshHints(aiState, showHints, aiResp.done);
    } finally {
      setIsAiThinking(false);
    }
  }

  async function startNewGame(nextHumanSide: Turn): Promise<void> {
    const init = emptyState();
    prevBoard.current = [];
    setState(init);
    setIsAiThinking(false);
    setError("");
    setDone(false);
    setLastResult(null);
    setShowHints(false);
    setHintMoves([]);
    setShowResult(false);

    if (nextHumanSide === -1) {
      await runAiTurn(init);
    }
  }

  async function onCellClick(idx: number): Promise<void> {
    if (done || isAiThinking) return;
    if (state.turn !== humanSide) return;
    if (state.board[idx] !== 0) return;

    setError("");
    try {
      const moveResp = await playerMove(state, idx);
      const nextState: GameState = { board: moveResp.board, turn: moveResp.turn };
      setState(nextState);
      setLastResult(moveResp);
      setDone(moveResp.done);

      if (moveResp.done) {
        setHintMoves([]);
        return;
      }

      await refreshHints(nextState, showHints, false);
      await runAiTurn(nextState);
    } catch (e) {
      setError(e instanceof Error ? e.message : t("unexpectedError"));
    }
  }

  async function onToggleHints(): Promise<void> {
    const enabled = !showHints;
    setShowHints(enabled);
    if (enabled) {
      try {
        await refreshHints(state, true);
      } catch (e) {
        setError(e instanceof Error ? e.message : t("failedFetchHints"));
      }
    } else {
      setHintMoves([]);
    }
  }

  async function onOrderChange(value: string): Promise<void> {
    const nextHumanSide: Turn = value === "first" ? 1 : -1;
    setHumanSide(nextHumanSide);
    await startNewGame(nextHumanSide);
  }

  async function onReset(): Promise<void> {
    await startNewGame(humanSide);
  }

  const renderCircumcircle = () => {
    if (!lastResult?.kyoen_points || lastResult.kyoen_points.length < 3) return null;
    const pts = lastResult.kyoen_points.slice(0, 3).map((idx) => {
      const row = Math.floor(idx / SIZE) + 1;
      const col = (idx % SIZE) + 1;
      return [col, row] as [number, number];
    });
    const circle = getCircumcircle(pts[0], pts[1], pts[2]);
    if (!circle) return null;
    return (
      <circle
        cx={circle.cx}
        cy={circle.cy}
        r={circle.r}
        fill="transparent"
        stroke="#f6c744"
        strokeWidth={0.05}
        strokeDasharray="0.12 0.06"
        opacity={0.8}
      />
    );
  };

  const statusText = getStatusText(t, lastResult, state.turn, humanSide, isAiThinking);
  const isWin = lastResult?.done && lastResult.turn === humanSide;
  const isLoss = lastResult?.done && lastResult.turn !== humanSide;

  return (
    <div className="min-h-screen grid place-items-center px-4 py-6 max-sm:px-2 max-sm:py-3">
      <main className="grid gap-4 w-full max-w-[560px]">
        {/* ── Header ── */}
        <header className="flex items-end justify-between">
          <div>
            <h1 className="font-display font-bold text-3xl tracking-tight max-sm:text-2xl">
              <span className="text-brand">Alpha</span>
              <span className="text-ink">Kyoen</span>
            </h1>
            <p className="text-xs text-ink/40 font-display tracking-widest uppercase mt-0.5">MCTS + Neural Net</p>
          </div>
          <div className="flex gap-1 items-center">
            {(["en", "ja"] as const).map((lng) => (
              <button
                key={lng}
                className={`px-2.5 py-1 text-xs rounded-md font-display font-bold cursor-pointer transition-all
                  ${i18n.language === lng
                    ? "bg-brand text-[#0d1117]"
                    : "bg-transparent text-ink/40 hover:text-ink/70"
                  }`}
                onClick={() => changeLanguage(lng)}
              >
                {lng.toUpperCase()}
              </button>
            ))}
          </div>
        </header>

        {/* ── Control Bar ── */}
        <section className="flex flex-wrap gap-2 items-center rounded-xl px-4 py-2.5 border border-line/60 bg-surface/80 backdrop-blur-sm">
          <div className="flex items-center gap-2">
            <label htmlFor="order-select" className="font-display font-bold text-sm text-ink/60">{t("order")}</label>
            <select
              id="order-select"
              className="control-input"
              value={humanSide === 1 ? "first" : "second"}
              onChange={(e) => void onOrderChange(e.target.value)}
            >
              <option value="first">{t("youFirst")}</option>
              <option value="second">{t("youSecond")}</option>
            </select>
          </div>

          <div className="flex gap-1.5 ml-auto">
            <button type="button" className={`btn ${showResult ? "btn-active" : ""}`} onClick={() => setShowResult((v) => !v)}>
              {t("result")}
            </button>
            <button type="button" className={`btn ${showHints ? "btn-active" : ""}`} onClick={onToggleHints} disabled={done || isAiThinking}>
              {showHints ? t("hintOff") : t("hint")}
            </button>
            <button type="button" className="btn btn-accent" onClick={() => void onReset()}>
              {t("reset")}
            </button>
          </div>
        </section>

        {/* ── Status ── */}
        <div className="flex items-center gap-3 px-1">
          <div className={`flex items-center gap-2 font-display font-bold text-sm
            ${isWin ? "text-brand" : isLoss ? "text-error" : isAiThinking ? "text-accent thinking-pulse" : "text-ink/70"}`}>
            {isAiThinking && (
              <span className="inline-block w-2 h-2 rounded-full bg-accent" />
            )}
            {statusText}
          </div>
          {showResult && lastResult?.kyoen_points?.length ? (
            <div className="text-xs font-display text-accent/80">
              {t("kyoenPoints")}: {lastResult.kyoen_points.map((i) => idxToLabel(i)).join(" · ")}
            </div>
          ) : null}
        </div>
        {error && <div className="text-error text-sm font-display font-bold px-1">{error}</div>}

        {/* ── Board ── */}
        <section className="w-full grid place-items-center" aria-label="Kyoen board">
          <div className="relative w-full max-w-[94vw] sm:max-w-[520px] rounded-2xl p-3 bg-surface border border-line/40">
            <svg className="block aspect-square w-full" viewBox="0 0 10 10" width="100%" height="100%">
              <defs>
                <radialGradient id="stone-human" cx="35%" cy="35%">
                  <stop offset="0%" stopColor="#ff6b6b" />
                  <stop offset="100%" stopColor="#c0392b" />
                </radialGradient>
                <radialGradient id="stone-computer" cx="35%" cy="35%">
                  <stop offset="0%" stopColor="#81e6d9" />
                  <stop offset="100%" stopColor="#319795" />
                </radialGradient>
                <filter id="stone-shadow">
                  <feDropShadow dx="0" dy="0.03" stdDeviation="0.04" floodOpacity="0.5" />
                </filter>
              </defs>

              {/* Board background */}
              <rect x="0.4" y="0.4" width="9.2" height="9.2" rx="0.15" fill="#1a2233" />

              {/* Grid Lines */}
              {Array.from({ length: SIZE }, (_, i) => i + 1).map((pos) => (
                <g key={`lines-${pos}`}>
                  <line x1={0.8} y1={pos} x2={9.2} y2={pos} stroke="#2d3748" strokeWidth={0.025} />
                  <line x1={pos} y1={0.8} x2={pos} y2={9.2} stroke="#2d3748" strokeWidth={0.025} />
                </g>
              ))}

              {/* Star points */}
              {[3, 5, 7].flatMap((r) =>
                [3, 5, 7].map((c) => (
                  <circle key={`star-${r}-${c}`} cx={c} cy={r} r={0.06} fill="#2d3748" />
                ))
              )}

              {/* Cells */}
              {state.board.map((cell, idx) => {
                const row = Math.floor(idx / SIZE) + 1;
                const col = (idx % SIZE) + 1;
                const hint = showHints && hintSet.has(idx) && cell === 0;
                const result = showResult && resultSet.has(idx);
                const disabled = done || isAiThinking || state.turn !== humanSide || cell !== 0;
                const isNew = newStones.has(idx);

                return (
                  <g key={`cell-${idx}`} transform={`translate(${col}, ${row})`}>
                    {(hint || result) && (
                      <rect
                        x={-0.5}
                        y={-0.5}
                        width={1}
                        height={1}
                        className={result ? "cell-bg result" : "cell-bg hint"}
                      />
                    )}

                    <rect
                      x={-0.5}
                      y={-0.5}
                      width={1}
                      height={1}
                      className={`cell-interactive ${disabled ? "disabled" : ""}`}
                      onClick={() => {
                        if (!disabled) void onCellClick(idx);
                      }}
                    />

                    {cell === humanSide && (
                      <circle
                        r={0.3}
                        fill="url(#stone-human)"
                        filter="url(#stone-shadow)"
                        className={`stone-you ${isNew ? "stone-new" : ""}`}
                      />
                    )}
                    {cell !== 0 && cell !== humanSide && (
                      <circle
                        r={0.3}
                        fill="url(#stone-computer)"
                        filter="url(#stone-shadow)"
                        className={`stone-ai ${isNew ? "stone-new" : ""}`}
                      />
                    )}
                  </g>
                );
              })}

              {showResult && renderCircumcircle()}
            </svg>
          </div>
        </section>
      </main>
    </div>
  );
}
