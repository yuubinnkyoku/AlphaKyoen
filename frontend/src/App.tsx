import { TFunction } from "i18next";
import { useMemo, useState } from "react";
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

  const hintSet = useMemo(() => new Set(hintMoves), [hintMoves]);
  const resultSet = useMemo(() => new Set(lastResult?.kyoen_points ?? []), [lastResult]);

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
        stroke="#4a4ae6"
        strokeWidth={0.04}
      />
    );
  };

  return (
    <div className="grid place-items-center px-3.5 pt-6 pb-10 max-sm:px-2.5 max-sm:pt-3 max-sm:pb-6">
      <main className="grid gap-4 w-full max-w-[980px]">
        <header className="relative bg-white/70 border-2 border-white rounded-[18px] px-5 py-[18px] backdrop-blur-sm">
          <div className="absolute top-2.5 right-3.5 flex gap-1">
            <button
              className={`px-2 py-1 text-[11px] rounded-md font-bold border border-line cursor-pointer transition-opacity ${
                i18n.language === "en" ? "bg-line text-white opacity-100" : "bg-transparent text-ink opacity-60 hover:opacity-80"
              }`}
              onClick={() => changeLanguage("en")}
            >
              EN
            </button>
            <button
              className={`px-2 py-1 text-[11px] rounded-md font-bold border border-line cursor-pointer transition-opacity ${
                i18n.language === "ja" ? "bg-line text-white opacity-100" : "bg-transparent text-ink opacity-60 hover:opacity-80"
              }`}
              onClick={() => changeLanguage("ja")}
            >
              JA
            </button>
          </div>
          <p className="m-0 font-display font-bold tracking-[0.08em] uppercase text-brand">{t("subtitle")}</p>
          <h1 className="mt-1 mb-0.5 font-display font-bold text-[clamp(1.7rem,3vw,2.5rem)]">{t("title")}</h1>
          <p className="m-0 opacity-[0.82]">{t("instruction")}</p>
        </header>

        <section className="bg-white/80 border-2 border-white rounded-2xl p-3.5">
          <div className="flex flex-wrap gap-2.5 items-center">
            <label htmlFor="size-select" className="font-display font-bold">{t("size")}</label>
            <select id="size-select" className="control-input" value="9x9" onChange={() => undefined}>
              <option value="9x9">{t("sizeAiOnly")}</option>
            </select>

            <label htmlFor="order-select" className="font-display font-bold">{t("order")}</label>
            <select id="order-select" className="control-input" value={humanSide === 1 ? "first" : "second"} onChange={(e) => void onOrderChange(e.target.value)}>
              <option value="first">{t("youFirst")}</option>
              <option value="second">{t("youSecond")}</option>
            </select>

            <button type="button" className="btn" onClick={() => setShowResult((v) => !v)} disabled={!lastResult?.kyoen_points?.length}>
              {t("result")}
            </button>

            <button type="button" className="btn" onClick={onToggleHints} disabled={done || isAiThinking}>
              {showHints ? t("hintOff") : t("hint")}
            </button>

            <button type="button" className="btn" onClick={() => void onReset()}>{t("reset")}</button>
          </div>

          <div className="mt-2.5 font-bold">{getStatusText(t, lastResult, state.turn, humanSide, isAiThinking)}</div>
          {showResult && lastResult?.kyoen_points?.length ? (
            <div className="mt-1.5 text-brand font-display font-bold">
              {t("kyoenPoints")}: {lastResult.kyoen_points.map((i) => idxToLabel(i)).join(" | ")}
            </div>
          ) : null}
          {error ? <div className="mt-1.5 text-error font-display font-bold whitespace-pre-wrap">{error}</div> : null}
        </section>

        <section className="w-full grid place-items-center" aria-label="Kyoen board">
          <svg className="block aspect-square w-full max-w-[94vw] sm:max-w-[560px]" viewBox="0 0 10 10" width="100%" height="100%">
            {/* Grid Lines */}
            {Array.from({ length: SIZE }, (_, i) => i + 1).map((pos) => (
              <g key={`lines-${pos}`}>
                {/* Horizontal */}
                <line x1={0.5} y1={pos} x2={9.5} y2={pos} stroke="#111" strokeWidth={0.03} />
                {/* Vertical */}
                <line x1={pos} y1={0.5} x2={pos} y2={9.5} stroke="#111" strokeWidth={0.03} />
              </g>
            ))}

            {/* Hint & Result Highlights / Interaction Areas */}
            {state.board.map((cell, idx) => {
              const row = Math.floor(idx / SIZE) + 1;
              const col = (idx % SIZE) + 1;
              const hint = showHints && hintSet.has(idx) && cell === 0;
              const result = showResult && resultSet.has(idx);
              const disabled = done || isAiThinking || state.turn !== humanSide || cell !== 0;

              return (
                <g key={`cell-${idx}`} transform={`translate(${col}, ${row})`}>
                  {/* Highlight (hint or result) */}
                  {(hint || result) && (
                    <rect
                      x={-0.5}
                      y={-0.5}
                      width={1}
                      height={1}
                      className={result ? "cell-bg result" : hint ? "cell-bg hint" : "cell-bg"}
                    />
                  )}

                  {/* Invisible rect for click events */}
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

                  {/* Stone */}
                  {cell === humanSide && <circle r={0.25} className="stone-you" />}
                  {cell !== 0 && cell !== humanSide && <circle r={0.25} className="stone-ai" />}
                </g>
              );
            })}

            {/* Circumcircle (Result) */}
            {showResult && renderCircumcircle()}
          </svg>
        </section>
      </main>
    </div>
  );
}
