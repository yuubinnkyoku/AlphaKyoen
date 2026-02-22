import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { TFunction } from "i18next";
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

  return (
    <div className="page">
      <main className="shell">
        <header className="hero">
          <div className="lang-switcher">
            <button
              className={i18n.language === "en" ? "active" : ""}
              onClick={() => changeLanguage("en")}
            >
              EN
            </button>
            <button
              className={i18n.language === "ja" ? "active" : ""}
              onClick={() => changeLanguage("ja")}
            >
              JA
            </button>
          </div>
          <p className="eyebrow">{t("subtitle")}</p>
          <h1>{t("title")}</h1>
          <p className="sub">{t("instruction")}</p>
        </header>

        <section className="panel">
          <div className="controls">
            <label htmlFor="size-select">{t("size")}</label>
            <select id="size-select" value="9x9" onChange={() => undefined}>
              <option value="9x9">{t("sizeAiOnly")}</option>
            </select>

            <label htmlFor="order-select">{t("order")}</label>
            <select id="order-select" value={humanSide === 1 ? "first" : "second"} onChange={(e) => void onOrderChange(e.target.value)}>
              <option value="first">{t("youFirst")}</option>
              <option value="second">{t("youSecond")}</option>
            </select>

            <button type="button" onClick={() => setShowResult((v) => !v)} disabled={!lastResult?.kyoen_points?.length}>
              {t("result")}
            </button>

            <button type="button" onClick={onToggleHints} disabled={done || isAiThinking}>
              {showHints ? t("hintOff") : t("hint")}
            </button>

            <button type="button" onClick={() => void onReset()}>{t("reset")}</button>
          </div>

          <div className="status">{getStatusText(t, lastResult, state.turn, humanSide, isAiThinking)}</div>
          {showResult && lastResult?.kyoen_points?.length ? (
            <div className="result-text">
              {t("kyoenPoints")}: {lastResult.kyoen_points.map((i) => idxToLabel(i)).join(" | ")}
            </div>
          ) : null}
          {error ? <div className="error">{error}</div> : null}
        </section>

        <section className="board-wrap" aria-label="Kyoen board">
          <div className="board" role="grid">
            {state.board.map((cell, idx) => {
              const hint = showHints && hintSet.has(idx) && cell === 0;
              const result = showResult && resultSet.has(idx);
              const disabled = done || isAiThinking || state.turn !== humanSide || cell !== 0;
              const cls = [
                "cell",
                cell === 1 ? "stone-o" : "",
                cell === -1 ? "stone-x" : "",
                hint ? "hint" : "",
                result ? "result" : "",
              ]
                .filter(Boolean)
                .join(" ");

              return (
                <button
                  key={idx}
                  type="button"
                  role="gridcell"
                  className={cls}
                  onClick={() => void onCellClick(idx)}
                  disabled={disabled}
                  aria-label={`cell-${idx}`}
                >
                  {cell === 1 ? "O" : cell === -1 ? "X" : ""}
                </button>
              );
            })}
          </div>
        </section>
      </main>
    </div>
  );
}
