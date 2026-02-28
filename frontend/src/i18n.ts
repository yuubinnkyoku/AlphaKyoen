import i18n from "i18next";
import LanguageDetector from "i18next-browser-languagedetector";
import { initReactI18next } from "react-i18next";

const resources = {
  en: {
    translation: {
      size: "Size",
      sizeAiOnly: "9x9 (AI only)",
      order: "Order",
      youFirst: "You first",
      youSecond: "You second",
      result: "Result",
      hint: "Hint",
      hintOff: "Hint Off",
      reset: "Reset",
      thinking: "AI thinking...",
      yourTurn: "Your turn",
      aiTurn: "AI turn",
      you: "You",
      ai: "AI",
      winAi: "You created Kyoen. AI wins.",
      winYou: "AI created Kyoen. You win!",
      kyoenPoints: "Kyoen points",
      failedFetchHints: "Failed to fetch hints",
      unexpectedError: "Unexpected error",
      darkMode: "Dark mode",
      lightMode: "Light mode",
    },
  },
  ja: {
    translation: {
      size: "サイズ",
      sizeAiOnly: "9x9 (AIのみ)",
      order: "手番",
      youFirst: "あなた先手",
      youSecond: "あなた後手",
      result: "結果",
      hint: "ヒント",
      hintOff: "ヒント消去",
      reset: "リセット",
      thinking: "AIが考えています...",
      yourTurn: "あなたの番",
      aiTurn: "AIの番",
      you: "あなた",
      ai: "AI",
      winAi: "あなたが共円を作りました。AIの勝ちです。",
      winYou: "AIが共円を作りました。あなたの勝ちです。",
      kyoenPoints: "共円の点",
      failedFetchHints: "ヒントの取得に失敗しました",
      unexpectedError: "予期せぬエラーが発生しました",
      darkMode: "ダークモード",
      lightMode: "ライトモード",
    },
  },
};

i18n
  .use(initReactI18next)
  .use(LanguageDetector)
  .init({
    resources,
    fallbackLng: "en",
    interpolation: {
      escapeValue: false,
    },
  });

export default i18n;
