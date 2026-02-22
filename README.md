# AlphaKyoen "Tsubura-円"

9x9共円パズルAI

- プリ計算
- Bitboard
- JAX

## 共円ゲームとは？
### 基本ルール
1. **順番に石を置く**：プレイヤーは順番に、盤の上の格子点（線の交点）に石を1個ずつ置いていく。すでに石がある場所には置けない。
2. **「共円」を作ってはいけない**：石を置いたとき、盤上の石のうち **「4つの石が同一円周上に並ぶ状態（＝共円）」** を作ってはいけない。
3. **直線もアウト**：数学上「直線は半径が無限大の円」とみなされるため、**「同一直線上に4つの石が並ぶ」のも共円と同じくアウト**。

### 指摘と脱落（勝敗）
* **指摘による脱落**：あるプレイヤーが石を置いた結果、「同一円周上に4つの石」が発生してしまった場合、他のプレイヤーは次の人が石を置くまでの間であればいつでもそれを指摘できる。「共円！」と宣言し、どの4点が円周上にあるかを説明します。正しければ、最後に石を置いたプレイヤーは **負け（脱落）** となる（最後に置かれた石は盤から取り除かれる）。
直後に相手が気づかなければセーフだが、相手はAI。**容赦なくプリ計算テーブルを使って指摘してくる**。

詳細: http://nadamath2012.web.fc2.com/program/kyouen1.html

## 学習

```cmd
uv sync
```

```cmd
uv run src/main.py
```

## Web UI (React + FastAPI)

- GitHub Pages 配信の React フロントエンド
- Hugging Face Spaces 配信の FastAPI バックエンド
- 機能: 9x9 AI対戦、Size UI (9x9固定)、Resultボタン、Hintボタン

### 1) FastAPI バックエンド起動

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

API:

- `POST /api/move` : 人間の手を適用
- `POST /api/ai_move` : AIの手を計算して適用
- `POST /api/hint` : 次に置くと共円になる手(黄色表示用)
- `GET /api/health` : ヘルスチェック

### 2) React フロントエンド起動

```bash
cd frontend
npm install
npm run dev
```

環境変数例は `frontend/.env.example` を参照。

### 3) GitHub Pages

- ワークフロー: `.github/workflows/deploy-pages.yml`
- `master` / `main` ブランチ push で `frontend/dist` を Pages にデプロイ
- `VITE_API_BASE_URL` は GitHub Repository Variables で設定
- 初回のみ GitHub の `Settings > Pages` で Build and deployment を `GitHub Actions` にしておく

Repository Variables 例:

- `VITE_API_BASE_URL=https://<your-space-name>.hf.space`

### 4) Hugging Face Spaces

- ルートの `Dockerfile` を使用して FastAPI を 7860 番ポートで起動
- Space 側でこのリポジトリを参照してデプロイ可能

推奨設定:

- Space SDK: `Docker`
- リポジトリ: このリポジトリを接続
- 公開URL: `https://<your-space-name>.hf.space`

### 5) API スモークテスト

```bash
uv run python scripts/smoke_api.py
```

AI の一手まで確認する場合:

```bash
uv run python scripts/smoke_api.py --with-ai
```

