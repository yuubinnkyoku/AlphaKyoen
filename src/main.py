import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from game import Board
from model import SmallResNet

# 設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100         # ループ回数
GAMES_PER_EPOCH = 50 # 1回の学習のために自己対戦する回数
BATCH_SIZE = 64
LR = 0.001

def self_play(net):
    """1局プレイしてデータを生成する"""
    board = Board()
    history = []
    net.eval()
    
    while True:
        # 盤面特徴量
        feature = board.get_feature() # (3, 9, 9)
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # 推論 (MCTSなしの簡易版: Policyに従って確率的に打つ)
        with torch.no_grad():
            log_pi, _ = net(feature_tensor)
            pi = torch.exp(log_pi).cpu().numpy()[0] # (81,)
            
        # 合法手のみにマスク
        valid_moves = board.valid_moves()
        if len(valid_moves) == 0:
            break
            
        # 合法手の確率を取り出し、正規化
        pi_valid = pi[valid_moves]
        pi_sum = pi_valid.sum()
        if pi_sum > 0:
            pi_valid /= pi_sum
        else:
            pi_valid = np.ones_like(pi_valid) / len(pi_valid)
            
        # 着手決定 (サンプリング)
        action_idx = np.random.choice(len(valid_moves), p=pi_valid)
        action = valid_moves[action_idx]
        
        # 履歴保存 (特徴量, 打った手, 誰の手番か)
        history.append([feature, action, board.turn])
        
        # 実行
        _, reward, done = board.step(action)
        
        if done:
            # 最終的な報酬 (勝った側=+1, 負けた側=-1)
            # historyには「その時点の手番プレイヤー」が入っているので、
            # 勝者と同じturnなら+1, 違うなら-1
            final_winner = board.turn * -1 if reward > 0 else 0 # 勝った人のID
            
            data = []
            for feat, act, turn in history:
                v = 1.0 if turn == final_winner else -1.0
                if final_winner == 0: v = 0.0 # 引き分け
                
                # Policyの正解データ: one-hotベクトル
                pi_target = np.zeros(81, dtype=np.float32)
                pi_target[act] = 1.0
                
                data.append((feat, pi_target, v))
            return data

def train():
    print(f"Using Device: {DEVICE}")
    net = SmallResNet().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        # 1. 自己対戦 (データ収集)
        dataset = []
        print(f"Epoch {epoch+1}/{EPOCHS}: Self-playing...")
        for _ in tqdm(range(GAMES_PER_EPOCH)):
            game_data = self_play(net)
            dataset.extend(game_data)
            
        # 2. 学習 (Update)
        print(f"Training on {len(dataset)} samples...")
        net.train()
        
        # バッチ学習
        np.random.shuffle(dataset)
        
        total_loss = 0
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[i:i+BATCH_SIZE]
            
            feats = torch.tensor(np.array([d[0] for d in batch]), dtype=torch.float32).to(DEVICE)
            pis = torch.tensor(np.array([d[1] for d in batch]), dtype=torch.float32).to(DEVICE)
            vs = torch.tensor(np.array([d[2] for d in batch]), dtype=torch.float32).unsqueeze(1).to(DEVICE)
            
            optimizer.zero_grad()
            out_pi, out_v = net(feats)
            
            # Loss: Policy(CrossEntropy) + Value(MSE)
            # out_piはlog_softmax済みなので KLDivLoss や NLLLoss を使う
            loss_pi = -torch.sum(pis * out_pi) / len(batch)
            loss_v = F.mse_loss(out_v, vs)
            
            loss = loss_pi + loss_v
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Loss: {total_loss / (len(dataset)/BATCH_SIZE):.4f}")
        
        # モデル保存
        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), f"model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()