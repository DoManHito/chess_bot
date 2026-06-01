import os
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from models.chess_nets import ChessCoreNet, MoveClassifierNet
from classifiers.self_play_dataset import ChessSelfPlayDataset
from classifiers.classification_config import CLASS_NAMES

def clean_old_self_play_games(db_path="chess_bot.db", keep_last_n_games=100):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='self_play_moves'")
    if not cursor.fetchone():
        conn.close()
        return

    cursor.execute("SELECT DISTINCT game_id FROM self_play_moves ORDER BY game_id DESC")
    games = [row[0] for row in cursor.fetchall()]
    
    if len(games) > keep_last_n_games:
        games_to_delete = games[keep_last_n_games:]
        print(f"🧹 Обнаружено {len(games)} партий в истории. Оставляем последние {keep_last_n_games}.")
        print(f"🗑️ Удаляем {len(games_to_delete)} старых партий (Game IDs: {min(games_to_delete)} - {max(games_to_delete)})...")
        
        placeholders = ",".join("?" for _ in games_to_delete)
        cursor.execute(f"DELETE FROM self_play_moves WHERE game_id IN ({placeholders})", games_to_delete)
        
        conn.commit()
        print("Очистка успешно завершена!")
    else:
        print(f"📦 Размер базы под управлением: {len(games)}/{keep_last_n_games} партий. Очистка не требуется.")
        
    conn.close()


def train_rl_iteration(epochs=10, batch_size=64, lr=0.001, alpha=0.5, keep_last_n_games=100):
    db_path = "chess_bot.db"
    
    clean_old_self_play_games(db_path=db_path, keep_last_n_games=keep_last_n_games)
    print("-" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство для обучения: {device}")

    dataset = ChessSelfPlayDataset(db_path=db_path)
    if len(dataset) < 100:
        print(f"Слишком мало данных для обучения ({len(dataset)} состояний)! Сыграй больше партий через self_play.py")
        return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    core = ChessCoreNet(in_channels=25)
    model = MoveClassifierNet(core_net=core, num_classes=len(CLASS_NAMES))
    
    weights_path = "models/weights_classifier.pth"
    if os.path.exists(weights_path):
        print("Загрузка существующих весов модели...")
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    
    model.to(device)

    criterion_class = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print("Старт обучения...")
    for epoch in range(epochs):
        model.train()
        total_class_loss = 0
        total_value_loss = 0
        total_loss = 0

        for inputs, class_targets, value_targets in train_loader:
            inputs = inputs.to(device)
            class_targets = class_targets.to(device)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()

            class_logits, value_preds = model(inputs)

            loss_class = criterion_class(class_logits, class_targets)
            loss_value = criterion_value(value_preds, value_targets)
            loss = loss_class + alpha * loss_value

            loss.backward()
            optimizer.step()

            total_class_loss += loss_class.item()
            total_value_loss += loss_value.item()
            total_loss += loss.item()

        model.eval()
        val_class_loss = 0
        val_value_loss = 0
        with torch.no_grad():
            for inputs, class_targets, value_targets in val_loader:
                inputs = inputs.to(device)
                class_targets = class_targets.to(device)
                value_targets = value_targets.to(device)

                class_logits, value_preds = model(inputs)
                val_class_loss += criterion_class(class_logits, class_targets).item()
                val_value_loss += criterion_value(value_preds, value_targets).item()

        print(f"Эпоха {epoch+1}/{epochs} | "
              f"Train Loss: {total_loss/len(train_loader):.4f} "
              f"(Class: {total_class_loss/len(train_loader):.4f}, Value: {total_value_loss/len(train_loader):.4f}) | "
              f"Val Class Loss: {val_class_loss/len(val_loader):.4f}, Val Value MSE: {val_value_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), weights_path)
    print(f"Новые веса успешно сохранены в {weights_path}!")

if __name__ == "__main__":
    train_rl_iteration(epochs=8, batch_size=64, lr=0.0005, keep_last_n_games=50)