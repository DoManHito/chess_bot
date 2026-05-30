import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import numpy as np
import random
from collections import Counter

from models.chess_nets import ChessCoreNet, MoveClassifierNet
from classifiers.move_classifier import MoveClassifier
from classifiers.classification_config import CLASS_NAMES, THRESHOLDS, NEGATIVE_THRESHOLD

class ChessDataset(Dataset):
    """Оптимизированный датасет: загружает fen_before и готовую fen_after напрямую из БД."""
    def __init__(self, db_path="chess_bot.db", game_ids=None):
        self.samples = [] 
        # Убираем жесткую инициализацию отсюда, чтобы процессы не конфликтовали
        self.classifier_utils = None 
        self.db_path = db_path
        
        # Превращаем в set для моментального поиска по O(1) при итерации по строкам
        self.game_ids = set(game_ids) if game_ids is not None else None
        
        self._load_and_label_data_lightweight(db_path)

    def _load_and_label_data_lightweight(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"Сканируем базу данных и рассчитываем классы (Фильтр по играм: {self.game_ids is not None})...")
        cursor.execute("""
            SELECT game_id, fen_before, fen_after, evaluation 
            FROM moves 
            ORDER BY game_id, id
        """)
        
        prev_game_id = None
        prev_eval = 0.3
        
        while True:
            row = cursor.fetchone()
            if row is None:
                break
                
            game_id, fen_before, fen_after, evaluation = row
            
            if not fen_after:
                continue

            try:
                current_id = int(game_id)
            except (ValueError, TypeError):
                continue

            # Фильтр по принадлежности к train/val подвыборке
            if self.game_ids is not None and current_id not in self.game_ids:
                continue
                
            # Сброс контекста оценки при смене партии
            if prev_game_id != current_id:
                prev_game_id = current_id
                prev_eval = 0.3 if evaluation is None else evaluation
            
            current_eval = evaluation if evaluation is not None else prev_eval
            
            # Определяем чья очередь ходить по FEN перед ходом
            is_white = " w " in fen_before
            
            # Считаем потерю оценки (разницу)
            if is_white:
                delta = prev_eval - current_eval
            else:
                delta = current_eval - prev_eval

            # Сглаживание дельты при сильном перевесе одной из сторон
            if abs(prev_eval) > 3.0:
                delta = delta * 0.3
                
            # --- ИСПРАВЛЕННЫЙ И БЕЗОПАСНЫЙ МАППИНГ ДЕЛЬТЫ НА КЛАССЫ ---
            from classifiers.classification_config import THRESHOLDS
            
            # Если ход сохранил или улучшил оценку (delta <= макс. значению для Best, например 0.02)
            if delta <= THRESHOLDS[0].max_evaluation:
                class_idx = 0  # Идеальный ход ("Best")
            else:
                class_idx = 5  # По умолчанию "Blunder" (если дельта огромная и не попала в диапазоны ниже)
                for idx, t in enumerate(THRESHOLDS):
                    # Используем строгое сравнение верхнего порога <, чтобы исключить наложения
                    if t.min_evaluation <= delta < t.max_evaluation:
                        class_idx = idx
                        break
            
            # Добавляем кортеж данных в список семплов
            self.samples.append((fen_before, fen_after, class_idx))
            
            # ВАЖНО: Запоминаем оценку. Теперь это гарантированно выполняется для каждого хода!
            prev_eval = current_eval

        conn.close()
        print(f"Успешно загружено ходов в датасет: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen_before, fen_after, class_idx = self.samples[idx]
        board_before = chess.Board(fen_before)
        board_after = chess.Board(fen_after)
            
        tensor = MoveClassifier._board_to_tensor_static(board_before, board_after)
        
        if tensor.ndim == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
            
        return tensor, class_idx


def get_all_game_ids(db_path="chess_bot.db"):
    """Вспомогательная функция для быстрого сбора уникальных ID партий из БД."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT game_id FROM moves WHERE game_id IS NOT NULL")
    
    # Приводим строго к числу int
    game_ids = []
    for row in cursor.fetchall():
        try:
            game_ids.append(int(row[0]))
        except (ValueError, TypeError):
            continue
            
    conn.close()
    return game_ids


if __name__ == "__main__":
    db_path = "chess_bot.db"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Загружаем и честно делим уникальные ID партий (80% на 20%)
    all_games = get_all_game_ids(db_path)
    print(f"Всего уникальных партий в базе данных: {len(all_games)}")
    
    random.seed(42) # Фиксируем seed для воспроизводимости разбиения
    random.shuffle(all_games)
    
    split_idx = int(0.8 * len(all_games))
    train_game_ids = all_games[:split_idx]
    val_game_ids = all_games[split_idx:]
    
    print(f"Выделено партий для обучения (train): {len(train_game_ids)}")
    print(f"Выделено партий для валидации (val): {len(val_game_ids)}")
    
    # 2. Инициализируем ДВА независимых объекта датасета
    print("\nИнициализация тренировочного датасета...")
    train_dataset = ChessDataset(db_path=db_path, game_ids=train_game_ids)
    
    print("\nИнициализация валидационного датасета...")
    val_dataset = ChessDataset(db_path=db_path, game_ids=val_game_ids)
    
    # 3. Оборачиваем в DataLoader (Разные оптимальные размеры батчей)
    num_workers = 4 

    train_loader = DataLoader(
        train_dataset, 
        batch_size=2048, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2048, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    # 4. Рассчитываем веса классов СТРОГО по тренировочной выборке
    train_targets = [sample[-1] for sample in train_dataset.samples]
    target_counts = Counter(train_targets)
    total_train_samples = len(train_targets)
    
    # ИСПРАВЛЕНО: Сглаживание через квадратный корень (Square Root Smoothing)
    # Это не даст редкому классу получить деструктивно огромный вес
    class_weights = []
    for i in range(len(CLASS_NAMES)):
        count = target_counts.get(i, 1)
        weight = (total_train_samples / count) ** 0.5
        class_weights.append(weight)
        
    # Нормализуем полученные веса, чтобы их среднее значение было равно 1.0
    mean_weight = sum(class_weights) / len(class_weights)
    class_weights = [w / mean_weight for w in class_weights]
        
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Рассчитанные (СГЛАЖЕННЫЕ и НОРМАЛИЗОВАННЫЕ) веса для классов: {class_weights}")
    
    # 5. Собираем модель классификатора
    core = ChessCoreNet(in_channels=25)
    model = MoveClassifierNet(core_net=core, num_classes=len(CLASS_NAMES))
    model = model.to(device)
    
    weights_path = "models/weights_classifier.pth"
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Успешно загружены существующие веса из {weights_path}.")
    except FileNotFoundError:
        print("Веса не найдены. Начинаем обучение модели с нуля.")
    
    # 6. Настройка функции потерь и оптимизатора с регуляризацией
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4) 

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)
    
    # 7. Основной цикл совмещенного обучения и валидации
    epochs = 20  
    best_val_loss = float('inf')
    
    print(f"\nЗапуск обучения классификатора на устройстве: {device}")
    for epoch in range(epochs):
        
        # --- ФАЗА ОБУЧЕНИЯ (Train) ---
        model.train() 
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
        epoch_train_loss = total_train_loss / len(train_loader)
        epoch_train_acc = 100.0 * train_correct / train_total
        print(f"Эпоха {epoch+1}/{epochs} | Лосс обучения: {epoch_train_loss:.4f} | Точность обучения: {epoch_train_acc:.2f}%")
        
        # --- ФАЗА ВАЛИДАЦИИ (Validation) ---
        model.eval() 
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): 
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_targets)
                
                total_val_loss += loss.item()
                _, val_predicted = val_outputs.max(1)
                val_total += val_targets.size(0)
                val_correct += val_predicted.eq(val_targets).sum().item()
        
        epoch_val_loss = total_val_loss / len(val_loader)
        epoch_val_acc = 100.0 * val_correct / val_total
        print(f"--> Валидация | Лосс: {epoch_val_loss:.4f} | ЧЕСТНАЯ Точность: {epoch_val_acc:.2f}%")

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Текущий шаг обучения (LR): {current_lr:.6f}")
        
        # --- СОХРАНЕНИЕ НАИЛУЧШИХ РЕЗУЛЬТАТОВ (Checkpoint) ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), weights_path)
            print(f"Веса сохранены! Лосс валидации снизился до наилучшего: {best_val_loss:.4f}")
        else:
            print(f"Веса не перезаписаны. Лучший лосс по-прежнему: {best_val_loss:.4f}")
            
        print("-" * 65)