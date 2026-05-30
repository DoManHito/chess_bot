import re
from dataclasses import dataclass, field
from typing import List, Optional
import chess
import chess.pgn


@dataclass
class ParsedMove:
    """Represents a parsed chess move with all associated metadata."""
    move: str
    san: str
    fen_before: Optional[str] = None
    fen_after: Optional[str] = None
    turn_num: Optional[int] = None
    turn_label: Optional[str] = None
    evaluation: Optional[float] = None


@dataclass
class ParsedGame:
    """Represents a parsed chess game with headers and moves."""
    headers: dict = field(default_factory=dict)
    moves: List[ParsedMove] = field(default_factory=list)
    evaluation: Optional[float] = None
    result: Optional[str] = None


class PGNParser:
    """Умный потоковый парсер, который ищет только 6% партий с оценками Stockfish."""
    
    def parse_file(self, file_path: str) -> List[ParsedGame]:
        games = []
        
        print(f"Начинаю фильтрацию и парсинг файла: {file_path}")
        print("Ищу партии со встроенным анализом Stockfish (те самые 6%)...")
        
        with open(file_path, 'r', encoding='utf-8') as pgn_file:
            checked_games = 0
            saved_games = 0
            
            while True:
                # Читаем одну партию из файла (тратит всего пару КБ памяти)
                lichess_game = chess.pgn.read_game(pgn_file)
                if lichess_game is None:
                    break  # Дошли до конца файла
                
                checked_games += 1
                
                # --- ТРЮК ФИЛЬТРАЦИИ ---
                # Заглядываем в первый ход игры. Если в нем нет упоминания '%eval',
                # значит Lichess не обсчитывал эту партию. Пропускаем её целиком!
                first_move_node = lichess_game.next()
                if first_move_node is None or "%eval" not in first_move_node.comment:
                    continue # Переходим к следующей игре
                
                # Если мы дошли сюда — ура! Это одна из тех 6% партий, которые нам нужны.
                parsed_game = ParsedGame()
                parsed_game.headers = dict(lichess_game.headers)
                parsed_game.result = parsed_game.headers.get("Result", "*")
                
                board = lichess_game.board()
                node = lichess_game
                turn_num = 1
                
                while node.variations:
                    next_node = node.variation(0)
                    move = next_node.move
                    
                    parsed_move = ParsedMove(
                        move=move.uci(),
                        san=board.san(move),
                        fen_before=board.fen(),
                        turn_num=turn_num,
                        turn_label="White" if board.turn == chess.WHITE else "Black"
                    )
                    
                    board.push(move)
                    parsed_move.fen_after = board.fen()
                    
                    # Извлекаем оценку [%eval 2.35] или [%eval #-4] (мат)
                    comment = next_node.comment
                    eval_match = re.search(r'%eval\s+([+-]?#?\d+\.?\d*)', comment)
                    
                    if eval_match:
                        eval_str = eval_match.group(1)
                        if '#' in eval_str:
                            # Если там мат (например #4 или #-2), превращаем в условную большую оценку
                            parsed_move.evaluation = 99.0 if '-' not in eval_str else -99.0
                        else:
                            parsed_move.evaluation = float(eval_str)
                    else:
                        parsed_move.evaluation = 0.0
                    
                    parsed_game.moves.append(parsed_move)
                    
                    node = next_node
                    if board.turn == chess.WHITE:
                        turn_num += 1
                
                games.append(parsed_game)
                saved_games += 1
                
                if saved_games % 100 == 0:
                    print(f"Сканировано игр: {checked_games} | Найдено и сохранено партий с ИИ: {saved_games}")
                
                # Для первого теста соберем ровно 1000 качественных ИИ-партий
                if saved_games >= 100000:
                    print(f"\nУспех! Собрано первые {saved_games} игр со Stockfish-оценками.")
                    break
                    
        return games