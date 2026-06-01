from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chess
import time
import math

from classifiers.move_classifier import MoveClassifier
from engine.mcts import MoveClassifierMCTS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = MoveClassifier(weights_path="models/weights_bot.pth")
engine = MoveClassifierMCTS(classifier)

@app.get("/analyze_move")
def analyze_move(fen: str, move_san: str):
    """Эндпоинт для мгновенного анализа хода игрока."""
    try:
        board = chess.Board(fen)
        classes, confidences, _ = classifier.classify_moves_batch(board, [move_san])
        
        return {
            "move_class": classes[0] if classes else "Good",
            "confidence": round(confidences[0], 2) if confidences else 1.0
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/get_move")
def get_move(fen: str, simulations: int = 100):
    """Эндпоинт для генерации и одновременного анализа хода бота."""
    try:
        board = chess.Board(fen)
        
        if board.is_game_over():
            return {
                "error": "Game over", 
                "result": board.result(),
                "is_checkmate": board.is_checkmate()
            }
            
        start_time = time.time()
        bot_move, visit_dict = engine.search(board, num_simulations=simulations)
        elapsed = time.time() - start_time

        move_san = board.san(bot_move)

        try:
            classes, confidences, _ = classifier.classify_moves_batch(board, [move_san])
            bot_move_class = classes[0] if classes else "Good"
            bot_move_conf = round(confidences[0], 2) if confidences else 1.0
        except Exception:
            bot_move_class = "Good"
            bot_move_conf = 1.0

        print(f"🤖 Бот сыграл {move_san} | Класс: {bot_move_class} (за {elapsed:.2f}с)")
        
        return {
            "move_uci": bot_move.uci(),
            "move_san": move_san,
            "bot_move_class": bot_move_class,
            "bot_move_confidence": bot_move_conf
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)