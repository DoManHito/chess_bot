from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chess
# Импортируем вашего обученного бота
from classifiers.move_classifier import MoveClassifier
from engine.mcts import MoveClassifierMCTS

app = FastAPI()

# Разрешаем сайту обращаться к нашему серверу
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # В продакшене укажите адрес вашего сайта
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализируем мозг бота один раз при старте сервера
print("Загрузка нейросети и MCTS...")
classifier = MoveClassifier()
engine = MoveClassifierMCTS(classifier, cpuct=1.5)

@app.get("/get_move")
def get_move(fen: str, simulations: int = 80):
    """Эндпоинт, который будет вызывать ваш сайт."""
    try:
        board = chess.Board(fen)
        if board.is_game_over():
            return {"error": "Game over", "result": board.result()}
            
        # Запускаем поиск хода
        bot_move = engine.search(board, num_simulations=simulations)
        
        return {
            "move_uci": bot_move.uci(), # Формат вроде "e2e4" для фронтенда
            "move_san": board.san(bot_move)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)