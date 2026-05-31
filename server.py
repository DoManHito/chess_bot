from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chess

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

# Initialize bot brain once at server startup
print("Loading neural network and MCTS...")
classifier = MoveClassifier()
engine = MoveClassifierMCTS(classifier)

@app.get("/get_move")
def get_move(fen: str, simulations: int = 100):
    """Endpoint that your site will call."""
    try:
        board = chess.Board(fen)
        if board.is_game_over():
            return {"error": "Game over", "result": board.result()}
            
        # Search for best move
        bot_move = engine.search(board, num_simulations=simulations)
        
        return {
            "move_uci": bot_move.uci(),
            "move_san": board.san(bot_move)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)