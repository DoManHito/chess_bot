from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chess
import chess.engine
import time
import math
import os
import uvicorn
import torch

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

STOCKFISH_PATH = "./stockfish-ubuntu-x86-64-avx2"
if not os.path.exists(STOCKFISH_PATH):
    print(f"⚠️ Stockfish is not there: {STOCKFISH_PATH}!")

print("=== Server Startup Timing ===")
print(f"PyTorch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"PyTorch version: {torch.__version__}")

start_time = time.time()
print("Loading classifier model...")
# Enable GPU if available for faster inference
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print("⚡ GPU acceleration enabled!")
else:
    print("⚠️ Running on CPU - consider installing CUDA-enabled PyTorch")
classifier = MoveClassifier(weights_path="models/weights_bot.pth", device=device)
classifier_load_time = time.time() - start_time
print(f"Classifier loaded in {classifier_load_time:.2f} seconds")

start_time = time.time()
print("Initializing MCTS engine...")
# Optimized MCTS: fewer simulations + move prioritization
engine = MoveClassifierMCTS(
    classifier,
    cpuct=2.0,
    max_simulations=100,
    top_moves_ratio=0.3 
)
engine_init_time = time.time() - start_time
print(f"MCTS engine initialized in {engine_init_time:.2f} seconds")

total_startup_time = time.time() - start_time
print(f"=== Total startup time: {total_startup_time:.2f} seconds ===")

def get_stockfish_verdict(board_before: chess.Board, move: chess.Move) -> str:
    if not os.path.exists(STOCKFISH_PATH):
        return "N/A"
        
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as sf:

            info_before = sf.analyse(board_before, chess.engine.Limit(time=0.05))
            score_before = info_before["score"].pov(board_before.turn)
            
            cp_before = score_before.score(mate_score=2000) / 100.0
            
            board_after = board_before.copy()
            board_after.push(move)
            
            info_after = sf.analyse(board_after, chess.engine.Limit(time=0.05))
            score_after = info_after["score"].pov(board_before.turn)
            cp_after = score_after.score(mate_score=2000) / 100.0
            
            loss = cp_before - cp_after
            
            if loss <= 0.01: return "Best"
            elif loss <= 0.2: return "Excellent"
            elif loss <= 0.5: return "Good"
            elif loss <= 1.0: return "Inaccuracy"
            elif loss <= 2.0: return "Mistake"
            else: return "Blunder"
    except Exception as e:
        print(f"Stockfish Error: {e}")
        return "Error"
    
@app.get("/analyze_move")
def analyze_move(fen: str, move_san: str):
    try:
        board = chess.Board(fen)
        move = board.parse_san(move_san)

        classes, _, _ = classifier.classify_moves_batch(board, [move_san])
        nn_class = classes[0] if classes else "Good"

        ideal_class = get_stockfish_verdict(board, move)
        
        return {
            "nn_class": nn_class,
            "ideal_class": ideal_class
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/get_move")
def get_move(fen: str, simulations: int = 20):
    import time
    try:
        start_time = time.time()
        board = chess.Board(fen)
        
        if board.is_game_over():
            return {"error": "Game over", "result": board.result()}
        
        # MCTS search
        search_start = time.time()
        bot_move, _ = engine.search(board, num_simulations=simulations)
        search_time = time.time() - search_start
        print(f"[DEBUG] /get_move: MCTS search took {search_time:.2f}s")
        
        move_san = board.san(bot_move)

        # Classifier
        classify_start = time.time()
        try:
            classes, _, _ = classifier.classify_moves_batch(board, [move_san])
            bot_nn_class = classes[0] if classes else "Good"
        except Exception:
            bot_nn_class = "Good"
        classify_time = time.time() - classify_start
        print(f"[DEBUG] /get_move: Classifier took {classify_time:.2f}s")

        # Stockfish
        stockfish_start = time.time()
        bot_ideal_class = get_stockfish_verdict(board, bot_move)
        stockfish_time = time.time() - stockfish_start
        print(f"[DEBUG] /get_move: Stockfish took {stockfish_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"[DEBUG] /get_move: Total time {total_time:.2f}s")
        
        return {
            "move_uci": bot_move.uci(),
            "move_san": move_san,
            "bot_nn_class": bot_nn_class,
            "bot_ideal_class": bot_ideal_class
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)