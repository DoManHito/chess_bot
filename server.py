from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
import chess
import chess.engine
import time
import os
import json
import uvicorn
import torch
import os
from fastapi import FastAPI, HTTPException

from classifiers.move_classifier import MoveClassifier
from engine.unified_mcts import UnifiedMCTS

app = FastAPI(title="Chess Bot API", version="2.0")

STOCKFISH_PATH = "./stockfish-ubuntu-x86-64-avx2"
stockfish_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global stockfish_engine
    if os.path.exists(STOCKFISH_PATH):
        try:
            stockfish_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            print("🚀 Stockfish engine loaded successfully from root!")
        except Exception as e:
            print(f"❌ Failed to start Stockfish: {e}")
    else:
        print(f"⚠️ Stockfish not found at {STOCKFISH_PATH}!")
        
    yield
    
    if stockfish_engine:
        stockfish_engine.quit()
        print("🛑 Stockfish engine closed.")

app = FastAPI(title="Chess Bot API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("=== Server Startup Timing ===")
print(f"PyTorch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"PyTorch version: {torch.__version__}")

start_time = time.time()
print("Loading unified classifier model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print("⚡ GPU acceleration enabled!")
else:
    print("⚠️ Running on CPU - consider installing CUDA-enabled PyTorch")

classifier = MoveClassifier(
    weights_path="models/weights_bot copy.pth", 
    device=device,
)
classifier_load_time = time.time() - start_time
print(f"Unified classifier loaded in {classifier_load_time:.2f} seconds")

start_time = time.time()
print("Initializing Unified MCTS engine...")
engine = UnifiedMCTS(
    unified_model=classifier,
    cpuct=2.0,
    max_simulations=100,
    top_moves_ratio=0.3,
    policy_output_dim=64
)
engine_init_time = time.time() - start_time
print(f"Unified MCTS engine initialized in {engine_init_time:.2f} seconds")

total_startup_time = time.time() - start_time
print(f"=== Total startup time: {total_startup_time:.2f} seconds ===")


def get_stockfish_verdict(board: chess.Board, played_move: chess.Move, move_san: str) -> str:
    if stockfish_engine is None:
        return "N/A"
    try:
        stockfish_engine.configure({"Clear Hash": True})
        move_limit = chess.engine.Limit(depth=18)
        
        current_player = board.turn
        
        analysis = stockfish_engine.analyse(board, move_limit)
        best_move = analysis.get("move")
        
        if played_move == best_move:
            return "Best"
            
        pov_best_score = analysis["score"].pov(current_player)
        
        board_after = board.copy()
        board_after.push(played_move)
        
        if board_after.is_checkmate():
            return "Best"
            
        stockfish_engine.configure({"Clear Hash": True})
        analysis_after = stockfish_engine.analyse(board_after, move_limit)
        
        pov_played_score = analysis_after["score"].pov(current_player)
        
        if pov_best_score.is_mate() and pov_best_score.mate() > 0:
            if not pov_played_score.is_mate() or pov_played_score.mate() <= 0:
                return "Blunder"
            mate_diff = pov_played_score.mate() - pov_best_score.mate()
            if mate_diff > 3: 
                return "Mistake"
            elif mate_diff > 1:
                return "Inaccuracy"
            return "Good"

        if pov_best_score.is_mate() and pov_best_score.mate() < 0:
            if pov_played_score.is_mate() and pov_played_score.mate() < 0:
                if pov_played_score.mate() > pov_best_score.mate(): 
                    return "Blunder"
                return "Good"
            return "Best"

        if pov_played_score.is_mate() and pov_played_score.col().mate() < 0:
            return "Blunder"

        best_cp = pov_best_score.score(mate_score=10000)
        played_cp = pov_played_score.score(mate_score=10000)
        
        if best_cp is None or played_cp is None:
            return "N/A"
            
        loss = (best_cp - played_cp) / 100.0
        if loss < 0:
            loss = 0.0
        
        print(f"[DEBUG] Stockfish verdict: player={current_player}, best_cp={best_cp}, played_cp={played_cp}, loss={loss:.2f}")

        if loss <= 0.02:
            return "Best"
        elif loss <= 0.07:
            return "Excellent"
        elif loss <= 0.15:
            return "Good"
        elif loss <= 0.30:
            return "Inaccuracy"
        elif loss <= 0.55:
            return "Mistake"
        else:
            return "Blunder"
            
    except Exception as e:
        print(f"[ERROR] Stockfish analysis failed: {e}")
        return "N/A"


@app.get("/")
def root():
    return {
        "name": "Chess Bot API v2.0",
        "description": "Unified chess analysis system with lookahead capability",
        "endpoints": [
            "/analyze_move - Analyze a move (requires FEN and SAN)",
            "/get_move - Get bot's best move (requires FEN, optional simulations count)",
            "/get_policy - Get move policy from unified model (requires FEN)"
        ]
    }


@app.get("/analyze_move")
def analyze_move(fen: str, move_san: str):
    try:
        board = chess.Board(fen)
        
        try:
            move = board.parse_san(move_san)
        except ValueError:
            move = board.parse_uci(move_san)

        clean_move_san = board.san(move)

        try:
            nn_result = classifier.classify_move(fen, clean_move_san)
            nn_class = nn_result.classification
        except Exception as e:
            print(f"[WARNING] NN Classification failed: {e}")
            nn_class = "N/A"

        ideal_class = get_stockfish_verdict(board, move, clean_move_san)
        
        return {
            "nn_class": nn_class,
            "ideal_class": ideal_class,
            "move_san": clean_move_san,
            "move_uci": move.uci()
        }
    except Exception as e:
        print(f"[CRITICAL ERROR] /analyze_move completely crashed: {e}")
        return {
            "error": str(e),
            "nn_class": "Good",
            "ideal_class": "Good",
            "move_san": move_san,
            "move_uci": ""
        }


@app.get("/get_move")
def get_move(fen: str, simulations: int = 100):    
    try:
        start_time = time.time()
        board = chess.Board(fen)

        if board.fullmove_number == 1 and hasattr(engine, 'board_cache'):
            engine.board_cache.clear()
        
        if board.is_game_over():
            return {"error": "Game over", "result": board.result()}
        
        if board.fullmove_number <= 8:
            current_temperature = 1.0  
        else:
            current_temperature = 0.0
        
        search_start = time.time()
        bot_move, visit_dict = engine.search(board, num_simulations=simulations, temperature=current_temperature)

        search_time = time.time() - search_start
        print(f"[DEBUG] /get_move: MCTS search took {search_time:.2f}s")
        
        if bot_move is None:
            return {"error": "No legal moves", "fen": fen}
        
        move_san = board.san(bot_move)

        classify_start = time.time()
        try:
            nn_result = classifier.classify_move(fen, move_san)
            bot_nn_class = nn_result.classification
        except Exception:
            bot_nn_class = "Good"
        classify_time = time.time() - classify_start
        print(f"[DEBUG] /get_move: Classifier took {classify_time:.2f}s")

        stockfish_start = time.time()
        bot_ideal_class = get_stockfish_verdict(board, bot_move, move_san)
        stockfish_time = time.time() - stockfish_start
        print(f"[DEBUG] /get_move: Stockfish took {stockfish_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"[DEBUG] /get_move: Total time {total_time:.2f}s")
        
        return {
            "move_uci": bot_move.uci(),
            "move_san": move_san,
            "bot_nn_class": bot_nn_class,
            "bot_ideal_class": bot_ideal_class,
            "visit_counts": visit_dict,
            "total_time": total_time
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/get_policy")
def get_policy(fen: str, top_k: int = 64):
    try:
        board = chess.Board(fen)
        if board.is_game_over():
            return {"error": "Game over", "result": board.result()}
        
        policy = classifier.get_policy(board, top_k=top_k)
        
        return {
            "fen": fen,
            "policy": policy,
            "total_moves": len(policy)
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
