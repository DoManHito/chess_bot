import chess
import chess.engine
from classifiers.move_classifier import MoveClassifier

STOCKFISH_PATH = "./stockfish-ubuntu-x86-64-avx2"
stockfish_engine = None

def init_stockfish():
    global stockfish_engine
    if stockfish_engine is None:
        if __import__('os').path.exists(STOCKFISH_PATH):
            try:
                stockfish_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
                print("🚀 Stockfish engine loaded successfully!")
            except Exception as e:
                print(f"❌ Failed to start Stockfish: {e}")
                import traceback
                traceback.print_exc()
                stockfish_engine = None

def get_stockfish_verdict(board: chess.Board, played_move: chess.Move) -> str:
    if stockfish_engine is None:
        return "N/A"

    try:
        move_limit = chess.engine.Limit(depth=18)
        is_white = board.turn == chess.WHITE

        analysis = stockfish_engine.analyse(board, move_limit)
        best_move = analysis.get("move")
        best_score = analysis["score"].white().score(mate_score=10000)

        if played_move == best_move:
            return "Best"

        board_after = board.copy()
        board_after.push(played_move)
        
        analysis_after = stockfish_engine.analyse(board_after, move_limit)
        played_score = analysis_after["score"].white().score(mate_score=10000)

        if is_white:
            loss = best_score - played_score
        else:
            loss = played_score - best_score

        loss = max(0, loss)

        if loss <= 15:   return "Best"
        if loss <= 40:   return "Excellent"
        if loss <= 90:   return "Good"
        if loss <= 175:  return "Inaccuracy"
        if loss <= 300:  return "Mistake"
        return "Blunder"

    except Exception as e:
        print(f"[ERROR] Stockfish analysis failed: {e}")
        return "N/A"

def test_nuances():
    print("Loading classifier for complex tactical test...")
    classifier = MoveClassifier()
    init_stockfish()
    
    if not classifier.has_weights:
        print("Error: Weights not found!")
        return
    
    def print_res(title, move_san, res, ideal_class="N/A"):
        print(f"\n--- {title} ---")
        print(f"Move: {move_san}")
        print(f"Network verdict: {res.classification} (Confidence: {res.confidence*100:.2f}%)")
        print(f"Stockfish ideal: {ideal_class}")

    multiplier = 100
    multiplier = 100 
    
    # Test 1: Passive h6 in Sicilian (Stockfish: Inaccuracy)
    board1 = chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
    move1 = board1.parse_san("h6")
    res1 = classifier.classify_move(board1.fen(), "h6", evaluation=-40, turn_num=2, turn_label="Black")    
    ideal1 = get_stockfish_verdict(board1, move1)
    print_res("Test 1: Passive h6 in Sicilian", "h6", res1, ideal1)
    
    # Test 2: Mate in 1 that Black allows (Stockfish: Blunder)
    board2 = chess.Board("r1bqkbnr/pppp1Qpp/2n5/4p3/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3")
    board2 = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 3 3")
    move2 = board2.parse_san("a6") # Ignoring mate threat on f7
    res2 = classifier.classify_move(board2.fen(), "a6", evaluation=-9900, turn_num=3, turn_label="Black")
    ideal2 = get_stockfish_verdict(board2, move2)
    print_res("Test 2: Allowing Mate in 1", "a6", res2, ideal2)
    
    # Test 3: Hanging Queen to Bishop attack (Stockfish: Blunder)
    board3 = chess.Board("rnb1kbnr/ppp2qpp/3p4/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 4")
    move3 = board3.parse_san("h6") # Instead of saving queen, just move pawn
    res3 = classifier.classify_move(board3.fen(), "h6", evaluation=-900, turn_num=4, turn_label="Black")
    ideal3 = get_stockfish_verdict(board3, move3)
    print_res("Test 3: Hanging Queen", "h6", res3, ideal3)
    
    # Test 4: Najdorf Theory 5... a6 (Stockfish: Best)
    board4 = chess.Board("rnbqkbnr/pp2pppp/3p4/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5")
    move4 = board4.parse_san("a6")
    res4 = classifier.classify_move(board4.fen(), "a6", evaluation=30, turn_num=5, turn_label="Black")
    ideal4 = get_stockfish_verdict(board4, move4)
    print_res("Test 4: Najdorf Theory 5... a6", "a6", res4, ideal4)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Neural Network vs Stockfish Comparison")
    print("="*60)
    print(f"Test 1: NN={res1.classification:10s} | Stockfish={ideal1:10s} | Match={res1.classification == ideal1}")
    print(f"Test 2: NN={res2.classification:10s} | Stockfish={ideal2:10s} | Match={res2.classification == ideal2}")
    print(f"Test 3: NN={res3.classification:10s} | Stockfish={ideal3:10s} | Match={res3.classification == ideal3}")
    print(f"Test 4: NN={res4.classification:10s} | Stockfish={ideal4:10s} | Match={res4.classification == ideal4}")
    
    matches = sum([res1.classification == ideal1, res2.classification == ideal2,
                   res3.classification == ideal3, res4.classification == ideal4])
    print(f"\nAccuracy: {matches}/4 tests match Stockfish verdict")
    
    if stockfish_engine is not None:
        stockfish_engine.quit()

if __name__ == "__main__":
    test_nuances()