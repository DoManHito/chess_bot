import chess
from classifiers.move_classifier import MoveClassifier

def test_nuances():
    print("Loading classifier for complex tactical test...")
    classifier = MoveClassifier()
    
    if not classifier.has_weights:
        print("Error: Weights not found!")
        return

    def print_res(title, move_san, res):
        print(f"\n--- {title} ---")
        print(f"Move: {move_san}")
        print(f"Network verdict: {res.classification} (Confidence: {res.confidence*100:.2f}%)")

    # TEST 1: Positional inaccuracy (Inaccuracy) in opening
    # Sicilian Defense. Instead of developing knights or d6, black plays passive h6.
    # Engines typically consider this a tempo loss (Inaccuracy).
    # =========================================================================
    board1 = chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
    res1, _ = classifier.classify_move(board1.fen(), "h6", evaluation=-0.4, turn_num=2, turn_label="Black")
    print_res("Test 1: Passive h6 in Sicilian (Inaccuracy)", "h6", res1)

    # TEST 2: Blunder opening mistake — Damiano Defense
    # 1. e4 e5 2. Nf3 f6? Move f6 severely weakens king, this is a known blunder.
    # =========================================================================
    board2 = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
    res2, _ = classifier.classify_move(board2.fen(), "f6", evaluation=-1.5, turn_num=2, turn_label="Black")
    print_res("Test 2: Damiano Defense 2... f6 (Mistake)", "f6", res2)

    # TEST 3: Deep tactical trap (Hidden blunder / Blunder)
    # Position from Cambridge Springs variation. Black pawn captures on c4 (dxc4).
    # Move appears logical (capture), but tactically white forces material win.
    # =========================================================================
    board3 = chess.Board("r1bqkb1r/pppn1ppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 4 5")
    # Simulate dxc4 move for black (though currently white's turn per FEN, switch context for test)
    board3.turn = chess.BLACK
    res3, _ = classifier.classify_move(board3.fen(), "dxc4", evaluation=-2.1, turn_num=5, turn_label="Black")
    print_res("Test 3: Capture dxc4 leading to tactical loss (Blunder)", "dxc4", res3)

    # TEST 4: High-theoretical strong move (Best / Excellent)
    # Sicilian Defense, Najdorf Variation (5... a6). Complex positional move.
    # =========================================================================
    board4 = chess.Board("rnbqkbnr/pp2pppp/3p4/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5")
    res4, _ = classifier.classify_move(board4.fen(), "a6", evaluation=0.3, turn_num=5, turn_label="Black")
    print_res("Test 4: Najdorf Theory 5... a6 (Best / Excellent)", "a6", res4)

if __name__ == "__main__":
    test_nuances()