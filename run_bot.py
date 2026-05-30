import chess
from classifiers.move_classifier import MoveClassifier
from engine.mcts import MoveClassifierMCTS

def play_against_bot():
    print("Loading bot brain (Neural network + MCTS)...")
    classifier = MoveClassifier()
    # MCTS engine initialization
    engine = MoveClassifierMCTS(classifier, cpuct=1.5)
    
    board = chess.Board()
    print("\nGame started! You play as White. Enter moves in SAN format (e.g., e4, Nf3, d6).")
    
    while not board.is_game_over():
        print("\n" + "="*30)
        print(board)
        print("="*30)
        
        # Human move
        if board.turn == chess.WHITE:
            move_ok = False
            while not move_ok:
                try:
                    human_move_str = input("\nYour move: ")
                    move = board.parse_san(human_move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        move_ok = True
                    else:
                        print("Illegal move! Try again.")
                except ValueError:
                    print("Invalid move format. Use SAN (e4, Nf3...).")
        # Bot move
        else:
            print("\nBot thinking (MCTS simulations)...")
            # 80-100 simulations for Python — comfortable speed (~2-4 seconds per move)
            bot_move = engine.search(board, num_simulations=80)
            print(f"Bot played: {board.san(bot_move)}")
            board.push(bot_move)
            
    print("\nGame over!")
    print(f"Result: {board.result()}")

if __name__ == "__main__":
    play_against_bot()