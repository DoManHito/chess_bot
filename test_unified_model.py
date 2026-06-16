"""
Test script for unified model with lookahead capability.

This script tests:
1. Model initialization
2. Move classification
3. Policy generation
4. Lookahead data generation
"""

import chess
import torch
import os
from models.unified_chess_nets import UnifiedMoveClassifierNet, ChessCoreNet, CLASS_NAMES
from classifiers.move_classifier import MoveClassifier
from engine.unified_mcts import UnifiedMCTS


def test_model_initialization():
    """Test model initialization."""
    print("=" * 60)
    print("TEST 1: Model Initialization")
    print("=" * 60)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Initialize core network
        core = ChessCoreNet(in_channels=25)
        print("✓ ChessCoreNet initialized")
        
        # Initialize unified model
        model = UnifiedMoveClassifierNet(
            core_net=core,
            num_classes=len(CLASS_NAMES),
            policy_output_dim=64
        )
        print("✓ UnifiedMoveClassifierNet initialized")
        
        # Load weights if available
        weights_path = "models/weights_bot.pth"
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state, strict=False)
            model.to(device)
            model.eval()
            print(f"✓ Weights loaded from {weights_path}")
        else:
            print(f"⚠️ Weights not found at {weights_path}, using random initialization")
        
        return model, device
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        raise


def test_move_classification(classifier, device):
    """Test move classification."""
    print("\n" + "=" * 60)
    print("TEST 2: Move Classification")
    print("=" * 60)
    
    try:
        # Test position: starting position
        board = chess.Board()
        move_san = "e4"
        
        # Single move classification using MoveClassifier wrapper
        result, value = classifier.classify_move(
            board_fen=board.fen(),
            move_san=move_san
        )
        
        print(f"Position: {board.fen()}")
        print(f"Move: {move_san}")
        print(f"Classification: {result.classification}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Evaluation: {value:.4f}")
        print("✓ Single move classification works")
        
        # Batch classification
        moves = ["e4", "Nf3", "Bb5", "d3"]
        classes, confidences, values = classifier.classify_moves_batch(board, moves)
        
        print(f"\nBatch classification for {len(moves)} moves:")
        for i, (cls, conf, val) in enumerate(zip(classes, confidences, values)):
            print(f"  {moves[i]}: {cls} (conf={conf:.4f}, eval={val:.4f})")
        print("✓ Batch classification works")
        
        return True
    except Exception as e:
        print(f"✗ Move classification failed: {e}")
        raise


def test_policy_generation(classifier, device):
    """Test policy generation."""
    print("\n" + "=" * 60)
    print("TEST 3: Policy Generation")
    print("=" * 60)
    
    try:
        board = chess.Board()
        
        # Get policy from MoveClassifier wrapper
        policy = classifier.get_policy(board, top_k=10)
        
        print(f"Policy for starting position (top 10 moves):")
        for move, prob in sorted(policy.items(), key=lambda x: -x[1])[:10]:
            print(f"  {move}: {prob:.4f}")
        
        print("✓ Policy generation works")
        return True
    except Exception as e:
        print(f"✗ Policy generation failed: {e}")
        raise


def test_mcts_integration(classifier, device):
    """Test MCTS integration with unified model."""
    print("\n" + "=" * 60)
    print("TEST 4: MCTS Integration")
    print("=" * 60)
    
    try:
        # Initialize MCTS engine with MoveClassifier wrapper
        mcts = UnifiedMCTS(
            unified_model=classifier,
            cpuct=2.0,
            max_simulations=10,
            top_moves_ratio=0.3,
            policy_output_dim=64
        )
        print("✓ UnifiedMCTS initialized")
        
        # Test search
        board = chess.Board()
        best_move, visit_dict = mcts.search(board, num_simulations=10)
        
        print(f"\nBest move: {board.san(best_move) if best_move else 'None'}")
        print(f"Visit counts: {visit_dict}")
        print("✓ MCTS search works")
        
        return True
    except Exception as e:
        print(f"✗ MCTS integration failed: {e}")
        raise


def test_full_classifier():
    """Test full MoveClassifier with unified model."""
    print("\n" + "=" * 60)
    print("TEST 5: Full MoveClassifier")
    print("=" * 60)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize classifier with unified model
        classifier = MoveClassifier(
            weights_path="models/weights_bot.pth",
            device=device,
            use_unified_model=True,
            policy_output_dim=64
        )
        print("✓ MoveClassifier initialized with unified model")
        
        # Test classification
        board = chess.Board()
        result, value = classifier.classify_move(
            board_fen=board.fen(),
            move_san="e4"
        )
        
        print(f"Classification: {result.classification}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Evaluation: {value:.4f}")
        print("✓ Full classifier works")
        
        # Test policy
        policy = classifier.get_policy(board, top_k=5)
        print(f"\nTop 5 moves from policy:")
        for move, prob in sorted(policy.items(), key=lambda x: -x[1])[:5]:
            print(f"  {move}: {prob:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Full classifier test failed: {e}")
        raise


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("UNIFIED MODEL TEST SUITE")
    print("=" * 60)
    
    # Test 1: Model initialization
    model, device = test_model_initialization()
    
    # Test 2: Move classification - use MoveClassifier wrapper
    classifier = MoveClassifier(
        weights_path="models/weights_bot.pth",
        device=device,
        use_unified_model=True,
        policy_output_dim=64
    )
    test_move_classification(classifier, device)
    
    # Test 3: Policy generation - use MoveClassifier wrapper
    test_policy_generation(classifier, device)
    
    # Test 4: MCTS integration - use MoveClassifier wrapper
    test_mcts_integration(classifier, device)
    
    # Test 5: Full classifier
    test_full_classifier()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
