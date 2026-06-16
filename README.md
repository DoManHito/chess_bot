# Chess Bot - Unified Model with Lookahead Capability

A chess analysis system that uses a unified neural network with lookahead capability to classify and evaluate chess moves. The system combines move classification, position evaluation, and policy prediction into a single model that learns to predict move quality based on future game outcomes.

## Table of Contents

- [Project Description](#project-description)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Training](#training)
- [Project Structure](#project-structure)
- [Unified Model Architecture](#unified-model-architecture)
- [Lookahead Capability](#lookahead-capability)
- [Move Classification](#move-classification)
- [Database Schema](#database-schema)

## Project Description

Chess Bot is an advanced chess analysis system with unified model architecture that:

- **Unified Model**: Single neural network with three heads (Classification, Value, Policy)
- **Lookahead Learning**: Trains on move sequences to understand consequences of each move
- **Classifies moves** as: Best, Excellent, Good, Inaccuracy, Mistake, Blunder
- **Evaluates positions** using values from -1 to 1
- **Predicts policies** for MCTS acceleration
- **Generates training data** through self-play games with lookahead sequences
- **Provides REST API** for system interaction

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- chess >= 1.1.0
- FastAPI >= 0.100.0
- Uvicorn >= 0.22.0
- tqdm >= 4.65.0

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download Stockfish (optional, for move analysis):
```bash
# Download from https://stockfishchess.org/download/
# Place at ./stockfish
```

3. Start the server:
```bash
python server.py
```

## Usage

### API Endpoints

- `GET /` - API information
- `GET /analyze_move` - Analyze a move (requires FEN and SAN)
- `GET /get_move` - Get bot's best move (requires FEN, optional simulations count)

#### `/analyze_move` Parameters
- `fen` (string): Board position in FEN notation
- `move_san` (string): Move in SAN notation

#### `/get_move` Parameters
- `fen` (string): Board position in FEN notation
- `simulations` (int, default: 100): Number of MCTS simulations

#### Response Example
```json
{
  "move_uci": "e2e4",
  "move_san": "e4",
  "bot_nn_class": "Excellent",
  "bot_ideal_class": "Best"
}
```

## Training

### Self-Play RL Training

Generate self-play games and train with MCTS:

```bash
python train_unified.py --mode rl --iterations 5 --games-per-iter 10 --sims 800 --epochs 3 --num-workers 2
```

Options:
- `--iterations`: Number of RL iterations (default: 5)
- `--games-per-iter`: Games generated per iteration (default: 10)
- `--sims`: MCTS simulations per position (default: 800)
- `--epochs`: Training epochs per iteration (default: 3)
- `--num-workers`: Parallel workers (default: 1)

### Supervised Training

Train on existing Stockfish games from the database using evaluation delta as ground truth:

```bash
python train_unified.py --mode supervised --epochs 10 --batch-size 256 --sample-rate 1.0
```

Options:
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 256)
- `--lr`: Learning rate (default: 0.001)
- `--alpha`: Value loss weight (default: 0.5)
- `--sample-rate`: Fraction of data to sample (1.0 = all data)
- `--no-val`: Disable validation split
- `--val-ratio`: Validation ratio (default: 0.1)

### Combined Training

First train on existing Stockfish games, then continue with self-play RL:

```bash
python train_unified.py --mode combined --epochs 5 --iterations 3
```

This is recommended for best results:
1. Phase 1: Learn move quality patterns from Stockfish games
2. Phase 2: Refine with self-play and MCTS

## Project Structure

```
chess_bot/
├── classifiers/              # Move classification module
│   ├── classification_config.py    # Class names and thresholds
│   ├── move_classifier.py          # Main move classifier
│   └── self_play_dataset.py        # Self-play training dataset
│
├── database/                 # Database operations module
│   ├── schema.py             # SQL schema definitions
│   ├── chess_db.py           # CRUD operations for games and moves
│   └── migrate_db.py         # Database migration script
│
├── engine/                   # Chess engine module
│   ├── unified_mcts.py       # Unified MCTS with policy head
│   └── rl_trainer.py         # Reinforcement learning training loop
│
├── models/                   # Neural network models
│   ├── unified_chess_nets.py # Unified model with lookahead support
│   ├── weights_bot.pth       # Bot weights for self-play
│   └── weights_bot copy.pth  # Backup weights
│
├── parsers/                  # Format parsers
│   └── pgn_parser.py         # PGN file parser with Stockfish filter
│
├── train_unified.py          # Unified training script
├── server.py                 # FastAPI server
├── test_inference.py         # Inference tests
├── test_unified_model.py     # Unified model tests
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore
```

## Unified Model Architecture

### UnifiedMoveClassifierNet

The unified model combines three heads into a single network:

```
┌─────────────────────────────────────────────────────────┐
│              Unified Chess Model                         │
│  ┌───────────────────────────────────────────────┐     │
│  │           Shared Backbone (ChessCoreNet)       │     │
│  │  (Conv2D + Residual Blocks)                    │     │
│  └─────────────┬─────────────────────────────────┘     │
│                │                                         │
│  ┌─────────────┴─────────────────────────────────┐     │
│  │  Classification Head (6 classes)               │     │
│  │  (Best/Excellent/Good/Inaccuracy/Mistake/Blunder)│     │
│  └───────────────────────────────────────────────┘     │
│  ┌───────────────────────────────────────────────┐     │
│  │  Value Head (-1 to 1)                          │     │
│  └───────────────────────────────────────────────┘     │
│  ┌───────────────────────────────────────────────┐     │
│  │  Policy Head (Move probabilities)              │     │
│  │  (for MCTS acceleration)                       │     │
│  └───────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### ChessCoreNet (Shared Backbone)
- Input: 25-channel board representation (12 piece types + turn indicator)
- Initial convolution: 3x3 kernel, 128 output channels
- 4 residual blocks with BatchNorm and ReLU
- Output: 128-channel feature map

### Classification Head
- Convolution reduction: 1x1 kernel to 16 channels
- Fully connected: 16*8*8 -> 256 -> 6 (class logits)
- Dropout: 0.4

### Value Head
- Fully connected: 16*8*8 -> 32 -> 1
- Output: tanh-scaled value in [-1, 1]

### Policy Head (NEW for MCTS)
- Fully connected: 16*8*8 -> 128 -> 32 -> 64
- Output: Softmax-normalized move probabilities
- Used to accelerate MCTS by predicting move distributions

## Lookahead Capability

### Concept
Instead of classifying only a single move, the model learns to evaluate:
1. **Current move** (as before)
2. **Subsequent N moves** (lookahead sequence)
3. **Final game result** after the sequence

### Lookahead Data Structure
```python
@dataclass
class LookaheadMoveData:
    """Data for lookahead training"""
    board_fen: str                    # Starting position
    move_san: str                     # Move to evaluate
    lookahead_depth: int              # Sequence depth (e.g., 3)
    future_moves: List[str]           # Subsequent moves
    final_evaluation: float           # Evaluation after sequence
    final_classification: str         # Class of final position
    move_sequence_classifications: List[str]  # Class of each move in sequence
```

### Example Lookahead Data
```json
{
  "board_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "move_san": "e4",
  "lookahead_depth": 3,
  "future_moves": ["e5", "Nf3", "Nc6"],
  "final_evaluation": 0.15,
  "final_classification": "Good",
  "move_sequence_classifications": ["Excellent", "Good", "Excellent"]
}
```

### Training with Lookahead
The unified model is trained with a combined loss:
```python
loss = (
    alpha * classification_loss +
    beta * value_loss +
    gamma * policy_loss
)
```

### Benefits
1. **Lookahead learning** — Model learns consequences of moves
2. **Unified architecture** — Less code, easier maintenance
3. **Policy head** — MCTS acceleration via predicted move distributions
4. **Consistency** — All predictions from single model
5. **Data efficiency** — Each self-play game yields more training examples

## Move Classification

The system classifies moves based on the evaluation delta (change in position value) before and after the move:

| Class | Description | Delta Range |
|-------|-------------|-------------|
| **Best** | Best move | 0.0 - 0.02 |
| **Excellent** | Very good move | 0.02 - 0.15 |
| **Good** | Good move | 0.15 - 0.40 |
| **Inaccuracy** | Inaccurate move | 0.40 - 0.80 |
| **Mistake** | Mistake | 0.80 - 1.50 |
| **Blunder** | Blunder | > 1.50 |

### MCTS Move Priorities

```
Best: 1.0
Excellent: 0.8
Good: 0.5
Inaccuracy: 0.2
Mistake: 0.05
Blunder: 0.001
```

## Database Schema

### games Table
```sql
CREATE TABLE games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    white_player TEXT NOT NULL,
    black_player TEXT NOT NULL,
    fen_start TEXT,
    fen_end TEXT,
    result TEXT,
    classification TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### moves Table
```sql
CREATE TABLE moves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    move_number INTEGER NOT NULL,
    fen_before TEXT,
    fen_after TEXT,
    move_san TEXT,
    classification TEXT,
    evaluation REAL,
    evaluation_change REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games (id) ON DELETE CASCADE
)
```

### self_play_moves Table (with Lookahead Support)
```sql
CREATE TABLE self_play_moves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER,
    fen_before TEXT,
    move_uci TEXT,
    mcts_policy TEXT,
    result_value REAL,
    lookahead_depth INTEGER,
    future_moves TEXT,
    final_classification TEXT,
    move_sequence_classes TEXT
)
```

## License

This project is open source.
