# Chess Bot - Unified Model with Lookahead Capability (Option A)

A chess analysis system that uses a unified neural network to classify and evaluate chess moves through value-based lookahead. The system computes move quality by comparing position evaluation before and after each move, eliminating the need for a separate classification head.

## Table of Contents

- [Project Description](#project-description)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Training](#training)
- [Project Structure](#project-structure)
- [Unified Model Architecture](#unified-model-architecture)
- [Move Classification](#move-classification)
- [Database Schema](#database-schema)

## Project Description

Chess Bot is an advanced chess analysis system using **Option A** architecture:

- **Unified Model**: Single neural network with Value and Policy heads (classification computed via value difference)
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
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download Stockfish (optional, for move analysis):
```bash
# Download from https://stockfishchess.org/download/
# Place at ./stockfish-ubuntu-x86-64-avx2
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
- `GET /get_policy` - Get move policy from unified model (requires FEN)

#### `/analyze_move` Parameters
- `fen` (string): Board position in FEN notation
- `move_san` (string): Move in SAN notation

#### `/get_move` Parameters
- `fen` (string): Board position in FEN notation
- `simulations` (int, default: 100): Number of MCTS simulations

#### `/get_policy` Parameters
- `fen` (string): Board position in FEN notation
- `top_k` (int, default: 64): Number of top moves to return

#### Response Examples

**`/analyze_move` Response:**
```json
{
  "nn_class": "Excellent",
  "ideal_class": "Best",
  "move_san": "e4",
  "move_uci": "e2e4"
}
```

**`/get_move` Response:**
```json
{
  "move_uci": "e2e4",
  "move_san": "e4",
  "bot_nn_class": "Excellent",
  "bot_ideal_class": "Best",
  "visit_counts": {"e2e4": 45, "g1f3": 30, ...},
  "total_time": 0.234
}
```

**`/get_policy` Response:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "policy": {"e7e5": 0.35, "g8f6": 0.28, "g8f6": 0.15, ...},
  "total_moves": 64
}
```

## Training

### Supervised Training

Train on existing Stockfish games from the database using evaluation delta as ground truth:

```bash
python train_unified.py --mode supervised --epochs 10 --batch-size 256 --lr 1e-3 --alpha 0.5 --sample-rate 1.0
```

Options:
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 256)
- `--lr`: Learning rate (default: 0.001)
- `--alpha`: Value loss weight (default: 0.5)
- `--sample-rate`: Fraction of data to sample (1.0 = all data)
- `--no-val`: Disable validation split
- `--val-ratio`: Validation ratio (default: 0.1)
- `--num-workers`: Parallel data loading workers (default: 4)

### Self-Play RL Training

Generate self-play games and train with MCTS:

```bash
python train_unified.py --mode rl --iterations 5 --games-per-iter 10 --sims 800 --epochs 3
```

Options:
- `--iterations`: Number of RL iterations (default: 5)
- `--games-per-iter`: Games generated per iteration (default: 10)
- `--sims`: MCTS simulations per position (default: 800)
- `--epochs`: Training epochs per iteration (default: 3)
- `--keep-last-n`: Keep only last N games in database (default: 1000)
- `--temperature`: Exploration temperature (default: 1.2)
- `--num-workers`: Parallel workers (default: 1)

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
│   ├── move_classifier.py          # Main move classifier (Option A)
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
│   ├── unified_chess_nets.py # Unified model (Option A)
│   ├── weights_bot.pth       # Bot weights
│   └── weights_bot copy.pth  # Backup weights
│
├── parsers/                  # Format parsers
│   └── pgn_parser.py         # PGN file parser with Stockfish filter
│
├── train_unified.py          # Unified training script
├── server.py                 # FastAPI server
├── test_inference.py         # Inference tests
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore
```

## Unified Model Architecture

### Option A: Value-Based Classification

The unified model uses **Option A** architecture where classification is computed via value difference rather than a dedicated classification head:

```
┌─────────────────────────────────────────────────────────┐
│              Unified Chess Model (Option A)              │
│  ┌───────────────────────────────────────────────┐     │
│  │           Shared Backbone (ChessCoreNet)       │     │
│  │  (Conv2D + Residual Blocks)                    │     │
│  └─────────────┬─────────────────────────────────┘     │
│                │                                         │
│  ┌─────────────┴─────────────────────────────────┐     │
│  │  Value Head (-1 to 1)                          │     │
│  │  (Position evaluation)                         │     │
│  └───────────────────────────────────────────────┘     │
│  ┌───────────────────────────────────────────────┐     │
│  │  Policy Head (Move probabilities)              │     │
│  │  (4096 dimensions for all legal moves)         │     │
│  └───────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘

Classification is computed as:
- loss = |value_before - value_after| (for White)
- loss = |value_after - value_before| (for Black)
```

### ChessCoreNet (Shared Backbone)
- Input: **13-channel** board representation (12 piece types + turn indicator)
  - Channels 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
  - Channels 6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
  - Channel 12: Active turn (1 = White, 0 = Black)
- Initial convolution: 3x3 kernel, 128 output channels
- 6 residual blocks with BatchNorm and ReLU
- Output: 128-channel feature map (8x8)

### Value Head
- Convolution reduction: 1x1 kernel to 32 channels
- Fully connected: 32*8*8 → 256 → 1
- Output: tanh-scaled value in [-1, 1]

### Policy Head
- Convolution reduction: 1x1 kernel to 32 channels
- Fully connected: 32*8*8 → 256 → 4096
- Output: Raw logits (softmax applied during inference)
- Used to accelerate MCTS by predicting move distributions

## Move Classification

The system classifies moves based on the evaluation delta (change in position value) before and after the move:

| Class | Description | Delta Range |
|-------|-------------|-------------|
| **Best** | Best move | ≤ 0.02 |
| **Excellent** | Very good move | ≤ 0.07 |
| **Good** | Good move | ≤ 0.15 |
| **Inaccuracy** | Inaccurate move | ≤ 0.30 |
| **Mistake** | Mistake | ≤ 0.55 |
| **Blunder** | Blunder | > 0.55 |

### MCTS Move Priorities

```
Best: 1.0
Excellent: 0.8
Good: 0.5
Inaccuracy: 0.2
Mistake: 0.05
Blunder: 0.001
```

### Confidence via Policy Head

The policy head provides move confidence as a probability distribution over all 4096 possible moves. The confidence for a specific move is its softmax probability.

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
