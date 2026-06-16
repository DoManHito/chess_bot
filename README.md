# Chess Bot - Unified Model with Lookahead Capability

A chess analysis system that uses a unified neural network with lookahead capability to classify and evaluate chess moves. The system combines move classification, position evaluation, and policy prediction into a single model that learns to predict move quality based on future game outcomes.

## Table of Contents

- [Project Description](#project-description)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [CLI Commands](#cli-commands)
- [Project Structure](#project-structure)
- [Unified Model Architecture](#unified-model-architecture)
- [Lookahead Capability](#lookahead-capability)
- [Move Classification](#move-classification)
- [Training Loop](#training-loop)
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
- **Supports web interface** for game analysis

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
- scikit-learn >= 0.24.0
- Stockfish binary (optional, for move analysis)

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

### CLI Commands

```bash
# Initialize the database
python main.py init

# Parse and classify PGN files
python main.py parse games.pgn

# Get player statistics
python main.py stats PlayerName

# List all games
python main.py list --limit 100

# Show classifications for a specific game
python main.py classify 1

# Run continuous self-play and training loop
python main.py auto-train --iters 10 --games 20 --sims 60 --epochs 5

# Train unified model on existing Stockfish games (supervised)
python train_unified.py --mode supervised --epochs 10 --batch-size 256

# Train unified model with self-play RL
python train_unified.py --mode rl --iterations 5 --games-per-iter 10 --sims 800

# Combined: supervised first, then RL
python train_unified.py --mode combined --epochs 5 --iterations 3
```

#### auto-train Options
- `--iters`: Number of global iterations per cycle (default: 10)
- `--games`: Number of games generated per iteration (default: 20)
- `--sims`: Number of MCTS simulations per move (default: 60)
- `--epochs`: Number of neural network training epochs per iteration (default: 5)
- `--keep-n`: Number of last games to keep in database (default: 100)
- `--db-path`: Path to database (default: chess_bot.db)
- `--workers`: Number of processes for parallel game generation (default: 1)
- `--lookahead`: Enable lookahead training (default: False)

### Training Modes

The `train_unified.py` script supports three training modes:

#### 1. Supervised Training

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

#### 2. Self-Play RL Training

Generate self-play games and train with MCTS:

```bash
python train_unified.py --mode rl --iterations 5 --games-per-iter 10 --sims 800
```

Options:
- `--iterations`: Number of RL iterations (default: 5)
- `--games-per-iter`: Games generated per iteration (default: 10)
- `--sims`: MCTS simulations per position (default: 800)
- `--epochs`: Training epochs per iteration (default: 3)
- `--keep-last-n`: Games to keep in database (default: 1000)
- `--temperature`: Temperature for move selection (default: 1.2)
- `--num-workers`: Parallel workers (default: 1)

#### 3. Combined Training

First train on existing Stockfish games, then continue with self-play RL:

```bash
python train_unified.py --mode combined --epochs 5 --iterations 3
```

This is recommended for best results:
1. Phase 1: Learn move quality patterns from 99k Stockfish games
2. Phase 2: Refine with self-play and MCTS

### Training Statistics

Training statistics are saved to `models/training_stats.json`:

```json
{
  "epochs": 10,
  "batch_size": 1024,
  "lr": 0.001,
  "alpha": 0.5,
  "dataset_size": 658,
  "train_size": 593,
  "val_size": 66,
  "best_val_loss": 0.8264,
  "class_distribution": {
    "0": 7,
    "2": 17,
    "3": 312,
    "4": 222,
    "5": 101
  }
}
```

Class distribution:
- **0 (Best)**: Ideal moves with minimal evaluation change
- **1 (Excellent)**: Very good moves
- **2 (Good)**: Solid moves
- **3 (Inaccuracy)**: Suboptimal moves
- **4 (Mistake)**: Poor moves
- **5 (Blunder)**: Very bad moves

### Web Interface

Open `index.html` in a browser to:
- Play chess on the board
- See classification of your moves
- Compare neural network moves with ideal moves

### Python Usage Example

```python
from classifiers.move_classifier import MoveClassifier

classifier = MoveClassifier()
result, value = classifier.classify_move(
    board_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    move_san="Nf3",
    evaluation=0.5,
    turn_num=1,
    turn_label="White"
)
print(f"Classification: {result.classification}")
print(f"Position value: {value}")
```

## Project Structure

```
chess_bot/
├── classifiers/              # Move classification module
│   ├── __init__.py
│   ├── classification_config.py    # Class names and thresholds
│   ├── move_classifier.py          # Main move classifier
│   ├── self_play_dataset.py        # Self-play training dataset
│   ├── stockfish_policy_dataset.py # Stockfish policy dataset
│   └── train_classifier.py         # Classifier training script
│
├── database/                 # Database operations module
│   ├── __init__.py
│   ├── chess_db.py           # CRUD operations for games and moves
│   └── schema.py             # SQL schema definitions
│
├── engine/                   # Chess engine module
│   ├── __init__.py
│   ├── mcts.py               # Monte Carlo Tree Search with classification priorities
│   ├── unified_mcts.py       # Unified MCTS with policy head
│   └── rl_trainer.py         # Reinforcement learning training loop
│
├── models/                   # Neural network models
│   ├── __init__.py
│   ├── chess_nets.py         # Legacy network definitions
│   ├── unified_chess_nets.py # Unified model with lookahead support
│   └── weights_bot.pth       # Bot weights for self-play
│
├── parsers/                  # Format parsers
│   ├── __init__.py
│   └── pgn_parser.py         # PGN file parser with Stockfish filter
│
├── plans/                    # Project planning documents
│   └── unified_model_plan.md # Unified model implementation plan
│
├── server.py                 # FastAPI server
├── main.py                   # CLI for PGN processing
├── prepare_dataset.py        # Database optimization script
├── export_pgn.py             # Export self-play games to PGN
├── test_inference.py         # Inference tests
├── test_unified_model.py     # Unified model tests
├── stockfish_policy_data.json # Stockfish policy data
├── self_play_games.pgn       # Example self-play games
├── index.html                # Web interface
└── requirements.txt          # Python dependencies
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

## Training Loop

The training loop consists of three phases:

### Phase 1: Self-Play Game Generation
Generates games using the MCTS engine with temperature-based exploration. Games are saved to the `self_play_moves` table with lookahead data.

### Phase 2: Unified Model Training
Trains the unified model using:
- Cross-entropy loss for classification
- MSE loss for value prediction
- KL divergence loss for policy prediction
- AdamW optimizer with learning rate scheduling
- Sliding window to keep only the most recent games

### Phase 3: Sliding Window
Maintains a fixed-size dataset by keeping only the last N games, ensuring the model trains on recent data.

### Running Training

```bash
# Standard training
python main.py auto-train --iters 10 --games 20 --sims 60 --epochs 5 --keep-n 100

# Training with lookahead capability
python main.py auto-train --iters 10 --games 20 --sims 60 --epochs 5 --keep-n 100 --lookahead
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

### move_sequences Table (Lookahead Sequences)
```sql
CREATE TABLE move_sequences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    move_number INTEGER NOT NULL,
    fen_before TEXT,
    move_san TEXT,
    lookahead_depth INTEGER,
    future_moves TEXT,
    final_evaluation REAL,
    final_classification TEXT,
    move_sequence_classes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games (id) ON DELETE CASCADE
)
```

## License

This project is open source.
