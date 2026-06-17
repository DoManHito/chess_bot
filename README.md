# Chess Bot - Zjednoczony Model z Możliwością Przewidywania (Opcja A)

System analizy szachów wykorzystujący zjednoczoną sieć neuronową do klasyfikacji i oceny ruchów szachowych poprzez przewidywanie wartości. System oblicza jakość ruchu poprzez porównanie oceny pozycji przed i po każdym ruchu, eliminując potrzebę oddzielnego głowicy klasyfikacyjnej.

## Spis treści

- [Opis Projektu](#opis-projektu)
- [Wymagania](#wymagania)
- [Instalacja](#instalacja)
- [Użycie](#użycie)
- [Koniec Punktów API](#koniec-punktów-api)
- [Trening](#trening)
- [Struktura Projektu](#struktura-projektu)
- [Architektura Zjednoczonego Modelu](#architektura-zjednoczonego-modelu)
- [Klasyfikacja Ruchów](#klasyfikacja-ruchów)
- [Schemat Bazy Danych](#schemat-bazy-danych)

## Opis Projektu

Chess Bot to zaawansowany system analizy szachów wykorzystujący architekturę **Opcja A**:

- **Zjednoczony Model**: Jedna sieć neuronowa z głowicami Value i Policy (klasyfikacja obliczana poprzez różnicę wartości)
- **Uczenie z Przewidywaniem**: Trening na sekwencjach ruchów do zrozumienia konsekwencji każdego ruchu
- **Klasifikuje ruchy** jako: Najlepszy, Wyśmienity, Dobry, Nieprecyzyjny, Błąd, Katastrofa
- **Ocenia pozycje** używając wartości od -1 do 1
- **Przewiduje polityki** dla przyspieszenia MCTS
- **Generuje dane treningowe** poprzez gry samogry z sekwencjami przewidywania
- **Dostarcza REST API** do interakcji z systemem

## Wymagania

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

## Instalacja

1. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

2. Pobierz Stockfish (opcjonalnie, do analizy ruchów):
```bash
# Pobierz z https://stockfishchess.org/download/
# Umieść w ./stockfish-ubuntu-x86-64-avx2
```

3. Uruchom serwer:
```bash
python server.py
```

## Użycie

### Koniec Punktów API

- `GET /` - Informacje o API
- `GET /analyze_move` - Przeanalizuj ruch (wymaga FEN i SAN)
- `GET /get_move` - Pobierz najlepszy ruch bota (wymaga FEN, opcjonalna liczba symulacji)
- `GET /get_policy` - Pobierz politykę ruchu ze zjednoczonego modelu (wymaga FEN)

#### Parametry `/analyze_move`
- `fen` (string): Pozycja planszy w notacji FEN
- `move_san` (string): Ruch w notacji SAN

#### Parametry `/get_move`
- `fen` (string): Pozycja planszy w notacji FEN
- `simulations` (int, domyślnie: 100): Liczba symulacji MCTS

#### Parametry `/get_policy`
- `fen` (string): Pozycja planszy w notacji FEN
- `top_k` (int, domyślnie: 64): Liczba najlepszych ruchów do zwrócenia

#### Przykłady Odpowiedzi

**Odpowiedź `/analyze_move`:**
```json
{
  "nn_class": "Wyśmienity",
  "ideal_class": "Najlepszy",
  "move_san": "e4",
  "move_uci": "e2e4"
}
```

**Odpowiedź `/get_move`:**
```json
{
  "move_uci": "e2e4",
  "move_san": "e4",
  "bot_nn_class": "Wyśmienity",
  "bot_ideal_class": "Najlepszy",
  "visit_counts": {"e2e4": 45, "g1f3": 30, ...},
  "total_time": 0.234
}
```

**Odpowiedź `/get_policy`:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "policy": {"e7e5": 0.35, "g8f6": 0.28, "g8f6": 0.15, ...},
  "total_moves": 64
}
```

## Trening

### Nadzorowany Trening

Trening na istniejących grach Stockfish z bazy danych używając delta oceny jako prawdy:

```bash
python train_unified.py --mode supervised --epochs 10 --batch-size 256 --lr 1e-3 --alpha 0.5 --sample-rate 1.0
lub
python train_unified.py --mode rl --iterations 10 --games-per-iter 20 --sims 100 --temperature 1.0 --keep-last-n 5000 --epochs 3 --num-workers 2
```

Opcje:
- `--epochs`: Liczba epok treningowych (domyślnie: 10)
- `--batch-size`: Rozmiar partii do treningu (domyślnie: 256)
- `--lr`: Wskaźnik uczenia (domyślnie: 0.001)
- `--alpha`: Waga straty wartości (domyślnie: 0.5)
- `--sample-rate`: Ułamek danych do pobrania (1.0 = wszystkie dane)
- `--no-val`: Wyłącz podział walidacji
- `--val-ratio`: Stosunek walidacji (domyślnie: 0.1)
- `--num-workers`: Pracownicy równoległego ładowania danych (domyślnie: 4)

### Trening RL Samogry

Generuj gry samogry i trenuj z MCTS:

```bash
python train_unified.py --mode rl --iterations 5 --games-per-iter 10 --sims 800 --epochs 3
```

Opcje:
- `--iterations`: Liczba iteracji RL (domyślnie: 5)
- `--games-per-iter`: Gry generowane na iterację (domyślnie: 10)
- `--sims`: Symulacje MCTS na pozycję (domyślnie: 800)
- `--epochs`: Epoki treningowe na iterację (domyślnie: 3)
- `--keep-last-n`: Zachowaj tylko ostatnie N gier w bazie danych (domyślnie: 1000)
- `--temperature`: Temperatura eksploracji (domyślnie: 1.2)
- `--num-workers`: Pracownicy równolegli (domyślnie: 1)

### Połączony Trening

Najpierw trenuj na istniejących grach Stockfish, a następnie kontynuuj z RL samogry:

```bash
python train_unified.py --mode combined --epochs 5 --iterations 3
```

Zalecane dla najlepszych wyników:
1. Faza 1: Naucz wzorce jakości ruchów z gier Stockfish
2. Faza 2: Udoskonal z samogry i MCTS

## Struktura Projektu

```
chess_bot/
├── classifiers/              # Moduł klasyfikacji ruchów
│   ├── classification_config.py    # Nazwy klas i progi
│   ├── move_classifier.py          # Główny klasyfikator ruchów (Opcja A)
│   └── self_play_dataset.py        # Zbiór danych treningowych samogry
│
├── database/                 # Moduł operacji bazy danych
│   ├── schema.py             # Definicje schematu SQL
│   ├── chess_db.py           # Operacje CRUD dla gier i ruchów
│   └── migrate_db.py         # Skrypt migracji bazy danych
│
├── engine/                   # Moduł silnika szachowego
│   ├── unified_mcts.py       # Zjednoczony MCTS z głowicą policy
│   └── rl_trainer.py         # Pętla treningu uczenia się wzmocnienia
│
├── models/                   # Modele sieci neuronowej
│   ├── unified_chess_nets.py # Zjednoczony model (Opcja A)
│   ├── weights_bot.pth       # Wagi bota
│   └── weights_bot copy.pth  # Zapasowe wagi
│
├── parsers/                  # Parserzy formatów
│   └── pgn_parser.py         # Parser plików PGN z filtrem Stockfish
│
├── train_unified.py          # Skrypt treningu zjednoczonego
├── server.py                 # Serwer FastAPI
├── test_inference.py         # Testy wnioskowania
├── requirements.txt          # Zależności Pythona
├── README.md                 # Ten plik
└── .gitignore
```

## Architektura Zjednoczonego Modelu

### Opcja A: Klasyfikacja oparta na wartości

Zjednoczony model wykorzystuje architekturę **Opcja A**, gdzie klasyfikacja jest obliczana poprzez różnicę wartości zamiast dedykowanej głowicy klasyfikacyjnej:

```
┌─────────────────────────────────────────────────────────┐
│              Zjednoczony Model Szachowy (Opcja A)         │
│  ┌───────────────────────────────────────────────┐     │
│  │           Wspólny Trzon (ChessCoreNet)         │     │
│  │  (Conv2D + Bloki Residualne)                   │     │
│  └─────────────┬─────────────────────────────────┘     │
│                │                                         │
│  ┌─────────────┴─────────────────────────────────┐     │
│  │  Głowica Value (-1 do 1)                       │     │
│  │  (Ocena pozycji)                               │     │
│  └───────────────────────────────────────────────┘     │
│  ┌───────────────────────────────────────────────┐     │
│  │  Głowica Policy (Prawdopodobieństwa ruchów)    │     │
│  │  (4096 wymiarów dla wszystkich legalnych ruchów) │     │
│  └───────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘

Klasyfikacja jest obliczana jako:
- strata = |value_before - value_after| (dla Białych)
- strata = |value_after - value_before| (dla Czarnych)
```

### ChessCoreNet (Wspólny Trzon)
- Wejście: **13-kanałowa** reprezentacja planszy (12 typów figur + wskaźnik ruchu)
  - Kanały 0-5: Figury białe (Pion, Skoczek, Latający Słup, Wieża, Królowa, Król)
  - Kanały 6-11: Figury czarne (Pion, Skoczek, Latający Słup, Wieża, Królowa, Król)
  - Kanał 12: Aktywny ruch (1 = Białe, 0 = Czarne)
- Początkowa konwolucja: jądro 3x3, 128 kanałów wyjściowych
- 6 bloków residualnych z BatchNorm i ReLU
- Wyjście: mapa cech 128-kanałowa (8x8)

### Głowica Value
- Redukcja konwolucyjna: jądro 1x1 do 32 kanałów
- Pełnie połączone: 32*8*8 → 256 → 1
- Wyjście: wartość skalowana tanh w [-1, 1]

### Głowica Policy
- Redukcja konwolucyjna: jądro 1x1 do 32 kanałów
- Pełnie połączone: 32*8*8 → 256 → 4096
- Wyjście: surowe logits (softmax zastosowane podczas wnioskowania)
- Używane do przyspieszenia MCTS poprzez przewidywanie dystrybucji ruchów

## Klasyfikacja Ruchów

System klasyfikuje ruchy na podstawie delta oceny (zmiana wartości pozycji) przed i po ruchu:

| Klasa | Opis | Zakres Delta |
|-------|------|--------------|
| **Najlepszy** | Najlepszy ruch | ≤ 0.02 |
| **Wyśmienity** | Bardzo dobry ruch | ≤ 0.07 |
| **Dobry** | Dobry ruch | ≤ 0.15 |
| **Nieprecyzyjny** | Nieprecyzyjny ruch | ≤ 0.30 |
| **Błąd** | Błąd | ≤ 0.55 |
| **Katastrofa** | Katastrofa | > 0.55 |

### Priorytety Ruchów MCTS

```
Najlepszy: 1.0
Wyśmienity: 0.8
Dobry: 0.5
Nieprecyzyjny: 0.2
Błąd: 0.05
Katastrofa: 0.001
```

### Pewność poprzez Głowicę Policy

Głowica policy dostarcza pewności ruchu jako dystrybucję prawdopodobieństwa nad wszystkimi 4096 możliwymi ruchami. Pewność dla konkretnego ruchu to jego prawdopodobieństwo softmax.

## Schemat Bazy Danych

### Tabela games
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

### Tabela moves
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

### Tabela self_play_moves (z obsługą Lookahead)
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
