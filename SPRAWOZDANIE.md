# Zaawansowany System Analizy Szachowej
## Oparty na Uczeniu ze Wzmocnieniem i Algorytmie MCTS

### Projekt Deep Learning

---

## Streszczenie

Celem projektu jest stworzenie zaawansowanego systemu analizy szachowej opartego na uczeniu ze wzmocnieniem (Reinforcement Learning) oraz algorytmie Monte Carlo Tree Search (MCTS). System wykorzystuje ujednoliconą architekturę sieci neuronowej do klasyfikacji jakości ruchów oraz oceny pozycji poprzez wartość Value Head. Dzięki zastosowaniu mechanizmu *lookahead*, bot efektywnie przewiduje konsekwencje posunięć, eliminując potrzebę zewnętrznych silników szachowych. Klasyfikacja ruchów odbywa się poprzez obliczenie różnicy wartości pozycji przed i po wykonaniu ruchu, co pozwala na automatyczną ocenę jakości: Najlepszy, Wyśmienity, Dobry, Nieprecyzyjny, Błąd oraz Katastrofa.

---

## 1. Wstęp

### 1.1 Znaczenie Sztucznej Inteligencji w Szachach

Szachy od wieków stanowią wzorzec do testowania zdolności intelektualnych sztucznej inteligencji. Wczesne podejścia opierały się na algorytmach minimaksowych i przeszukiwaniu drzewa gry (alfa-beta pruning), co prowadziło do stworzenia silników takich jak Deep Blue czy Stockfish. Jednak podejścia te wymagały ręcznego zdefiniowania heurystyk oceny pozycji oraz funkcji ewaluacyjnych.

### 1.2 Ewolucja od Klasycznych Silników do AlphaZero

#### Wczesne rozwiązania (Deep Blue, Rybka, Stockfish):
- Drzewa przeszukiwań (minimax, alfa-beta pruning)
- Heurystyczne funkcje oceny pozycji
- Bazy wiedzy szachowej
- Ograniczone czasowo (~3-5 minut na ruch)

#### Nowoczesne podejścia (AlphaGo, AlphaZero):
- Sieci neuronowe do oceny pozycji (Value Head)
- Polityki sieci (Policy Head) do kierowania przeszukiwaniem
- Algorytm MCTS z priorytetami wyuczonymi przez sieć
- Uczenie ze wzmocnieniem poprzez grę samogry
- Skalowalne i niezależne od wiedzy szachowej

Projekt Chess Bot wdraża to nowoczesne podejście w skali edukacyjnej, łącząc uczenie nadzorowane (Supervised Learning) na grach Stockfish z uczeniem ze wzmocnieniem (Reinforcement Learning) poprzez samogry.

---

## 2. Cel Projektu

### 2.1 Problem Subiektywnej Oceny Jakości Ruchu

W tradycyjnych silnikach szachowych ocena ruchu ogranicza się do wartości liczbowej (np. +2.5 dla białych). Nie dostarcza to bezpośredniej informacji o *jakości* ruchu w kategoriach, które rozumie gracz szachowy (np. czy ruch był optymalny, czy stanowił błąd).

### 2.2 Cel Implementacji

Niniejszy projekt implementuje model, który:

1. **Ocenia pozycję** - Value Head przewiduje wartość od -1 (wygranie dla czarnych) do +1 (wygranie dla białych)
2. **Klasyfikuje ruch** - Na podstawie *delta oceny* (różnicy między wartością pozycji przed i po ruchu)
3. **Przewiduje politykę ruchów** - Policy Head dostarcza dystrybucję prawdopodobieństwa nad wszystkimi 4096 możliwymi ruchami
4. **Przyspieszenia MCTS** - Uwzględnia priorytety wyuczone przez sieć
5. **Eliminuje zależność** - Od zewnętrznych silników szachowych

---

## 3. Opis Danych

### 3.1 Źródło Danych

Projekt wykorzystuje bazę danych SQLite (`chess_bot.db`) zawierającą:
- Tabela `games` – metadane gier szachowych
- Tabela `moves` – ruchy z ocenami z Stockfisha
- Tabela `self_play_moves` – ruchy z samogier dla treningu RL

### 3.2 Przygotowanie Danych: Konwersja FEN na Tensor

#### Notacja FEN (Forsyth-Edwards Notation)

FEN to standardowa notacja opisująca pozycję szachownicy. Przykład:
```
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
```

#### Konwersja do Tensora

Każda pozycja FEN jest konwertowana na tensor 13-kanałowy o wymiarach `[13, 8, 8]`:

| Kanały | Liczba | Znaczenie |
|--------|--------|-----------|
| 0–5    | 6      | Figury białe (Pion, Skoczek, Słup, Wieża, Królowa, Król) |
| 6–11   | 6      | Figury czarne (pion, skoczek, słup, wieża, królowa, król) |
| 12     | 1      | Wskaźnik ruchu (1 = białe, 0 = czarne) |
| **Razem** | **13** | **Tensory wejściowe** |

### 3.3 Struktura Tabeli self_play_moves

```sql
CREATE TABLE self_play_moves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER,
    fen_before TEXT,              -- Pozycja przed ruchem
    move_uci TEXT,                -- Ruch w notacji UCI
    mcts_policy TEXT,             -- Polityka z MCTS (JSON)
    result_value REAL,            -- Ocena Value Head dla tej pozycji
    lookahead_depth INTEGER,      -- Głębokość lookahead
    future_moves TEXT,            -- Sekwencja przyszłych ruchów
    final_classification TEXT,    -- Końcowa klasa ruchu
    move_sequence_classes TEXT    -- Klasy sekwencji ruchów
);
```

### 3.4 Etapy Przygotowania Danych

1. **Ładowanie z bazy** – Pobranie ruchów z ocenami z gier Stockfish
2. **Obliczenie delta oceny** – Różnica między wartościami pozycji przed i po
3. **Przypisanie klasy** – Mapowanie delta oceny na jedną z 6 klas
4. **Normalizacja** – Skalowanie wartości do zakresu [-1, 1]
5. **Konwersja na tensory** – Transformacja FEN na 13-kanałowy tensor

---

## 4. Architektura Modelu

### 4.1 Przegląd Ogólny: Opcja A

Projekt implementuje *Opcję A*, gdzie klasyfikacja ruchu nie jest wprost przewidywana przez odrębną głowicę klasyfikacyjną, lecz obliczana poprzez **różnicę wartości pozycji** przed i po ruchu.

```
┌─────────────────────────────────────────────────────────┐
│        Zjednoczony Model Szachowy (Opcja A)              │
│  ┌───────────────────────────────────────────────────┐  │
│  │      Wspólny Trzon (ChessCoreNet)                  │  │
│  │  (Conv2D + Bloki Residualne)                       │  │
│  └─────────────┬───────────────────────────────────┘  │
│                │                                       │
│  ┌─────────────┴───────────────────────────────────┐  │
│  │  Głowica Value (-1 do 1)                         │  │
│  │  (Ocena pozycji)                                 │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │  Głowica Policy (Prawdopodobieństwa ruchów)   │  │
│  │  (4096 wymiarów dla wszystkich ruchów)        │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

Klasyfikacja obliczana jako:
Δ = |V(s) - V(s')|
gdzie s = pozycja przed, s' = pozycja po
```

### 4.2 ChessCoreNet – Wspólny Trzon

| Komponent | Parametry | Wyjście |
|-----------|-----------|---------|
| Wejście | 13 kanałów, 8×8 | [B, 13, 8, 8] |
| Conv2D (init) | kernel=3×3, out=128 | [B, 128, 8, 8] |
| BatchNorm | – | [B, 128, 8, 8] |
| Bloki Residualne | 6 × (Conv+BN+ReLU) | [B, 128, 8, 8] |
| **Wyjście** | – | [B, 128, 8, 8] |

#### Blok Residualny

Każdy blok residualny zawiera:
- Conv2D (3×3, 128→128)
- BatchNorm + ReLU
- Conv2D (3×3, 128→128)
- BatchNorm
- Połączenie rezydualnego: `output = ReLU(x + block(x))`

### 4.3 Głowica Value

Ocenia wartość pozycji w zakresie [-1, 1]:

| Warstwa | Parametry | Wyjście |
|---------|-----------|---------|
| Conv2D (reduce) | kernel=1×1, 128→32 | [B, 32, 8, 8] |
| BatchNorm | – | [B, 32, 8, 8] |
| Flatten | – | [B, 2048] |
| FC1 | 2048→256 | [B, 256] |
| ReLU | – | [B, 256] |
| FC2 | 256→1 | [B, 1] |
| Tanh | – | [B, 1] ∈ [-1, 1] |

### 4.4 Głowica Policy

Przewiduje rozkład prawdopodobieństwa nad 4096 możliwymi ruchami:

| Warstwa | Parametry | Wyjście |
|---------|-----------|---------|
| Conv2D (reduce) | kernel=1×1, 128→32 | [B, 32, 8, 8] |
| Flatten | – | [B, 2048] |
| FC1 | 2048→256 | [B, 256] |
| ReLU | – | [B, 256] |
| FC (logits) | 256→4096 | [B, 4096] |
| Softmax (inf.) | – | [B, 4096] (∑=1) |

---

## 5. Klasyfikacja Ruchów

### 5.1 Schemat Klasyfikacji

Ruchy są klasyfikowane na podstawie **delta oceny** – różnicy między wartościami Value Head dla pozycji przed i po ruchu.

| Klasa | Opis | Zakres Delta |
|-------|------|--------------|
| **Najlepszy** | Optymalny ruch | Δ ≤ 0.02 |
| **Wyśmienity** | Bardzo dobry ruch | 0.02 < Δ ≤ 0.07 |
| **Dobry** | Dobry ruch | 0.07 < Δ ≤ 0.15 |
| **Nieprecyzyjny** | Ruch słabszy | 0.15 < Δ ≤ 0.30 |
| **Błąd** | Znaczący błąd | 0.30 < Δ ≤ 0.55 |
| **Katastrofa** | Bardzo zły ruch | Δ > 0.55 |

### 5.2 Priorytet Ruchów w MCTS

Głowica Policy dostarcza rozkład prawdopodobieństwa dla każdego ruchu. Priorytety używane w MCTS są dostrajane na podstawie przewidywanej klasy:

| Klasa | Priorytet |
|-------|-----------|
| Najlepszy | 1.0 |
| Wyśmienity | 0.8 |
| Dobry | 0.5 |
| Nieprecyzyjny | 0.2 |
| Błąd | 0.05 |
| Katastrofa | 0.001 |

---

## 6. Algorytm MCTS

### 6.1 Monte Carlo Tree Search – Przegląd

MCTS to algorytm przeszukiwania drzewa gry, który łączy losowe symulacje (playouty) z budowaniem systematycznego drzewa przeszukiwań. W naszym projekcie, MCTS jest akcelerowany poprzez:
- **Value Head** – ocena pozycji bez głębokich symulacji
- **Policy Head** – kierowanie przeszukiwaniem poprzez priorytety

### 6.2 Składniki UnifiedMCTS

1. **Węzły drzewa (MCTSNode)** – reprezentują pozycje szachowe
2. **Policytyka (Priors)** – prawdopodobieństwa ruchów z Policy Head
3. **Ocena (Value)** – wartość z Value Head
4. **UCB Score** – równanie wyboru: Q(s,a) + U(s,a)

### 6.3 Etapy Algorytmu

1. **Selection** – Wybierz ścieżkę z korzenia maksymalizując UCB
2. **Expansion** – Jeśli węzeł nie rozwinięty, oceń wszystkie ruchy za pomocą sieci
3. **Evaluation** – Value Head oceni pozycję
4. **Backup** – Propaguj wynik w górę drzewa

### 6.4 UCB1 Score

Dla każdego ruchu, obliczane jest:

```
UCB(a) = Q(a) / N(a) + c * P(a) * √(N) / (1 + N(a))
```

gdzie:
- Q(a) – suma wartości dla akcji a
- N(a) – liczba wizyt dla akcji a
- P(a) – priorytet z Policy Head
- N – liczba wizyt węzła nadrzędnego
- c – współczynnik eksploracji (cpuct ≈ 2.0)

---

## 7. Proces Treningu

### 7.1 Trzy Tryby Treningu

#### 1. Tryb Nadzorowany (Supervised)

Trening na istniejących grach Stockfish z bazy danych:

```bash
python train_unified.py --mode supervised \
  --epochs 10 --batch-size 256 --lr 1e-3 \
  --alpha 0.5 --sample-rate 1.0
```

**Parametry:**
- `--epochs` – liczba epok treningowych
- `--batch-size` – rozmiar mini-batcha
- `--lr` – wskaźnik uczenia (learning rate)
- `--alpha` – waga straty Value (domyślnie 0.5)
- `--sample-rate` – ułamek danych do treningu (1.0 = wszystkie)

#### 2. Tryb Uczenia ze Wzmocnieniem (RL)

Generowanie samogier z MCTS i trening iteracyjny:

```bash
python train_unified.py --mode rl \
  --iterations 5 --games-per-iter 10 \
  --sims 800 --epochs 3
```

**Parametry:**
- `--iterations` – liczba iteracji RL
- `--games-per-iter` – liczba gier na iterację
- `--sims` – symulacje MCTS na pozycję
- `--keep-last-n` – zachowaj ostatnie N gier
- `--temperature` – temperatura eksploracji (domyślnie 1.2)

#### 3. Tryb Połączony (Combined)

Trening nadzorowany + uczenie ze wzmocnieniem:

```bash
python train_unified.py --mode combined \
  --epochs 5 --iterations 3
```

### 7.2 Funkcja Straty

Łączna strata jest kombinacją:

```
L = α * L_value + (1-α) * L_policy
```

gdzie:
- L_value = MSE(predicted_value, target_value) – strata wartości
- L_policy = CrossEntropy(policy_logits, target_policy) – strata polityki
- α – waga balansu (domyślnie 0.5)

### 7.3 Data Augmentation

System może generować nowe pozycje poprzez:
- **Obroty** – rotacje szachownicy (8-krotna symetria)
- **Odbicia** – lustrzane odbicia
- **Sekwencje przyszłości** – lookahead do przyszłych ruchów

---

## 8. API REST

Serwer FastAPI (`server.py`) udostępnia interfejs do interakcji z modelem:

### 8.1 Dostępne Endpointy

#### GET /
Zwraca informacje o API i dostępnych endpointach.

#### GET /analyze_move
Analizuje pojedynczy ruch i zwraca jego klasyfikację.

**Parametry:**
- `fen` (string) – pozycja w notacji FEN
- `move_san` (string) – ruch w notacji SAN (algebraiczna)

**Przykładowa odpowiedź:**
```json
{
  "nn_class": "Excellent",
  "ideal_class": "Best",
  "move_san": "e4",
  "move_uci": "e2e4"
}
```

#### GET /get_move
Zwraca najlepszy ruch bota dla danej pozycji używając MCTS.

**Parametry:**
- `fen` (string) – pozycja w notacji FEN
- `simulations` (int) – liczba symulacji MCTS (domyślnie 100)

**Przykładowa odpowiedź:**
```json
{
  "move_uci": "e2e4",
  "move_san": "e4",
  "bot_nn_class": "Excellent",
  "visit_counts": {"e2e4": 45, "g1f3": 30},
  "total_time": 0.234
}
```

#### GET /get_policy
Zwraca politykę ruchu ze zjednoczonego modelu (rozkład prawdopodobieństwa).

**Parametry:**
- `fen` (string) – pozycja w notacji FEN
- `top_k` (int) – liczba najlepszych ruchów (domyślnie 64)

**Przykładowa odpowiedź:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "policy": {
    "e7e5": 0.35,
    "g8f6": 0.28,
    "c7c6": 0.15
  },
  "total_moves": 64
}
```

---

## 9. Struktura Katalogów

```
chess_bot/
├── classifiers/
│   ├── classification_config.py      # Progi klasyfikacji
│   ├── move_classifier.py            # Logika klasyfikacji
│   └── self_play_dataset.py          # Dataset samogier
│
├── engine/
│   ├── unified_mcts.py               # Algorytm MCTS
│   └── rl_trainer.py                 # Pętla treningu RL
│
├── models/
│   ├── unified_chess_nets.py         # Architektura sieci
│   ├── weights_bot.pth               # Wagi modelu
│   └── weights_bot_copy.pth          # Zapas wag
│
├── parsers/
│   └── pgn_parser.py                 # Parser plików PGN
│
├── train_unified.py                  # Główny skrypt treningu
├── server.py                         # API FastAPI
├── test_inference.py                 # Testy
├── requirements.txt                  # Zależności
└── README.md
```

---

## 10. Literatura

1. **AlphaGo Zero / AlphaZero papers** – DeepMind (Silver et al., 2017)
   - `Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm`

2. **Monte Carlo Tree Search** – Kocsis et al., 2006
   - `Bandit Based Monte-Carlo Planning`

3. **PyTorch Documentation** – https://pytorch.org/docs/stable/

4. **python-chess Library** – https://python-chess.readthedocs.io/

5. **Reinforcement Learning: An Introduction** – Sutton & Barto (2018)

6. **Deep Learning** – Goodfellow, Bengio, Courville (2016)

---

## 11. Wymagania Techniczne

### 11.1 Zależności

- `torch>=2.0.0` – Framework deep learning
- `numpy>=1.21.0` – Obliczenia numeryczne
- `chess>=1.1.0` – Logika szachów
- `fastapi>=0.100.0` – Framework API
- `uvicorn>=0.22.0` – Serwer ASGI
- `tqdm>=4.65.0` – Pasek postępu
- `pandas>=1.3.0` – Analiza danych
- `matplotlib>=3.4.0` – Wizualizacja
- `seaborn>=0.11.0` – Zaawansowana wizualizacja
- `scikit-learn>=0.24.0` – Uczenie maszynowe

### 11.2 Instalacja

```bash
pip install -r requirements.txt
```

### 11.3 Uruchomienie

```bash
# Serwer API
python server.py

# Trening nadzorowany
python train_unified.py --mode supervised --epochs 10

# Trening RL
python train_unified.py --mode rl --iterations 5

# Testy
python test_inference.py
```

---

## 12. Wnioski

Projekt Chess Bot prezentuje zaawansowaną implementację zjednoczonego systemu analizy szachowej, łączącego:

✓ **Architektura Opcji A** – Klasyfikacja oparta na delta oceny zamiast dedykowanej głowicy  
✓ **MCTS z priority** – Przeszukiwanie drzewa gry przyspieszane przez Policy Head  
✓ **Uczenie multitaskowe** – Wspólny trzon optymalizowany jednocześnie dla Value i Policy  
✓ **Skalowalność** – System umożliwia szkolenie na dowolnie dużych zbiorach danych  
✓ **Praktyczne zastosowania** – REST API do integracji z innymi systemami  
✓ **Samowystarczalność** – Eliminuje zależność od zewnętrznych silników szachowych  

Podejście to, inspirowane sukcesami AlphaGo i AlphaZero, oferuje samodzielny system oparty na głębokim uczeniu (Deep Learning) i uczeniu ze wzmocnieniem (Reinforcement Learning), który efektywnie uczy się grać w szachy poprzez samogry i ocenę opartą na wartości pozycji.

---

*Wygenerowane: 2026-06-17*
