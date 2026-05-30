import chess
from classifiers.move_classifier import MoveClassifier

def test_nuances():
    print("Загрузка классификатора для сложного тактического теста...")
    classifier = MoveClassifier()
    
    if not classifier.has_weights:
        print("Ошибка: Веса не найдены!")
        return

    def print_res(title, move_san, res):
        print(f"\n--- {title} ---")
        print(f"Ход: {move_san}")
        print(f"Вердикт сети: {res.classification} (Уверенность: {res.confidence*100:.2f}%)")

    # =========================================================================
    # ТЕСТ 1: Позиционная неточность (Inaccuracy) в дебюте
    # Сицилианская защита. Вместо развития коней или d6, черные делают пассивный ход h6.
    # Движки обычно считают это потерей темпа (Inaccuracy).
    # =========================================================================
    board1 = chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
    res1 = classifier.classify_move(board1.fen(), "h6", evaluation=-0.4, turn_num=2, turn_label="Black")
    print_res("Тест 1: Пассивный ход h6 в Сицилианке (Inaccuracy)", "h6", res1)

    # =========================================================================
    # ТЕСТ 2: Грубая дебютная ошибка (Mistake) — Защита Дамиано
    # 1. e4 e5 2. Nf3 f6? Ход f6 жестко ослабляет короля, это известная ошибка.
    # =========================================================================
    board2 = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
    res2 = classifier.classify_move(board2.fen(), "f6", evaluation=-1.5, turn_num=2, turn_label="Black")
    print_res("Тест 2: Защита Дамиано 2... f6 (Mistake)", "f6", res2)

    # =========================================================================
    # ТЕСТ 3: Глубокая тактическая ловушка (Скрытый зевок / Blunder)
    # Позиция из варианта Кембридж-Спрингс. Черная пешка бьет на c4 (dxc4).
    # Внешне ход кажется логичным (взятие), но тактически белые форсированно выигрывают фигуру.
    # =========================================================================
    board3 = chess.Board("r1bqkb1r/pppn1ppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 4 5")
    # Моделируем ход dxc4 для черных (хотя сейчас ход белых по FEN, для теста переключим context)
    board3.turn = chess.BLACK
    res3 = classifier.classify_move(board3.fen(), "dxc4", evaluation=-2.1, turn_num=5, turn_label="Black")
    print_res("Тест 3: Взятие dxc4, ведущее к тактическому проигрышу (Blunder)", "dxc4", res3)

    # =========================================================================
    # ТЕСТ 4: Высокотеоретический сильный ход (Best / Excellent)
    # Сицилианская защита, Вариант Найдорфа (5... a6). Сверхсложный позиционный ход.
    # =========================================================================
    board4 = chess.Board("rnbqkbnr/pp2pppp/3p4/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5")
    res4 = classifier.classify_move(board4.fen(), "a6", evaluation=0.3, turn_num=5, turn_label="Black")
    print_res("Тест 4: Теория Найдорфа 5... a6 (Best / Excellent)", "a6", res4)

if __name__ == "__main__":
    test_nuances()