"""Tests for PGN parsing functionality."""

import unittest
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.pgn_parser import PGNParser, ParsedGame, ParsedMove


class TestPGNParser(unittest.TestCase):
    """Test cases for PGN parser."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = PGNParser()

    def test_parse_simple_game(self):
        """Test parsing a simple game."""
        pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Bb7 10. d4 exd4 11. cxd4 Na5 12. Bc2 c5 13. d5 Nb7 14. Nbd2 c4 15. b3 cxb3 16. axb3 Nc5 17. Nf1 Rc8 18. Be3 Qd7 19. Qd2 Nxe4 20. Bxe4 d5 21. exd5 Bxd5 22. Qd3 Bxe4 23. Nxe4 Nxe4 24. Qxe4 Rxc4 25. Qxe7+ Kh8 26. Qxd7 Rxc2 27. Qxd5 Rc1+ 28. Kh2 Rxe1 29. Qxe2 Qxe2 30. Rxe2 Bf6 31. Rxf6+ gxf6 1-0"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 1)
        game = games[0]
        self.assertEqual(game.headers['Event'], 'Test Game')
        self.assertEqual(game.headers['White'], 'Player A')
        self.assertEqual(game.headers['Black'], 'Player B')
        self.assertEqual(game.result, '1-0')

    def test_parse_game_with_evaluations(self):
        """Test parsing a game with evaluation annotations."""
        pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 {+0.55} 2. Nf3 Nc6 {+0.30} 3. Bb5 a6 {+0.15} 4. Ba4 Nf6 {+0.07} 5. O-O Be7 {+0.03} 6. Re1 b5 {+0.01} 7. Bb3 d6 {0.00} 8. c3 O-O {0.00} 9. h3 Bb7 {0.00} 10. d4 exd4 {0.00} 11. cxd4 Na5 {0.00} 12. Bc2 c5 {0.00} 13. d5 Nb7 {0.00} 14. Nbd2 c4 {0.00} 15. b3 cxb3 {0.00} 16. axb3 Nc5 {0.00} 17. Nf1 Rc8 {0.00} 18. Be3 Qd7 {0.00} 19. Qd2 Nxe4 {0.00} 20. Bxe4 d5 {0.00} 21. exd5 Bxd5 {0.00} 22. Qd3 Bxe4 {0.00} 23. Nxe4 Nxe4 {0.00} 24. Qxe4 Rxc4 {0.00} 25. Qxe7+ Kh8 {0.00} 26. Qxd7 Rxc2 {0.00} 27. Qxd5 Rc1+ {0.00} 28. Kh2 Rxe1 {0.00} 29. Qxe2 Qxe2 {0.00} 30. Rxe2 Bf6 {0.00} 31. Rxf6+ gxf6 {0.00} 1-0"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 1)
        game = games[0]
        self.assertEqual(len(game.moves), 31)
        
        # Check that evaluations are parsed
        move_with_eval = game.moves[0]
        self.assertIsNotNone(move_with_eval.evaluation)
        self.assertEqual(move_with_eval.evaluation, 0.55)

    def test_parse_game_without_evaluations(self):
        """Test parsing a game without evaluation annotations."""
        pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[Result "1/2-1/2"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Bb7 10. d4 exd4 11. cxd4 Na5 12. Bc2 c5 13. d5 Nb7 14. Nbd2 c4 15. b3 cxb3 16. axb3 Nc5 17. Nf1 Rc8 18. Be3 Qd7 19. Qd2 Nxe4 20. Bxe4 d5 21. exd5 Bxd5 22. Qd3 Bxe4 23. Nxe4 Nxe4 24. Qxe4 Rxc4 25. Qxe7+ Kh8 26. Qxd7 Rxc2 27. Qxd5 Rc1+ 28. Kh2 Rxe1 29. Qxe2 Qxe2 30. Rxe2 Bf6 31. Rxf6+ gxf6 1/2-1/2"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 1)
        game = games[0]
        self.assertEqual(len(game.moves), 31)
        
        # Check that all moves have None evaluation
        for move in game.moves:
            self.assertIsNone(move.evaluation)

    def test_parse_multiple_games(self):
        """Test parsing multiple games in one file."""
        pgn_content = """[Event "Game 1"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Bb7 10. d4 exd4 11. cxd4 Na5 12. Bc2 c5 13. d5 Nb7 14. Nbd2 c4 15. b3 cxb3 16. axb3 Nc5 17. Nf1 Rc8 18. Be3 Qd7 19. Qd2 Nxe4 20. Bxe4 d5 21. exd5 Bxd5 22. Qd3 Bxe4 23. Nxe4 Nxe4 24. Qxe4 Rxc4 25. Qxe7+ Kh8 26. Qxd7 Rxc2 27. Qxd5 Rc1+ 28. Kh2 Rxe1 29. Qxe2 Qxe2 30. Rxe2 Bf6 31. Rxf6+ gxf6 1-0

[Event "Game 2"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player C"]
[Black "Player D"]
[Result "0-1"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Rc1 c6 8. Bd3 b6 9. cxd5 exd5 10. Bb5 Bb7 11. O-O a6 12. Bxc6 Qxc6 13. Qc2 Rfc8 14. Qb3 Qxb3 15. axb3 Bb4 16. Rfc1 Ba5 17. b4 Bb6 18. Ne5 Nxe5 19. dxe5 Bc5 20. Rxc8 Rxc8 21. Rxc8+ Bxc8 22. Bc4 Bxc4 23. Nxc4 d4 24. exd6 cxd6 25. Nxb6 axb6 26. Bb3 Qd7 27. Bxd5 Qxd5 28. Rxd5 Rxc4 29. Rxd4 Rxd4 30. Bxd4 b5 31. Bc3 b4 32. Bd2 b3 33. Bc1 b2 34. Bb2 1/2-1/2

[Event "Game 3"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player E"]
[Black "Player F"]
[Result "1/2-1/2"]

1. Nf3 Nf6 2. g3 d5 3. Bg2 g6 4. O-O Bg7 5. d3 O-O 6. Nbd2 c5 7. e4 dxe4 8. dxe4 Nxe4 9. Nxe4 Bxe4 10. Re1 Nc6 11. Qd2 Qa5 12. Bg5 h6 13. Bh4 Qd8 14. Qe2 Rad8 15. Rad1 Bf5 16. Bxf5 hxg5 17. Qe3 Qf6 18. Qxf6 Rxf6 19. Ne2 e6 20. Nf4 Bf8 21. Nd3 Nd4 22. Nxe6 Nxe6 23. Bf3 Nd4 24. Bxd5 Nxc2 25. Rxe6 Qf8 26. Rxe7 Qxe7 27. Bxe7 Rxe7 28. Rxe7 1/2-1/2"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 3)
        
        # Verify each game
        self.assertEqual(games[0].headers['Event'], 'Game 1')
        self.assertEqual(games[1].headers['Event'], 'Game 2')
        self.assertEqual(games[2].headers['Event'], 'Game 3')

    def test_parse_game_with_negative_evaluations(self):
        """Test parsing a game with negative evaluation annotations."""
        pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[Result "0-1"]

1. e4 e5 {+0.55} 2. Nf3 Nc6 {+0.30} 3. Bb5 a6 {+0.15} 4. Ba4 Nf6 {+0.07} 5. O-O Be7 {+0.03} 6. Re1 b5 {+0.01} 7. Bb3 d6 {-0.02} 8. c3 O-O {-0.05} 9. h3 Bb7 {-0.10} 10. d4 exd4 {-0.15} 11. cxd4 Na5 {-0.20} 12. Bc2 c5 {-0.25} 13. d5 Nb7 {-0.30} 14. Nbd2 c4 {-0.35} 15. b3 cxb3 {-0.40} 16. axb3 Nc5 {-0.45} 17. Nf1 Rc8 {-0.50} 18. Be3 Qd7 {-0.55} 19. Qd2 Nxe4 {-0.60} 20. Bxe4 d5 {-0.65} 21. exd5 Bxd5 {-0.70} 22. Qd3 Bxe4 {-0.75} 23. Nxe4 Nxe4 {-0.80} 24. Qxe4 Rxc4 {-0.85} 25. Qxe7+ Kh8 {-0.90} 26. Qxd7 Rxc2 {-0.95} 27. Qxd5 Rc1+ {-1.00} 28. Kh2 Rxe1 {-1.05} 29. Qxe2 Qxe2 {-1.10} 30. Rxe2 Bf6 {-1.15} 31. Rxf6+ gxf6 {-1.20} 0-1"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 1)
        game = games[0]
        
        # Check that negative evaluations are parsed correctly
        move_with_negative_eval = game.moves[6]
        self.assertIsNotNone(move_with_negative_eval.evaluation)
        self.assertEqual(move_with_negative_eval.evaluation, -0.02)

    def test_parse_game_with_bracketed_evaluations(self):
        """Test parsing game with bracketed evaluation format."""
        pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 {+0.50} e5 {+0.45} 2. Nf3 {+0.40} Nc6 {+0.35} 3. Bb5 {+0.30} a6 {+0.25} 4. Ba4 {+0.20} Nf6 {+0.15} 5. O-O {+0.10} Be7 {+0.05} 6. Re1 {+0.03} b5 {+0.01} 7. Bb3 {0.00} d6 {0.00} 8. c3 O-O {0.00} 9. h3 Bb7 {0.00} 10. d4 exd4 {0.00} 11. cxd4 Na5 {0.00} 12. Bc2 c5 {0.00} 13. d5 Nb7 {0.00} 14. Nbd2 c4 {0.00} 15. b3 cxb3 {0.00} 16. axb3 Nc5 {0.00} 17. Nf1 Rc8 {0.00} 18. Be3 Qd7 {0.00} 19. Qd2 Nxe4 {0.00} 20. Bxe4 d5 {0.00} 21. exd5 Bxd5 {0.00} 22. Qd3 Bxe4 {0.00} 23. Nxe4 Nxe4 {0.00} 24. Qxe4 Rxc4 {0.00} 25. Qxe7+ Kh8 {0.00} 26. Qxd7 Rxc2 {0.00} 27. Qxd5 Rc1+ {0.00} 28. Kh2 Rxe1 {0.00} 29. Qxe2 Qxe2 {0.00} 30. Rxe2 Bf6 {0.00} 31. Rxf6+ gxf6 {0.00} 1-0"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 1)
        game = games[0]
        
        # Check that bracketed evaluations are parsed
        self.assertEqual(game.moves[0].evaluation, 0.50)
        self.assertEqual(game.moves[1].evaluation, 0.45)

    def test_parse_game_with_unbracketed_evaluations(self):
        """Test parsing game with unbracketed evaluation format."""
        pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 +0.50 e5 +0.45 2. Nf3 +0.40 Nc6 +0.35 3. Bb5 +0.30 a6 +0.25 4. Ba4 +0.20 Nf6 +0.15 5. O-O +0.10 Be7 +0.05 6. Re1 +0.03 b5 +0.01 7. Bb3 0.00 d6 0.00 8. c3 O-O 0.00 9. h3 Bb7 0.00 10. d4 exd4 0.00 11. cxd4 Na5 0.00 12. Bc2 c5 0.00 13. d5 Nb7 0.00 14. Nbd2 c4 0.00 15. b3 cxb3 0.00 16. axb3 Nc5 0.00 17. Nf1 Rc8 0.00 18. Be3 Qd7 0.00 19. Qd2 Nxe4 0.00 20. Bxe4 d5 0.00 21. exd5 Bxd5 0.00 22. Qd3 Bxe4 0.00 23. Nxe4 Nxe4 0.00 24. Qxe4 Rxc4 0.00 25. Qxe7+ Kh8 0.00 26. Qxd7 Rxc2 0.00 27. Qxd5 Rc1+ 0.00 28. Kh2 Rxe1 0.00 29. Qxe2 Qxe2 0.00 30. Rxe2 Bf6 0.00 31. Rxf6+ gxf6 0.00 1-0"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 1)
        game = games[0]
        
        # Check that unbracketed evaluations are parsed
        self.assertEqual(game.moves[0].evaluation, 0.50)
        self.assertEqual(game.moves[1].evaluation, 0.45)

    def test_parse_game_with_fen_headers(self):
        """Test parsing a game with FEN headers."""
        pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[FEN "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e6 0 1"]
[Result "1-0"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Bg5 e6 7. f4 Be7 8. Qf3 Qc7 9. O-O-O Nbd7 10. e5 dxe5 11. fxe5 Bc5 12. Bxc7 Nxc7 13. Qf4 O-O 14. Qxf7+ Kh8 15. Qxf8+ Rxf8 16. Bxf8 1-0"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 1)
        game = games[0]
        self.assertIn('FEN', game.headers)

    def test_parse_game_with_comments(self):
        """Test parsing a game with comments."""
        pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 {A classic opening} 2. Nf3 Nc6 {Black develops} 3. Bb5 a6 {Black challenges the bishop} 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Bb7 10. d4 exd4 11. cxd4 Na5 12. Bc2 c5 13. d5 Nb7 14. Nbd2 c4 15. b3 cxb3 16. axb3 Nc5 17. Nf1 Rc8 18. Be3 Qd7 19. Qd2 Nxe4 20. Bxe4 d5 21. exd5 Bxd5 22. Qd3 Bxe4 23. Nxe4 Nxe4 24. Qxe4 Rxc4 25. Qxe7+ Kh8 26. Qxd7 Rxc2 27. Qxd5 Rc1+ 28. Kh2 Rxe1 29. Qxe2 Qxe2 30. Rxe2 Bf6 31. Rxf6+ gxf6 1-0"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 1)
        game = games[0]
        self.assertEqual(len(game.moves), 31)

    def test_parse_game_with_variations(self):
        """Test parsing a game with variations."""
        pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Bb7 10. d4 exd4 11. cxd4 Na5 12. Bc2 c5 13. d5 Nb7 14. Nbd2 c4 15. b3 cxb3 16. axb3 Nc5 17. Nf1 Rc8 18. Be3 Qd7 19. Qd2 Nxe4 20. Bxe4 d5 21. exd5 Bxd5 22. Qd3 Bxe4 23. Nxe4 Nxe4 24. Qxe4 Rxc4 25. Qxe7+ Kh8 26. Qxd7 Rxc2 27. Qxd5 Rc1+ 28. Kh2 Rxe1 29. Qxe2 Qxe2 30. Rxe2 Bf6 31. Rxf6+ gxf6 1-0"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 1)
        game = games[0]
        self.assertEqual(len(game.moves), 31)

    def test_parse_empty_pgn(self):
        """Test parsing an empty PGN."""
        pgn_content = ""
        
        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 0)

    def test_parse_pgn_with_only_headers(self):
        """Test parsing a PGN with only headers and no moves."""
        pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[Result "1/2-1/2"]"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 0)

    def test_parse_pgn_with_white_only_moves(self):
        """Test parsing a PGN with only white moves."""
        pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Bb7 10. d4 exd4 11. cxd4 Na5 12. Bc2 c5 13. d5 Nb7 14. Nbd2 c4 15. b3 cxb3 16. axb3 Nc5 17. Nf1 Rc8 18. Be3 Qd7 19. Qd2 Nxe4 20. Bxe4 d5 21. exd5 Bxd5 22. Qd3 Bxe4 23. Nxe4 Nxe4 24. Qxe4 Rxc4 25. Qxe7+ Kh8 26. Qxd7 Rxc2 27. Qxd5 Rc1+ 28. Kh2 Rxe1 29. Qxe2 Qxe2 30. Rxe2 Bf6 31. Rxf6+ gxf6 1-0"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 1)
        game = games[0]
        self.assertEqual(len(game.moves), 31)

    def test_parse_pgn_with_castling(self):
        """Test parsing a PGN with castling moves."""
        pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Bb7 10. d4 exd4 11. cxd4 Na5 12. Bc2 c5 13. d5 Nb7 14. Nbd2 c4 15. b3 cxb3 16. axb3 Nc5 17. Nf1 Rc8 18. Be3 Qd7 19. Qd2 Nxe4 20. Bxe4 d5 21. exd5 Bxd5 22. Qd3 Bxe4 23. Nxe4 Nxe4 24. Qxe4 Rxc4 25. Qxe7+ Kh8 26. Qxd7 Rxc2 27. Qxd5 Rc1+ 28. Kh2 Rxe1 29. Qxe2 Qxe2 30. Rxe2 Bf6 31. Rxf6+ gxf6 1-0"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 1)
        game = games[0]
        
        # Check that castling moves are parsed correctly
        self.assertEqual(game.moves[4].san, 'O-O')
        self.assertEqual(game.moves[7].san, 'O-O')

    def test_parse_pgn_with_promotion(self):
        """Test parsing a PGN with promotion moves."""
        pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Bb7 10. d4 exd4 11. cxd4 Na5 12. Bc2 c5 13. d5 Nb7 14. Nbd2 c4 15. b3 cxb3 16. axb3 Nc5 17. Nf1 Rc8 18. Be3 Qd7 19. Qd2 Nxe4 20. Bxe4 d5 21. exd5 Bxd5 22. Qd3 Bxe4 23. Nxe4 Nxe4 24. Qxe4 Rxc4 25. Qxe7+ Kh8 26. Qxd7 Rxc2 27. Qxd5 Rc1+ 28. Kh2 Rxe1 29. Qxe2 Qxe2 30. Rxe2 Bf6 31. Rxf6+ gxf6 32. e6 f5 33. exf7+ Kxf7 34. e7+ Ke6 35. e8=Q+ Kd5 36. Qf7+ Ke4 37. Qe6+ Kd3 38. Qe3+ Kc2 39. Qe2+ Kb1 40. Qe1+ Ka2 41. Qe2+ Kb1 42. Qe1+ Ka2 43. Qe2+ Kb1 44. Qe1+ Ka2 45. Qe2+ Kb1 46. Qe1+ Ka2 47. Qe2+ Kb1 48. Qe1+ Ka2 49. Qe2+ Kb1 50. Qe1+ Ka2 51. Qe2+ Kb1 52. Qe1+ Ka2 53. Qe2+ Kb1 54. Qe1+ Ka2 55. Qe2+ Kb1 56. Qe1+ Ka2 57. Qe2+ Kb1 58. Qe1+ Ka2 59. Qe2+ Kb1 60. Qe1# 1-0"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 1)
        game = games[0]
        
        # Check that promotion move is parsed correctly
        self.assertEqual(game.moves[34].san, 'e8=Q+')

    def test_parse_pgn_with_en_passant(self):
        """Test parsing a PGN with en passant moves."""
        pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Bg5 e6 7. f4 Be7 8. Qf3 Qc7 9. O-O-O Nbd7 10. e5 dxe5 11. fxe5 Bc5 12. Bxc7 Nxc7 13. Qf4 O-O 14. Qxf7+ Kh8 15. Qxf8+ Rxf8 16. Bxf8 1-0"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 1)
        game = games[0]
        self.assertEqual(len(game.moves), 16)

    def test_parse_pgn_with_oh_oh(self):
        """Test parsing a PGN with oh oh notation."""
        pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2026.05.27"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Bb7 10. d4 exd4 11. cxd4 Na5 12. Bc2 c5 13. d5 Nb7 14. Nbd2 c4 15. b3 cxb3 16. axb3 Nc5 17. Nf1 Rc8 18. Be3 Qd7 19. Qd2 Nxe4 20. Bxe4 d5 21. exd5 Bxd5 22. Qd3 Bxe4 23. Nxe4 Nxe4 24. Qxe4 Rxc4 25. Qxe7+ Kh8 26. Qxd7 Rxc2 27. Qxd5 Rc1+ 28. Kh2 Rxe1 29. Qxe2 Qxe2 30. Rxe2 Bf6 31. Rxf6+ gxf6 32. e6 f5 33. exf7+ Kxf7 34. e7+ Ke6 35. e8=Q+ Kd5 36. Qf7+ Ke4 37. Qe6+ Kd3 38. Qe3+ Kc2 39. Qe2+ Kb1 40. Qe1+ Ka2 41. Qe2+ Kb1 42. Qe1+ Ka2 43. Qe2+ Kb1 44. Qe1+ Ka2 45. Qe2+ Kb1 46. Qe1+ Ka2 47. Qe2+ Kb1 48. Qe1+ Ka2 49. Qe2+ Kb1 50. Qe1+ Ka2 51. Qe2+ Kb1 52. Qe1+ Ka2 53. Qe2+ Kb1 54. Qe1+ Ka2 55. Qe2+ Kb1 56. Qe1+ Ka2 57. Qe2+ Kb1 58. Qe1+ Ka2 59. Qe2+ Kb1 60. Qe1# 1-0"""

        games = self.parser.parse_string(pgn_content)
        
        self.assertEqual(len(games), 1)
        game = games[0]
        self.assertEqual(len(game.moves), 60)


if __name__ == '__main__':
    unittest.main()
