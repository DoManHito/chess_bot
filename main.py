#!/usr/bin/env python3
"""
Chess Bot - Main entry point with CLI interface.

This module provides a command-line interface for parsing PGN files,
classifying moves, and managing game statistics.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add current directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent))

# Import all modules
from parsers import PGNParser, ParsedGame
from classifiers import MoveClassifier, MoveClassificationResult, MoveData
from database import ChessDatabase, ParsedGame as DBParsedGame, MoveClassification


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def init_database(db_path: str = None) -> ChessDatabase:
    """Initialize the database.
    
    Args:
        db_path: Path to the database file. Defaults to chess_bot.db
        
    Returns:
        Initialized ChessDatabase instance
        
    Raises:
        Exception: If database initialization fails
    """
    try:
        db = ChessDatabase(db_path)
        logger.info("Database initialized successfully")
        return db
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def parse_pgn_file(
    file_path: str,
    db: ChessDatabase,
    classifier: MoveClassifier,
    progress_callback=None
) -> List[DBParsedGame]:
    """Parse a PGN file and classify all moves.
    
    Args:
        file_path: Path to the PGN file
        db: ChessDatabase instance for storage
        classifier: MoveClassifier instance for classification
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of parsed and saved games
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If parsing fails
    """
    parser = PGNParser()
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"PGN file not found: {file_path}")
    
    logger.info(f"Parsing PGN file: {file_path}")
    
    try:
        games = parser.parse_file(file_path)
    except Exception as e:
        logger.error(f"Failed to parse PGN file: {e}")
        raise
    
    if not games:
        logger.warning("No games found in the PGN file")
        return []
    
    logger.info(f"Found {len(games)} game(s) in the file")
    
    processed_games = []
    total_moves = 0
    
    for game in games:
        try:
            # Classify moves for this game
            move_data = []
            for move in game.moves:
                fen_before = move.fen_before or ''
                san = move.san or ''
                
                move_data.append(MoveData(
                    board_fen=fen_before,
                    move_san=san,
                    evaluation=move.evaluation if move.evaluation is not None else 0.0,
                    turn_num=move.turn_num or 0,
                    turn_label=move.turn_label or 'w'
                ))
            
            # Classify all moves
            classifications = classifier.classify_moves(move_data)

            DEFAULT_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

            fen_end = game.moves[-1].fen_after if game.moves else DEFAULT_START_FEN
            
            # Create game object for database
            db_game = DBParsedGame(
                id=0,  # Will be set by database
                white_player=game.headers.get('White', 'Unknown'),
                black_player=game.headers.get('Black', 'Unknown'),
                fen_start=game.headers.get('FEN', DEFAULT_START_FEN),
                fen_end=fen_end,
                result=game.headers.get('Result'),
                classification=game.headers.get('Evaluation', 'Unknown'),
                created_at=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            # Update game with classifications BEFORE saving to DB
            db_game.classification = classifications[0].classification if classifications else 'Unknown'
            
            # Save game and moves
            game_id = db.save_game(db_game)
            db_game.id = game_id # Текущий ID из базы
            db.save_moves_batch(game_id, classifications)
            
            processed_games.append(db_game)
            
            total_moves += len(classifications)
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(game.headers.get('Event', 'Unknown'), len(processed_games), len(games))
                
        except Exception as e:
            logger.error(f"Failed to process game {game.headers.get('Event', 'Unknown')}: {e}")
            continue
    
    logger.info(f"Processed {len(processed_games)} game(s) with {total_moves} move(s)")
    return processed_games


def get_player_statistics(player: str, db: ChessDatabase) -> dict:
    """Get statistics for a player.
    
    Args:
        player: Player name
        db: ChessDatabase instance
        
    Returns:
        Dictionary with player statistics
        
    Raises:
        Exception: If query fails
    """
    try:
        stats = db.get_player_statistics(player)
        logger.info(f"Statistics for {player}:")
        logger.info(f"  Total games: {stats['total_games']}")
        logger.info(f"  Wins: {stats['wins']}")
        logger.info(f"  Losses: {stats['losses']}")
        logger.info(f"  Draws: {stats['draws']}")
        logger.info(f"  Win rate: {stats['win_rate']:.2f}%")
        logger.info(f"  Classification breakdown: {stats['classification_breakdown']}")
        return stats
    except Exception as e:
        logger.error(f"Failed to get player statistics: {e}")
        raise


def list_all_games(db: ChessDatabase, limit: int = 100) -> List[DBParsedGame]:
    """List all games in the database.
    
    Args:
        db: ChessDatabase instance
        limit: Maximum number of games to return
        
    Returns:
        List of games
        
    Raises:
        Exception: If query fails
    """
    try:
        games = db.get_all_games(limit=limit)
        logger.info(f"Total games in database: {db.get_game_count()}")
        logger.info(f"Showing {len(games)} game(s):")
        
        for game in games:
            logger.info(f"  Game {game.id}: {game.white_player} vs {game.black_player} - {game.classification}")
        
        return games
    except Exception as e:
        logger.error(f"Failed to list games: {e}")
        raise


def classify_game(game_id: int, db: ChessDatabase) -> Optional[DBParsedGame]:
    """Get classifications for a specific game.
    
    Args:
        game_id: The game ID
        db: ChessDatabase instance
        
    Returns:
        The game with its moves, or None if not found
        
    Raises:
        Exception: If query fails
    """
    try:
        game = db.get_game_by_id(game_id)
        if not game:
            logger.error(f"Game not found: {game_id}")
            return None
        
        moves = db.get_moves_by_game(game_id)
        
        logger.info(f"Game {game_id}: {game.white_player} vs {game.black_player}")
        logger.info(f"Classification: {game.classification}")
        logger.info(f"Moves ({len(moves)}):")
        
        for move in moves:
            logger.info(f"  Move {move.move_number}: {move.move_san} - {move.classification}")
        
        return game
    except Exception as e:
        logger.error(f"Failed to get game classifications: {e}")
        raise


def process_multiple_files(
    file_paths: List[str],
    db: ChessDatabase,
    classifier: MoveClassifier,
    progress_callback=None
) -> int:
    """Process multiple PGN files.
    
    Args:
        file_paths: List of PGN file paths
        db: ChessDatabase instance
        classifier: MoveClassifier instance
        progress_callback: Optional callback for progress updates
        
    Returns:
        Total number of games processed
        
    Raises:
        Exception: If processing fails
    """
    total_games = 0
    processed_files = []
    
    for file_path in file_paths:
        try:
            games = parse_pgn_file(file_path, db, classifier, progress_callback)
            total_games += len(games)
            processed_files.append(file_path)
            logger.info(f"Processed: {file_path} - {len(games)} game(s)")
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            continue
    
    logger.info(f"Batch processing complete. Total games processed: {total_games}")
    return total_games


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Chess Bot - PGN parser and move classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py init                    # Initialize database
  python main.py parse games.pgn        # Parse and classify a PGN file
  python main.py stats PlayerName       # Get player statistics
  python main.py list                   # List all games
  python main.py classify <game_id>     # Show classifications for a game
  python main.py parse game1.pgn game2.pgn  # Batch process multiple files
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # === НОВАЯ КОМАНДА ДЛЯ ЦИКЛА ОБУЧЕНИЯ ===
    rl_parser = subparsers.add_parser('auto-train', help='Запустить зацикленный процесс Self-Play и обучения')
    rl_parser.add_argument('--iters', type=int, default=10, help='Количество глобальных итераций цикла (по умолчанию: 10)')
    rl_parser.add_argument('--games', type=int, default=20, help='Партий генерируется за 1 итерацию (по умолчанию: 20)')
    rl_parser.add_argument('--sims', type=int, default=60, help='Симуляций MCTS на каждый ход (по умолчанию: 60)')
    rl_parser.add_argument('--epochs', type=int, default=5, help='Эпох обучения нейросети за 1 итерацию (по умолчанию: 5)')
    rl_parser.add_argument('--keep-n', type=int, default=100, help='Сколько последних партий хранить в базе (по умолчанию: 100)')
    rl_parser.add_argument('--db-path', default='chess_bot.db', help='Путь к БД (по умолчанию: chess_bot.db)')
    rl_parser.add_argument('--workers', type=int, default=1, help='Количество процессов для параллельной генерации партий (по умолчанию: 1)')
    
    # init command
    init_parser = subparsers.add_parser('init', help='Initialize the database')
    init_parser.add_argument(
        '--db-path',
        default='chess_bot.db',
        help='Path to the database file (default: chess_bot.db)'
    )
    
    # parse command
    parse_parser = subparsers.add_parser('parse', help='Parse and classify a PGN file')
    parse_parser.add_argument(
        'file',
        nargs='+',
        help='PGN file(s) to parse'
    )
    parse_parser.add_argument(
        '--db-path',
        default='chess_bot.db',
        help='Path to the database file (default: chess_bot.db)'
    )
    parse_parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    # stats command
    stats_parser = subparsers.add_parser('stats', help='Get player statistics')
    stats_parser.add_argument(
        'player',
        help='Player name'
    )
    stats_parser.add_argument(
        '--db-path',
        default='chess_bot.db',
        help='Path to the database file (default: chess_bot.db)'
    )
    
    # list command
    list_parser = subparsers.add_parser('list', help='List all games')
    list_parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum number of games to show (default: 100)'
    )
    list_parser.add_argument(
        '--db-path',
        default='chess_bot.db',
        help='Path to the database file (default: chess_bot.db)'
    )
    
    # classify command
    classify_parser = subparsers.add_parser('classify', help='Show classifications for a game')
    classify_parser.add_argument(
        'game_id',
        type=int,
        help='Game ID'
    )
    classify_parser.add_argument(
        '--db-path',
        default='chess_bot.db',
        help='Path to the database file (default: chess_bot.db)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize database
    try:
        db = init_database(args.db_path)
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)
    
    # Initialize classifier
    classifier = MoveClassifier()
    
    # Execute command
    try:
        if args.command == 'init':
            # Already initialized above, just confirm
            logger.info("Database is already initialized")
            
        elif args.command == 'parse':
            # Process single or multiple files
            if args.quiet:
                # Suppress logging for quiet mode
                old_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.WARNING)
            
            games = process_multiple_files(
                args.file,
                db,
                classifier,
                progress_callback=lambda event, current, total: None
            )
            
            if not args.quiet:
                logging.getLogger().setLevel(logging.INFO)
            
            logger.info(f"Successfully processed {len(args.file)} file(s)")
            
        elif args.command == 'stats':
            stats = get_player_statistics(args.player, db)
            
        elif args.command == 'list':
            games = list_all_games(db, args.limit)
            
        elif args.command == 'classify':
            game = classify_game(args.game_id, db)
        
        elif args.command == 'auto-train':
            from engine.rl_trainer import run_continuous_loop
            run_continuous_loop(
                iterations=args.iters,
                games_per_iter=args.games,
                sims=args.sims,
                epochs=args.epochs,
                keep_last_n=args.keep_n,
                db_path=args.db_path,
                num_workers=args.workers,
            )
            
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    
    # Close database connection
    db.close()
    
    logger.info("Session complete")


if __name__ == '__main__':
    main()
