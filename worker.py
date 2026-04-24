import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import chess.pgn
import math
import random
import os
import time
import pickle
from collections import deque
from torch.optim.lr_scheduler import StepLR
import mcts_engine
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class AlphaZeroNet(nn.Module):
    def __init__(self, num_res_blocks=12):
        super(AlphaZeroNet, self).__init__()
        
        self.conv_input = nn.Sequential(
            nn.Conv2d(103, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(num_res_blocks)])
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4672)
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_input(x)
        x = self.res_blocks(x)
        return self.policy_head(x), self.value_head(x)

def history_to_tensor(history):
    """
    Konwertuje historię plansz (do 8 wstecz) do tensora [103, 8, 8].
    history: lista obiektów chess.Board. Ostatni element to aktualny stan.
    """
    tensor = np.zeros((103, 8, 8), dtype=np.float32)
    current_board = history[-1]
    is_black = (current_board.turn == chess.BLACK)
    
    
    for i, board in enumerate(reversed(history)):
        if i >= 8:
            break
        base_idx = i * 12 
        
        for pt in range(1, 7):
            my_pieces = board.pieces(pt, current_board.turn)
            for sq in my_pieces:
                r, c = divmod(sq, 8)
                if is_black: r = 7 - r
                tensor[base_idx + pt - 1, r, c] = 1.0
                
            op_pieces = board.pieces(pt, not current_board.turn)
            for sq in op_pieces:
                r, c = divmod(sq, 8)
                if is_black: r = 7 - r
                tensor[base_idx + 6 + pt - 1, r, c] = 1.0
                
    
    if is_black: tensor[96, :, :] = 1.0
    
    
    if current_board.has_kingside_castling_rights(current_board.turn): tensor[97, :, :] = 1.0
    if current_board.has_queenside_castling_rights(current_board.turn): tensor[98, :, :] = 1.0
    if current_board.has_kingside_castling_rights(not current_board.turn): tensor[99, :, :] = 1.0
    if current_board.has_queenside_castling_rights(not current_board.turn): tensor[100, :, :] = 1.0
    
    
    if current_board.ep_square is not None:
        r, c = divmod(current_board.ep_square, 8)
        if is_black: r = 7 - r
        tensor[101, r, c] = 1.0
        
    
    tensor[102, :, :] = current_board.halfmove_clock / 100.0
    
    return tensor

def move_to_index(move, turn):
    f_sq, t_sq = move.from_square, move.to_square
    if turn == chess.BLACK:
        f_sq, t_sq = chess.square_mirror(f_sq), chess.square_mirror(t_sq)
    if move.promotion is None or move.promotion == chess.QUEEN:
        return f_sq * 64 + t_sq
    promo_offset = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
    direction = (t_sq % 8) - (f_sq % 8) + 1 
    return 4096 + (f_sq % 8) * 9 + direction * 3 + promo_offset[move.promotion]

def batched_self_play(model, num_games=10, num_simulations=100):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    boards = [chess.Board() for _ in range(num_games)]
    roots = [mcts_engine.MCTSNode(1.0) for _ in range(num_games)]
    game_data = [[] for _ in range(num_games)]
    active_games = list(range(num_games))
    board_histories = [[boards[i].copy()] for i in range(num_games)]

    while active_games:
        for _ in range(num_simulations):
            leaves_to_eval = []
            histories_to_eval = []
            
            for i in active_games:
                node = roots[i]
                search_board = boards[i].copy()
                path = [node]

                while node.children:
                    move_uci = node.select_child_move(2.0) 
                    if move_uci is None: break
                    
                    search_board.push_uci(move_uci)
                    next_node = node.get_child(move_uci)
                    if not next_node: break
                    
                    node = next_node
                    path.append(node)
                    if node.is_terminal: break

                if node.is_terminal:
                    v = node.term_value
                    # Backpropagate
                    for n in reversed(path):
                        n.update(float(v))
                        v = -v
                    continue

                temp_history = board_histories[i][-7:] + [search_board]
                leaves_to_eval.append((i, node, path, search_board.fen()))
                histories_to_eval.append(temp_history)

            if histories_to_eval:
                tensors = [history_to_tensor(h) for h in histories_to_eval]
                bt = torch.from_numpy(np.stack(tensors)).to(device)
                
                with torch.inference_mode():
                    ps, vs = model(bt)
                ps, vs = ps.cpu().numpy(), vs.cpu().numpy()

                for idx, (game_idx, leaf_node, path, leaf_fen) in enumerate(leaves_to_eval):
                    policy_logits = ps[idx]
                    v = vs[idx].item()
                    
                    temp_board = chess.Board(leaf_fen)
                    p_dict = {m.uci(): float(policy_logits[move_to_index(m, temp_board.turn)]) 
                             for m in temp_board.legal_moves}
                    
                    leaf_node.expand(leaf_fen, p_dict)
                    
                    for n in reversed(path):
                        n.update(float(v))
                        v = -v

        new_active_games = []
        for i in active_games:
            root = roots[i]
            counts = []
            moves = []
            for m_uci in root.children_moves():
                child = root.get_child(m_uci)
                counts.append(child.n)
                moves.append(m_uci)
            
            if not counts:
                active_games.remove(i)
                continue

            probs = np.array(counts, dtype=np.float32)
            s = probs.sum()
            if s > 0:
                probs /= s
            else:
                probs = np.ones_like(probs) / len(probs)
            
            full_probs = np.zeros(4672, dtype=np.float32)
            for m_idx, m_uci in enumerate(moves):
                m_obj = chess.Move.from_uci(m_uci)
                full_probs[move_to_index(m_obj, boards[i].turn)] = probs[m_idx]
            
            game_data[i].append([history_to_tensor(board_histories[i]), full_probs, None])

            move_uci = random.choices(moves, weights=probs)[0]
            move = chess.Move.from_uci(move_uci)
            
            boards[i].push(move)
            board_histories[i].append(boards[i].copy())
            roots[i] = root.get_child(move_uci)

            if boards[i].is_game_over():
                result = boards[i].result()
                if result == "1-0":
                    final_reward = 1.0
                elif result == "0-1":
                    final_reward = -1.0
                else:
                    final_reward = 0.0
                
                for step in game_data[i]:
                    step[2] = float(final_reward)
                
                active_games.remove(i)
            else:
                new_active_games.append(i)

        
        active_games = new_active_games

    final_samples = []
    for g in game_data:
        final_samples.extend(g)
    return final_samples