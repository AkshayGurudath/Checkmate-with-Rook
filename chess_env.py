import chess
import numpy as np

class ChessEnv:

    def __init__(self):
        pass

    def _state_to_uci(self):
        pass

    def _uci_to_state(self):
        pass

    def _push_move(self):
        pass

    def _get_reward(self):
        pass

    def _get_current_state(self, board):
        state = []
        pieces = []
        piece_map = board.piece_map()
        inv_piece_map = {val: key for key, val in piece_map.items()}
        if self.player_color == chess.WHITE:
            pieces.extend(['K', 'R', 'k'])
        else:
            pieces.extend(['k', 'r', 'K'])
        
        for piece in pieces:
            position = inv_piece_map.get(chess.Piece.from_symbol(piece), "")
            if position:
                state.append(( position // 8 ) + 1)  # x co-ordinate
                state.append(( position % 8 ) + 1)  # y co-ordinate

        return np.asarray(state)

    def _episode_ended(self):
        pass

    def _legal_moves(self):
        pass

    def reset(self):
        pass

    def step(self):
        pass

    def render(self):
        pass