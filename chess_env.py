import chess
import chess.engine
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

    def _get_current_state(self):
        state = []
        pieces = []
        piece_map = self.board.piece_map()
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

    def engine_move(self):
        result = self.engine.play(board, chess.engine.Limit(time=self.time_linit_for_engine))
        self.board.push(result.move)

    def step(self, action):

        try:
            self._push_move(action)
        except ValueError:
            raise

        done = self._episode_ended()

        if not done:
            self.engine_move()
            done = self._episode_ended()

        reward = self._get_reward()
        next_state = self._get_current_state()

        return next_state, reward, done, 0

    def render(self):
        pass