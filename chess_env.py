import chess
import random
import numpy as np
import chess.svg

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
        pass

    def _episode_ended(self):  # Returns boolean to check if the episode has ended or not
        if self.board.is_checkmate() or self.board.is_stalemate() or self.board.can_claim_fifty_moves() or \
                self.board.has_insufficient_material():
            return True
        else:
            return False

    def _legal_moves(self):
        pass

    def reset(self):
        chess.Board.clear_board(self.board)
        agent_rook = chess.Piece(chess.ROOK, self.player_color)
        agent_king = chess.Piece(chess.KING, self.player_color)
        opponent_king = chess.Piece(chess.KING, not self.player_color)
        # An empty board is not a valid board
        while not chess.Board.is_valid(self.board):  #This function checks if the board is valid or not. Considers checkmates and checks
            chess.Board.clear_board(self.board)
            agent_rook_pos, agent_king_pos, opponent_king_pos = random.sample(range(0, 64), 3)
            map_dict = {agent_rook_pos: agent_rook, agent_king_pos: agent_king, opponent_king_pos: opponent_king}
            chess.Board.set_piece_map(self.board, map_dict)  # position to piece mapping
            self.board.set_castling_fen("-")  # Sets castling rights to none
            self.board.turn = self.player_color
        if self.player_color == chess.BLACK:
            self.engine_move()

        return self._get_current_state()


    def step(self):
        pass

    def render(self):
        pass




