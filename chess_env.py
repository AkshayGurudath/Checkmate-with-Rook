import chess
import random
import numpy as np
import chess.svg
import chess.engine
import numpy as np

class ChessEnv:

    def __init__(self, engine_path, time_limit_for_engine=0.1, player_color = chess.WHITE, draw_penalty=-50):
        self.board = chess.Board()
        self.engine_path = engine_path
        self.time_limit_for_engine = time_limit_for_engine
        self.player_color = player_color
        self.draw_penalty = draw_penalty

        self.state_dimension = 6
        self.action_dimension = 36

    def _state_to_uci(self):
        pass

    def _uci_to_state(self):
        pass

    def _episode_ended(self):  # Returns boolean to check if the episode has ended or not
        if self.board.is_checkmate() or self.board.is_stalemate() or self.board.can_claim_fifty_moves() or \
                (self.board.has_insufficient_material(chess.WHITE) and self.board.has_insufficient_material(chess.BLACK)):
            return True
        else:
            return False

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

    # takes tuple (x,y) and converts to uci repr
    # Eg - (1,2) = 'a2'
    def _pos_to_uci(self, pos):
        x, y = pos # x, y are in the range 1 to 8
        if  x > 8 or x < 0 or y > 8 or y < 0:
            return ""
        uci_x = chr(x + 96) # 97 = 'a'
        uci_y = str(y)
        return uci_x + uci_y

    # takes uci representation for one position
    # and converts to (x,y)
    # Eg - 'a2' = (1,2)
    def _uci_to_pos(self, uci):
        uci_x, uci_y = uci
        x, y = ord(uci_x) - 96, int(uci_y) # 'a' = 97
        return (x,y)

    # Given a uci string, returns action index
    # Eg: input - "e1e2". Suppose king was at e1.
    # Then e1 to e2 is an action up. Therefore, the
    # action index is 1. The function returns 1.
    def _action_index_from_uci(self, move_uci):
        piece_map = self.board.piece_map()
        initial_pos = self._uci_to_pos(move_uci[0:2])
        piece = piece_map[ (initial_pos[1]-1)*8 + (initial_pos[0]-1) ]
        if piece.color != self.player_color:
            return -1
        next_pos = self._uci_to_pos(move_uci[2:])
        x,y = initial_pos
        nx, ny = next_pos
        dx, dy = nx - x, ny - y
        if str.lower(piece.symbol()) == 'k': # king move
            king_delta_action_dict = { (-1,1) : 0, (0, 1): 1, (1, 1): 2, (1, 0): 3, \
                                                        (1, -1): 4, (0, -1): 5, (-1, -1):6, (-1, 0): 7}
            action = king_delta_action_dict[(dx, dy)]
        elif str.lower(piece.symbol()) == 'r':
            map_to_index = list(range(7,0,-1))
            if dx == 0: # vertical move
                if dy < 0: # down
                    action = 7 + 14 + map_to_index[abs(dy)-1] # 0-7 - king, 14 - horizontal
                else:
                    action = 7 + 14 + 7 + abs(dy) # 0-7 - king, 14 - horizontal, 7 - down
            else: # horizontal move
                if dx < 0: # left
                    action = 7 + map_to_index[abs(dx) - 1] # 0-7 - king
                else:
                    action = 7 + 7 + abs(dx) # 0-7 - king, 7 - left
        return action

    # action is of the form -
    # [Kdiagupleft, Kup, Kdiagupright,
    #   Kright,
    #   Kdiagdownright, Kdown, Kdiagdownleft,
    #   Kleft,
    #   RookHorizontal: -7 to + 7 (excluding 0),
    #   RookVertical: -7 to +7 (excluding 0)]
    # Total dims - 8 + 14 + 14 = 36
    # Thus, input here is from 0 to 35
    # [K1..K8, -7..-1, 1..7, -7..-1,1..7]
    # [0...7, 8...14, 15...21, 22...28, 29...35]
    def _push_move(self, action):
        current_state = self._get_current_state()
        uci_from, uci_to = "", ""
        if action < 8: # king move
            king_position = tuple(current_state[0:2])
            uci_from = self._pos_to_uci(king_position)
            if action <= 2: # king is moving up
                uci_to = self._pos_to_uci( (king_position[0] + (action-1), king_position[1] + 1) )
            elif action == 3: # king moving right
                uci_to = self._pos_to_uci( (king_position[0] + 1, king_position[1]) )
            elif action <= 6:
                uci_to = self._pos_to_uci( (king_position[0] - (action-5), king_position[1] - 1) )
            else:
                uci_to = self._pos_to_uci( (king_position[0] - 1, king_position[1]) )
        else: # rook move
            rook_position = tuple(current_state[2:4])
            uci_from = self._pos_to_uci(rook_position)
            if action < 22: # rook horizontal move
                # get dx in range -7 to 7, excluding 0
                dx = (action - 15) if action <= 14 else (action - 14)
                uci_to = self._pos_to_uci( (rook_position[0] + dx, rook_position[1]))
            else: # rook vertical move
                # get dy in range -7 to 7, excluding 0
                dy = (action - 29) if action <= 28 else (action - 28)
                uci_to = self._pos_to_uci( (rook_position[0], rook_position[1] + dy) )

        if uci_to == "":
            raise ValueError("Illegal move - {} in state {} ".format(action, self._get_current_state()))
        move_uci_str = uci_from + uci_to
        move = chess.Move.from_uci(move_uci_str)
        if move not in self.board.legal_moves:
            raise ValueError("Illegal move - {} in state {} ".format(action, self._get_current_state()))
        self.board.push(move)
        return

    def _get_reward(self):
        if self._episode_ended():
            if self.board.result() == "1/2-1/2":
                return self.draw_penalty # penalize for draw
            else:
                return 0 # do nothing for victory
            # note - the player can never lose
        return -1 # give -1 for every step taken

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
            position = inv_piece_map.get(chess.Piece.from_symbol(piece), -1)
            if position != -1:
                state.append(( position % 8 ) + 1)  # x co-ordinate
                state.append(( position // 8 ) + 1)  # y co-ordinate

        if len(state) == 4: # rook has been taken
            x, y = state[2:4]
            state.extend( [x,y] ) # rook and king share same position now
        return np.asarray(state)

    # returns a boolean list of length 36
    def legal_moves(self):
        legal_actions = [False] * 36
        # get uci strings of all permissible moves
        legal_moves = list(map(lambda x: x.uci(), self.board.legal_moves))
        # get indices for permissible actions
        indices = list(map(lambda x : self._action_index_from_uci(x), legal_moves))
        # prune out all -1s, which correspond to engine's permissible moves
        indices = list(filter(lambda x: x != -1, indices))
        for i in indices:
            legal_actions[i] = True
        return legal_actions

    def engine_move(self):
        engine=chess.engine.SimpleEngine.popen_uci(self.engine_path)
        result = engine.play(self.board, chess.engine.Limit(time=self.time_limit_for_engine))
        self.board.push(result.move)
        engine.quit()


    def step(self, action):
        try:
            self._push_move(action)
        except ValueError:
            print("Invalid action")
            raise

        done = self._episode_ended()

        if not done:
            self.engine_move()
            done = self._episode_ended()

        reward = self._get_reward()
        next_state = self._get_current_state()

        return next_state, reward, done, 0

    def render(self):
        print(self.board)
