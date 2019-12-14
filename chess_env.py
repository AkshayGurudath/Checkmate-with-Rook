import chess

class ChessEnv:

    def __init__(self, engine_path, time_limit_for_engine=0.1, player_color = chess.BLACK, draw_penalty=-10):
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.time_limit_for_engine = time_limit_for_engine
        self.player_color = player_color
        self.draw_penalty = draw_penalty
        pass

    def _state_to_uci(self):
        pass

    def _uci_to_state(self):
        pass

    # takes tuple (x,y) and converts to uci repr
    # Eg - (1,2) = 'a2'
    def _pos_to_uci(self, pos):
        x, y = pos # x, y are in the range 1 to 8
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

    def _action_index_from_uci(self, move_uci):
        piece_map = self.board.piece_map()
        initial_pos = self._uci_to_pos(move_uci[0:2])
        piece = piece_map[ initial_pos[0]*8 + initial_pos[y] ]
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
            if dx == 0: # vertical move
                if dy < 0: # down
                    action = 8 + 15 + abs(dy) # 8 - king, 15 - horizontal
                else:
                    action = 8 + 15 + 15 + abs(dy) # 8 - king, 15 - horizontal, 7 - down
            else: # horizontal move
                if dx < 0: # left
                    action = 8 + abs(dx) # 8 - king
                else:
                    action = 8 + 7 + abs(dx) # 8 - king
        return action

    # action is of the form -
    # [Kdiagupleft, Kup, Kdiagupright,
    #   Kright,
    #   Kdiagdownright, Kdown, Kdiagdownleft,
    #   Kleft,
    #   RookHorizontal: -7 to + 7 (excluding 0),
    #   RookVertical: -7 to +7 (excluding 0)]
    # Total dims - 8 + 14 + 14 = 36
    # Thus, input here is from 0 to 35 #TODO: correct
    # [K1..K8, -7..-1, 1..7, -7..-1,1..7]
    # [0...8, 9...15, 16...22, 23...29, 30...36]
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
                action -= 15 # this renormalizes action to range -7 to +7
                uci_to = self._pos_to_uci( (rook_position[0] + action, rook_position[1]))
            else: # rook vertical move
                action -= 28 # this renormalizes action to range -7 to +7
                uci_to = self._pos_to_uci( (rook_position[0], rook_position[1] + action) )

        move_uci_str = uci_from + uci_to
        move = chess.Move.from_uci(move_uci_str)
        if move not in self.board.legal_moves:
            raise ValueError("Illegal move")
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
        pass

    def _episode_ended(self):
        pass

    # returns a boolean list of length 36
    def legal_moves(self):
        legal_actions = [False] * 36
        # get uci strings of all permissible moves
        legal_moves = map(lambda x: x.uci(), self.board.legal_moves)
        # get indices for permissible actions
        indices = map(lambda x : self._action_index_from_uci(), legal_moves)
        # prune out all -1s, which correspond to engine's permissible moves
        indices = filter(lambda x: x != -1, indices)
        for i in indices:
            legal_actions[i] = True
        return legal_actions

    def reset(self):
        pass

    def step(self):
        pass

    def render(self):
        pass
