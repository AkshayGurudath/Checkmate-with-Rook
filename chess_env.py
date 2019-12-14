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

    def _pos_to_uci(self, pos):
        x, y = pos # x, y are in the range 1 to 8
        uci_x = chr(x + 96) # 97 = 'a'
        uci_y = str(y)
        return uci_x + uci_y


    # action is of the form -
    # [Kdiagupleft, Kup, Kdiagupright,
    #   Kright,
    #   Kdiagdownright, Kdown, Kdiagdownleft,
    #   Kleft,
    #   RookHorizontal: -7 to + 7,
    #   RookVertical: -7 to +7]
    # Total dims - 8 + 15 + 15 = 38
    # Thus, input here is from 0 to 37
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
        pass

    def _get_current_state(self):
        pass

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
