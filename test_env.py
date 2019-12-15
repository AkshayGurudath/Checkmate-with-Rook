import chess
from chess_env import ChessEnv

env = ChessEnv(r"C:\Users\CSC\Desktop\Leiden\stockfish-10-win\stockfish-10-win\Windows\stockfish_10_x64.exe", player_color =chess.WHITE)
start_state = env.reset()
print(start_state)
print(env.board)
# legal_moves = env.legal_moves()
# print("king ", legal_moves[:8])
# print("rook horizontal ", legal_moves[8:22])
#print("rook vertical ", legal_moves[22:])
next_state, reward, done, _ = env.step(16)
print(next_state)
print(reward)
print(done)
print(env.board)
print(env.board.fullmove_number)

