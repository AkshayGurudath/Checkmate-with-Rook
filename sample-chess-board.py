
import chess.engine
import chess
board = chess.Board()
chess.STARTING_BOARD_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
Nf3 = chess.Move.from_uci("c2c3")
board.push(Nf3)
engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\CSC\Desktop\stockfish-10-win\stockfish-10-win\Windows\stockfish_10_x64.exe")
engine.id.get("name")
result = engine.play(board, chess.engine.Limit(time=0.100))
board.push(result.move)
engine.quit()
print(board)



