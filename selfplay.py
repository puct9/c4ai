"""
Generate, save and prepare selfplay games for training and for profit?
"""
import numpy as np
from keras.models import Model

from c4game import C4Game
# from mcts import MCTS
from mcts_v2 import MCTS


def do_selfplay(num: int, playouts: int,
                c_puct: int, mdl: Model,
                dir_alpha: float, temp_cutoff: int,
                mcts_batch_size: int) -> tuple:
    """
    Do and save to a file some selfplay games
    Parameters
    ----
    num: `int`
        The number of selfplay games to make
    playouts: `int`
        The amount of playouts in MCTS
    c_puct: `int`
        PUCT for MCTS
    mdl: `tensorflow.keras.models.Model`
        Model used for predictions
    dir_alpha: `float`
        Dirichlet noise alpha value

    Yields
    ------
    `Tuple[np.ndarray, int, int]`
    """
    for game_num in range(num):
        print(f'Starting self-play game {game_num + 1}/{num}')
        game = C4Game()
        searcher = MCTS(game, True, mdl, c_puct, playouts, dir_alpha=dir_alpha,
                        batch_size=mcts_batch_size)
        state_logs = []
        move_logs = []
        move_search_logs = []
        while game.check_terminal() is None:
            # temperature decay
            move_search_logs.append(np.array(searcher.playout_to_max()))
            move = searcher.pick_move(temp=1 if len(game.move_history) <
                                      temp_cutoff
                                      else 1e-3)
            state_logs.append(game.state)
            move_logs.append(move)
            game.play_move(move)
            # tree reuse
            searcher_ = MCTS(game, True, mdl, c_puct, playouts,
                             dir_alpha=dir_alpha, batch_size=mcts_batch_size)
            for n in searcher.top_node.children:
                if n.move == move:
                    n.move = None
                    n.parent = None
                    n.P = None
                    searcher_.top_node = n
                    break
            searcher = searcher_
        yield state_logs, game.check_terminal(), move_logs, move_search_logs
