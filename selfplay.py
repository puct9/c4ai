"""
Generate, save and prepare selfplay games for training and for profit?
"""
import numpy as np
from keras.models import Model

from c4game import C4Game
from mcts import MCTS


def do_selfplay(num: int, playouts: int,
                c_puct: int, mdl: Model,
                dir_alpha: float, temp_cutoff: int) -> tuple:
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
        print('Starting self-play game')
        game = C4Game()
        searcher = MCTS(game, True, mdl, c_puct, playouts, dir_alpha)
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
            searcher_ = MCTS(game, True, mdl, c_puct, playouts, dir_alpha)
            for n in searcher.top_node.children:
                if n.move == move:
                    n.move = None
                    n.parent = None
                    n.P = None
                    searcher_.top_node = n
            # print(f'info winrate {searcher_.top_node.Q * 50 + 50}%')
            # print('Tree reuse efficiency '
            #       f'[{searcher_.top_node.N}/{playouts}]')
            # print(game)
            searcher = searcher_
        # print(f'Finished {game_num + 1}/{num} self-play games, '
        #       f'res={game.check_terminal()}')
        yield state_logs, game.check_terminal(), move_logs, move_search_logs
