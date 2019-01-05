from keras.models import Model, load_model

from c4game import C4Game
from mcts import MCTS
from dnn import scaled_mse_loss, azero_loss


def vs_ai(mdl: Model, go_first: bool = True) -> None:
    """
    Allows a human player to play against the AI
    Parameters
    ----------
    mdl: `keras.models.Model`
        The neural network to use
    go_first: `bool`
        True of the player wishes to go first, else False
    """
    game = C4Game()
    moves = 0
    print('Starting the game!')
    print(game)
    while game.check_terminal() is None:
        if moves % 2 != int(go_first):
            # human
            while True:
                try:
                    move = int(input('Move (0 to 6): '))
                    game.play_move(move)
                    break
                except Exception:
                    pass
        else:
            # ai
            searcher = MCTS(game, False, mdl, 5, 100)
            move = searcher.pick_move()
            game.play_move(move)
            # pv
            pv = searcher.get_pv()
            print(f'Expected score: {pv[0].Q}')
            # for i, node in enumerate(pv):
            #     if node.move is None:
            #         continue
            #     print(node)
            if pv[0].Q < -0.9:
                print(game, '\nI resign!')
                break
        print(game)
        moves += 1
    print('Game over!')


if __name__ == '__main__':
    vs_ai(load_model('./test11/save_1840.ntwk',
                     custom_objects={'scaled_mse_loss': scaled_mse_loss,
                                     'azero_loss': azero_loss}),
          True)
