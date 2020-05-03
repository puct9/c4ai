"""
Alternate version of play_vs_ai.py but with PONDERING
PONDERING is when the engine thinks in the opponent's time
"""
import sys
import threading

import tensorflow as tf
from keras.models import Model, load_model

from c4game import C4Game
from mcts_v2 import MCTS


# network
MODEL_FILE = './testXVI/save_2071.ntwk'
MODEL = load_model(MODEL_FILE)
POSITION = C4Game()
ENG_POSITION = C4Game()

MODEL._make_predict_function()
tf.get_default_graph().finalize()


class SearchThread(threading.Thread):

    def __init__(self):
        super(SearchThread, self).__init__()
        self._stopping = threading.Event()
        self._finished = threading.Event()
        self._target = search
        self._args = (ENG_POSITION, MODEL, ENGINE)
        self._kwargs = {}

    def stop(self):
        self._stopping.set()

    def stopped(self) -> bool:
        return self._stopping.is_set()

    def is_finished(self) -> bool:
        return self._finished.is_set()


def search(position: C4Game, model: Model, eng: MCTS):
    # non-stop build the search tree
    # search thread method
    while True:
        eng.playouts += 30
        eng.playout_to_max()
        if SEARCH_THREAD.stopped():
            break
    SEARCH_THREAD._finished.set()


def apply_move(eng: MCTS, move: int):
    for node in eng.top_node.children:
        if node.move == move:
            break
    node.parent = None
    node.move = None
    node.P = None
    eng.top_node = node
    eng.playouts = node.N


# finish the definitions
ENGINE = MCTS(ENG_POSITION, False, MODEL, 3, 2, 10)

search_info = []


while POSITION.check_terminal() is None:
    # start searching as the player thinks irl
    search_info.append([ENGINE.top_node.N])
    SEARCH_THREAD = SearchThread()
    SEARCH_THREAD.start()
    # player move
    print(POSITION)
    while True:
        try:
            move = int(input('Move (0 to 6): '))
            POSITION.play_move(move)
            break
        except Exception:
            pass
        except KeyboardInterrupt:
            SEARCH_THREAD.stop()
            sys.exit()
    print(POSITION)
    SEARCH_THREAD.stop()
    while not SEARCH_THREAD.is_finished():
        pass
    if POSITION.check_terminal() is not None:
        break
    search_info[-1].append(ENGINE.top_node.N)
    apply_move(ENGINE, move)
    ENG_POSITION.play_move(move)
    if ENGINE.top_node.N < 3000:
        ENGINE.playouts = 3000
        ENGINE.playout_to_max()

    eng_move = ENGINE.pick_move()
    POSITION.play_move(eng_move)
    ENG_POSITION.play_move(eng_move)
    apply_move(ENGINE, eng_move)

    winrate = ENGINE.top_node.Q / 2 + 0.5
    print(f'Expected score: {round(winrate, 2)}')


print('Game over!')
