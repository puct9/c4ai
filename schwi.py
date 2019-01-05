"""
UCI-like interface for c4game engine
"""
import os
import re
import time
import threading

import tensorflow as tf
from keras.models import Model, load_model

from c4game import C4Game
from mcts import MCTS
from dnn import azero_loss


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# network
MODEL = load_model('./test11/save_1840.ntwk',
                   custom_objects={'azero_loss': azero_loss})
POSITION = C4Game()


class SearchThread(threading.Thread):

    def __init__(self, *, stime: int = None, nodes: int = 2000):
        """
        Instantiate a SearchThread
        Parameters
        ----------
        stime: `int`
            Search time (seconds)
        nodes: `int`
            Amount of nodes for MCTS
        """
        super(SearchThread, self).__init__()
        self._stopping = threading.Event()
        self._target = search
        self._args = (POSITION, MODEL)
        if stime is not None:
            self._kwargs = {'stime': stime}
        else:
            self._kwargs = {'nodes': nodes}

    def stop(self):
        """Stop the search after the next batch of playouts"""
        self._stopping.set()

    def stopped(self) -> bool:
        """
        Shows whether the next batch of playouts would be the last, or if the
        search has already been stopped by the stop command
        Returns
        -------
        stopped: `bool`
            True if the search is stopping or stopped, else False
        """
        return self._stopping.is_set()


def search(position: C4Game, model: Model, *,
           stime: int = None, nodes: int = 2000):
    """
    Search function, searching from a given position, controlled by thread
    Parameters
    ----------
    position: `C4Game`
        The game state to search from
    model: `keras.models.Model`
        The neural network to use
    stime: `int`
        Search time in seconds (optional)
    nodes: `int`
        Amount of nodes to search (optional)
    """
    if stime is not None:
        nodes = float('inf')
    # search thread method
    eng = MCTS(position, False, model, 5, 25)
    start_time = time.time()
    while eng.playouts < nodes:
        eng.playout_to_max()
        eng.playouts += min(25, nodes - eng.playouts)
        # check if thread has been asked to end
        if SEARCH_THREAD.stopped():
            break
        if stime is not None and time.time() - start_time > stime:
            break
    pv = eng.get_pv()
    print('pv ' + ' '.join(str(x.move) for x in pv))
    print(f'bestmove {pv[0]}')
    SEARCH_THREAD.stop()  # set stopped flag


def main():
    global SEARCH_THREAD
    SEARCH_THREAD = None
    # prepare the model
    MODEL._make_predict_function()
    tf.get_default_graph().finalize()
    searching = False
    while True:
        inp = input()
        if SEARCH_THREAD is None or SEARCH_THREAD.stopped():
            searching = False  # check
        if inp.startswith('go'):
            nodes = 2000
            match_n = re.match(r'^go n ?=? ?(\d+)', inp)  # match node
            match_t = re.match(r'^go t ?=? ?(\d+)', inp)  # match time
            if match_n:
                nodes = int(match_n.group(1))
                SEARCH_THREAD = SearchThread(nodes=nodes)
            elif match_t:
                stime = int(match_t.group(1))
                SEARCH_THREAD = SearchThread(stime=stime)
            SEARCH_THREAD.start()
            searching = True
        if inp == 'd':
            print(POSITION)
        if inp == 'stop' and searching:
            searching = False
            SEARCH_THREAD.stop()
        if inp == 'isready':
            print('readyok')
        if inp.startswith('mv') and not searching:
            try:
                move = int(inp.split(' ')[1])
                POSITION.play_move(move)
            except Exception as e:
                print(e)
                continue
        if inp == 'undo' and not searching:
            try:
                POSITION.undo_move()
            except Exception:
                continue


if __name__ == '__main__':
    main()
