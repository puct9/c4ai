"""
UCI-like interface for c4game engine
"""
import os
import re
import time
import threading

import cv2
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model

from c4game import C4Game
# from mcts import MCTS
from mcts_v2 import MCTS
from dnn import azero_loss


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# network
MODEL_FILE = './testXVI/save_2071.ntwk'
MODEL = load_model(MODEL_FILE,
                   custom_objects={'azero_loss': azero_loss})
POSITION = C4Game()


class SearchThread(threading.Thread):

    def __init__(self, *, stime: int = None, nodes: int = 5000):
        super(SearchThread, self).__init__()
        self._stopping = threading.Event()
        self._target = search
        self._args = (POSITION, MODEL)
        if stime is not None:
            self._kwargs = {'stime': stime}
        else:
            self._kwargs = {'nodes': nodes}

    def stop(self):
        self._stopping.set()

    def stopped(self) -> bool:
        return self._stopping.is_set()


def search(position: C4Game, model: Model, *,
           stime: int = None, nodes: int = 5000):
    if stime is not None:
        nodes = float('inf')
    # search thread method
    eng = MCTS(position, False, model, 3, 2, 10)
    start_time = time.time()
    cycle = -1
    pv = []
    while eng.playouts < nodes:
        cycle += 1
        eng.playout_to_max()
        eng.playouts += min(30, nodes - eng.playouts)
        # check if the pv has changed
        if not cycle % 5:  # == 0, arbitrarily chosen number
            new_pv = eng.get_pv()
            if len(new_pv) != len(pv):
                pv = new_pv
                print(f'nodes {eng.playouts} pv ' + ' '.join(str(x.move)
                      for x in pv))
            else:
                if not all(a is b for a, b in zip(pv, new_pv)):
                    pv = new_pv
                    print(f'nodes {eng.playouts} pv ' +
                          ' '.join(str(x.move) for x in pv))
        # check if thread has been asked to end
        if SEARCH_THREAD.stopped():
            break
        if stime is not None and time.time() - start_time > stime:
            break
    # show prior information
    print('\n'.join(str(x) for x in eng.top_node.children))
    pv = eng.get_pv()
    print(f'nodes {eng.playouts} pv ' + ' '.join(str(x.move) for x in pv))
    print(f'bestmove {pv[0]}')
    SEARCH_THREAD.stop()  # set stopped flag


def main():
    global SEARCH_THREAD
    global POSITION
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
            else:
                SEARCH_THREAD = SearchThread()
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
        if inp == 'static' and not searching:
            # show static evaluation of policy net
            value, policy = MODEL.predict(np.expand_dims(POSITION.state, 0))
            print(f'V={value[0][0]}')
            print('\n'.join(f'MV={i} P={round(100 * p, 2)}%'
                            for i, p in enumerate(policy[0])))
        if inp.startswith('position') and not searching:
            inp = inp.split(' ')
            if len(inp) < 2:
                continue
            if inp[1] == 'startpos':
                POSITION = C4Game()
                if len(inp) > 3 and inp[2] == 'moves':
                    for m in inp[3:]:
                        try:
                            POSITION.play_move(int(m))
                        except Exception as e:
                            print(e)
                            POSITION = C4Game()
            if len(inp) > 3 and inp[1] == 'set':
                POSITION = C4Game()
                pstr = inp[2]  # position string representation
                gstr = ''  # geometric string representation
                for c in pstr:
                    if c.upper() in 'XO/':
                        gstr += c.upper()
                    if c.isdigit():
                        gstr += ' ' * int(c)
                gstr = gstr.split('/')
                rpos = np.array([list(x) for x in gstr])  # rotated position
                pos90 = np.rot90(rpos, k=3)  # rotated correctly
                mat = np.zeros((7, 6)) - (pos90 == 'X') + (pos90 == 'O')
                POSITION.position = mat
                POSITION.to_move = -1 if inp[3].upper() == 'X' else 1
                POSITION.position_history = [mat.copy()]
            inp = ' '.join(inp)
        if inp.startswith('image'):
            print(np.moveaxis(POSITION.state, 2, 0))
            if len(inp.split(' ')) == 2:
                fout = inp.split(' ')[1]  # file out name
                try:
                    cv2.imwrite(fout, POSITION.state * 255)
                except Exception as e:
                    print(e)
        if inp == 'exit' or inp == 'quit':
            print('Goodbye ~ uwu')
            os.sys.exit()


if __name__ == '__main__':
    print(f'schwi20190816 is using {MODEL_FILE} and at your service ~ owo')
    main()
