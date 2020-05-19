"""
Multithreaded selfplay game generation using subprocesses
This should work out of the box at with Windows. Binaries will need to be
manually compiled on other platforms and their respective run commands edited.
"""
import random
from subprocess import Popen, PIPE
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from keras.models import Model

from c4game import C4Game
from onnx_converter import save as save_as_onnx

THREADS = 6


def do_selfplay(num: int, playouts: int,
                c_puct: float, mdl: Model,
                dir_alpha: float, temp_cutoff: int,
                *args) -> tuple:
    """
    Do and save to a file some selfplay games
    Parameters
    ----
    num: `int`
        The number of selfplay games to make
    playouts: `int`
        The amount of playouts in MCTS
    c_puct: `float`
        PUCT for MCTS
    mdl: `tensorflow.keras.models.Model`
        Model used for predictions
    dir_alpha: `float`
        Dirichlet noise alpha value

    Yields
    ------
    `Tuple[np.ndarray, int, int]`
    """
    save_as_onnx(mdl, './standalone/Release_x64/Models/temp.onnx')

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        funcs = [executor.submit(fast_selfplay, playouts, c_puct,
                                 dir_alpha, temp_cutoff,
                                 random.randint(1, 4294967295))
                 for _ in range(num)]
        results = [f.result() for f in funcs]
    for res in results:
        yield res


def fast_selfplay(playouts: int,
                  c_puct: float, dir_alpha: float, temp_cutoff: int,
                  force_seed: int = None):
    if force_seed is None:
        force_seed = random.randint(1, 4294967295)

    # spawn process
    sub = Popen('./standalone/Release_x64/C4UCT.exe',
                cwd='./standalone/Release_x64',
                universal_newlines=True,
                stdin=PIPE,
                stdout=PIPE, stderr=PIPE)
    sub.stdin.write(f'ssp\nseed {force_seed}\nc_puct set {c_puct}\n'
                    f'dir_alpha set {dir_alpha}\n'
                    f'temp_cutoff set {temp_cutoff}\n'
                    f'playouts set {playouts}\nsspgo\n')
    sub.stdin.flush()

    # ignore some lines we don't want
    while True:
        line = sub.stdout.readline().strip()
        if line.startswith('seed set to '):
            break

    game = C4Game()
    state_logs = []
    move_logs = []
    move_search_logs = []
    while True:
        info = sub.stdout.readline().strip()
        if info == 'done':
            break
        move = info[-1]
        info = info[:-3]
        probs = map(float, info.split(' '))
        move_search_logs.append(np.array(list(probs)))
        state_logs.append(game.state)
        move_logs.append(int(move))
        game.play_move(int(move))

    sub.kill()
    return state_logs, game.check_terminal(), move_logs, move_search_logs
