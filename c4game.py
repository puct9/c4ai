"""
Holds the necessary classes and methods for the game of connect-4
Game is played on a position with 7 columns and 6 rows
"""
from typing import Iterable, Tuple

import numpy as np


class C4Game:

    def __init__(self, history_frames: int = 1) -> None:
        """
        Parameters
        ----------
        history_frames: `int`
            Defaults to 1. The amount of history frames to feed to the neural
            network.
        """
        # initialise history, game state
        # game state representation as a 2d array, column by column
        self.move_history = []
        self.position = np.zeros((7, 6))
        self.to_move = -1  # -1 p1 to move, 1 is p2 to move
        self.history_frames = history_frames
        self.position_history = [np.zeros((7, 6))]

    @classmethod
    def find_four(self, span: Iterable) -> bool:
        """
        Parameters
        ----------
        span: `Iterable`
            An iterable with same type items implementing __eq__
        Returns
        -------
        contiguous_four: `bool`
            True if there are is a section of a contiguous set of 4 of the
            same item, as checked by __eq__
        """
        looking_for = None
        contiguous = 1
        for c in span:
            if not c:
                contiguous = 0
                continue
            if looking_for is None:
                looking_for = c
                contiguous = 1
            elif looking_for == c:
                contiguous += 1
                if contiguous == 4:
                    return True
            else:  # != c
                looking_for = c
                contiguous = 1
        return False

    @property
    def state(self) -> np.ndarray:
        """
        Returns
        -------
        ret: `np.ndarray`
            Inputs to the neural network
        """
        # no need for legal moves plane
        # start off with the legal moves input plane
        # ret = [np.zeros((7, 6))]
        # legal = self.legal_moves()
        # for i, c in enumerate(legal):
        #     ret[0][i, :] = c
        ret = []
        # who is it to move?
        ret.append(np.ones((7, 6)) if self.to_move == -1 else np.zeros((7, 6)))
        # board position input planes
        last_n = self.position_history[-self.history_frames:]
        if len(last_n) < self.history_frames:
            missing = self.history_frames - len(last_n)
            for _ in range(missing):
                last_n.insert(0, np.zeros((7, 6)))  # fill in
        for pos in last_n:
            ret.append((pos == -1).astype('float'))
            ret.append((pos == 1).astype('float'))
        return np.array(ret).reshape(7, 6, self.history_frames * 2 + 1)

    def state_copy(self) -> 'C4Game':
        """
        Returns
        -------
        new_game: `C4Game`
            A new C4Game object, NOT including full history, but returning
            enough history to statisfy every history frame
        """
        new_game = C4Game()
        new_game.move_history = self.move_history[-self.history_frames:]
        new_game.position_history = [x.copy() for x in
                                     self.position_history[
                                         -self.history_frames:]]
        new_game.position = self.position.copy()
        new_game.to_move = self.to_move
        return new_game

    def legal_moves(self) -> Tuple[int, int, int, int, int, int, int]:
        """
        Returns
        -------
        ret: `Tuple[int, int, int, int, int, int, int]`
            A 7-tuple of ints where 1 is legal to move and 0 is not
        """
        return tuple(int(x[-1] == 0) for x in self.position)

    def play_move(self, col: int) -> None:
        """
        Parameters
        ----------
        col: `int`
            The column of which the piece would be played
        Returns
        -------
        ret: `None`
        Raises
        ------
        `IndexError`
            The `col` argument is out of range of the columns
        `ValueError`
            The column specified is fully occupied
        """
        if not 0 <= col < 7:
            raise IndexError(f'Out of range column {col}')
        for i, c in enumerate(self.position[col]):
            if not c:
                self.position[col, i] = self.to_move
                self.to_move *= -1
                self.move_history.append(col)
                self.position_history.append(self.position.copy())
                return
        raise ValueError(f'Column is fully occupied')

    def undo_move(self) -> None:
        """
        Parameters
        ----------
        Returns
        -------
        ret: `None`
        Raises
        ------
        `IndexError`
            No moves have been played
        """
        if len(self.position_history) <= 1:
            raise IndexError('No moves have been played')
        self.move_history.pop()
        self.to_move *= -1
        self.position_history.pop()
        self.position = self.position_history[-1].copy()

    def check_terminal(self) -> bool:
        """
        Returns
        -------
        term:
            1 if 4 in a row is present on the board else 0 if draw else None
        """
        # check board full
        if not any(self.legal_moves()):
            return 0
        # check columns
        for col in self.position:
            # start from the bottom of the column (index 0)
            if C4Game.find_four(col):
                return 1
        # check rows
        for i in range(6):
            if C4Game.find_four(self.position[:, i]):
                return 1
        # check diagonals
        flipped = np.fliplr(self.position)
        for i in range(-3, 3):
            # main diagonal
            if C4Game.find_four(self.position.diagonal(i)):
                return 1
            # non-main diagonal
            if C4Game.find_four(flipped.diagonal(i)):
                return 1
        return None

    def __str__(self) -> str:
        """
        Returns
        -------
        ret: `str`
            String representation of the current state
        """
        ret = ''
        for row in range(6):
            sub = self.position[:, 5 - row]
            data = '| '
            data += ' | '.join('X' if x == -1 else 'O'
                               if x == 1 else ' ' for x in sub)
            data += ' |'
            ret += data + '\n'
        ret += '-' * (len(ret) // 6 - 1)
        return ret + '\n  0   1   2   3   4   5   6'

    def __repr__(self) -> str:
        """
        Returns
        -------
        ret: `str`
            String representation of the current state, plus ID of object
        """
        return f'{str(self)}\nid={str(id(self))}'
