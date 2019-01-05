import os
import pickle
import random
from collections import deque
from typing import List, Tuple

import numpy as np
import keras.backend as K
from keras.models import Model  # , load_model
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

import dnn2 as dnn
from selfplay import do_selfplay


class TrainingPipeline:
    """
    Training pipeline for the game.
    Uses TRPO (Trust Region Policy Optimisation) for retraining model
    """

    def __init__(self, playouts: int = 200,
                 history: int = 1, c_puct: float = 5, dir_alpha: float = 0.16,
                 buffer: deque = None, buffer_len: int = 10000,
                 model: Model = None, save_path: str = None,
                 resume: bool = False, lr_mul: float = 1,
                 tb_active: bool = False, kl_tgt: float = 2e-3,
                 temp_cutoff: int = 32, minibatch_size: int = 256) -> None:
        """
        Parameters
        ----------
        playouts: `int`
            Default 200. The amount of playouts for each MCTS search for
            training games
        history: `int`
            Default 1. The amount of previous board states (including current)
            to include in the input planes.
        c_puct: `float`
            Default 5. The constant controlling exploration
        dir_alpha: `float`
            Default 0.16. The alpha to use for diriclet noise for training
            games
        buffer: `collections.deque`
            Default None. The training buffer of positions. If no buffer is
            passed, a new empty one would be instantiated
        buffer_len: `int`
            Default 10000. The amount of past positions to store in the buffer
            before they are discarded
        model: `keras.models.Model`
            Default None. The neural network to load from. If no model is
            passed, a new one would be instantiated
        save_path: `str`
            Default None. The save directory. If no save_path is passed, the
            path would be named "tmp0"
        resume: `bool`
            Default False. Will resume from directory, model and buffer if
            applicable if resume is True, else a new run is created
        lr_mul: `float`
            Default 1. The learning rate multiplier
        tb_active: `bool`
            Default False.
        """
        # safety checks
        if save_path:
            if os.path.exists(save_path) and not resume:
                raise ValueError('There is an existing save.')
        if resume and not model:
            raise ValueError('No model to resume.')
        # end safety checks
        self.save_path = (save_path if save_path else os.path.join(os.getcwd(),
                                                                   'tmp0'))
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.playouts = playouts
        self.history = history
        self.c_puct = c_puct
        self.dir_alpha = dir_alpha
        self.learning_rate = 2e-3  # 0.002
        self.lr_multiplier = lr_mul  # beta
        self.minibatch_size = minibatch_size
        self.train_epochs = 5
        self.kl_tgt = kl_tgt  # 0.15 by default
        self.temp_cutoff = temp_cutoff
        if tb_active:
            self.tf_callback = TensorBoard(log_dir=os.path.join(self.save_path,
                                                                'logs'))
            if resume:
                self.tf_callback.write_graph = False
        else:
            self.tf_callback = None
        self.data_buffer = buffer if buffer else deque(maxlen=buffer_len)
        self.model = (model if model else
                      dnn.create_model(history * 2 + 1))
        print('Training Pipeline fueled and ready for liftoff!'
              '\nSummary:\n'
              f'Saving to: {self.save_path}\n'
              f'Board parameters:\nSearch parameters:'
              f'\nPlayouts={self.playouts} | History per player={self.history}'
              f' | CPUCT={self.c_puct} | Training dirichlet={self.dir_alpha}\n'
              f'Training parameters:\nLR={self.learning_rate} | Minibatch size'
              f'={self.minibatch_size} | Training epochs={self.train_epochs} |'
              f' LR Multiplier={self.lr_multiplier} | KL Target={self.kl_tgt}'
              f'\nTensorboard Active: {"yes" if tb_active else "no"}')
        print(f'Graph summary:')
        self.model.summary()

    def ext_equivalent_data(self,
                            data: List[Tuple[np.ndarray, int, np.ndarray,
                                             np.ndarray]]
                            ) -> None:
        """
        Extend the data buffer with the same data passed in but rotated
        Parameters
        ----------
        data: `List[Tuple[np.ndarray, int, np.ndarray]]`
            data in, where the tuple is in the form of (board state,
            end label, move made)
        Returns
        -------
        `None`
        """
        equivalent = []
        for state, winner, move_made, mvisits in data:
            # the input
            state: np.ndarray  # shape (7, 6, 3)
            # the winner, trains value mse
            winner: int  # 1 if won by c4, else 0
            # the move made, trains policy cross-entropy
            move_made: np.ndarray  # shape (7,)
            # the visits, trains policy mse
            mvisits: np.ndarray  # shape (7,)

            # W
            equivalent.append((state, winner, move_made, mvisits))
            # flipped
            equivalent.append((np.fliplr(state.reshape(3, 7, 6)
                                         ).reshape(7, 6, 3),
                               winner, move_made[::-1], mvisits[::-1]))
        self.data_buffer.extend(equivalent)

    def gen_sp_data(self) -> None:
        """
        Do some selfplay games. Handles all the rotating, flipping, game
        generation all in this method via some magic.
        Parameters
        ----------
        Returns
        -------
        `None`
        """
        gen = do_selfplay(1, self.playouts,
                          self.c_puct, self.model,
                          self.dir_alpha, self.temp_cutoff)
        # we're only doing 1 so we can just take the data via next() and the
        # generator will end
        states, result, moves, mvisits = next(gen)  # this is next gen stuff
        # result will be 1 if won by connecting 4, else it was a draw
        data = []
        for state, move, _mvisits in zip(states[::-1], moves[::-1],
                                         mvisits[::-1]):
            data.append((state, result, to_categorical([move],
                                                       num_classes=7)[0],
                         _mvisits * self.playouts / (self.playouts - 1)))
            # (above), multiply by scalar because
            # one playout is spent on expanding the root node
            result *= -1
        self.ext_equivalent_data(data)

    def update_network(self, e: int = 0) -> None:
        """
        Update the network with latest training data.
        Parameters
        ----------
        Returns
        -------
        `None`
        """
        minibatch = random.sample(self.data_buffer, self.minibatch_size)
        states = np.array([d[0] for d in minibatch])
        results = np.array([d[1] for d in minibatch])
        # ignore move made
        mvisits = np.array([d[3] for d in minibatch])
        K.set_value(self.model.optimizer.lr,
                    self.learning_rate * self.lr_multiplier)
        old_probs = self.model.predict(states)[1]
        for i in range(self.train_epochs):
            # callback only on first training epoch
            # hence tensorboard / history will be reflective
            # of a model's true performance, disregarding overfitting
            if self.tf_callback and not i:
                self.model.fit(x=states, y=[results, mvisits],
                               batch_size=self.minibatch_size,
                               callbacks=[self.tf_callback],
                               verbose=False)
                self.tf_callback.write_graph = False
            else:
                self.model.fit(x=states, y=[results, mvisits],
                               batch_size=self.minibatch_size,
                               verbose=False)
            new_probs = self.model.predict(states)[1]
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) -
                                             np.log(new_probs + 1e-10)),
                                axis=1))
            if kl > self.kl_tgt * 4:
                # https://www.inference.vc/alphago-zero-policy-improvement-and-vector-fields/
                break  # TRPO
        # adaptive learning rate!
        if kl > self.kl_tgt * 2 and self.lr_multiplier > 1e-10:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_tgt / 2 and self.lr_multiplier < 17:
            self.lr_multiplier *= 1.5
        print(f'Retrained network successfully. kl:{round(float(kl), 5)}, '
              f'lr_mul:{round(self.lr_multiplier, 3)}')

    def run(self, start_cycle: int = 0) -> None:
        """
        Start running the training cycle
        Parameters
        ----------
        start_cycle: `int`
            If resuming, set start_cycle to the number of the network being
            loaded from
        """
        cycle = start_cycle
        while True:
            cycle += 1
            for _ in range(1):
                self.gen_sp_data()
            print(f'INFO: cycle={cycle}, datapoints={len(self.data_buffer)}')
            if len(self.data_buffer) >= self.minibatch_size * 2:
                self.update_network(cycle)
            if cycle % 10:
                continue
            self.model.save(os.path.join(self.save_path, f'save_{cycle}.ntwk'))
            pickle.dump(self.data_buffer, open(os.path.join(self.save_path,
                                                            'data_buffer.dbuf'
                                                            ), 'wb'))


def main() -> None:
    model = dnn.create_model(3)
    path = './test0'
    buf = None
    pipeline = TrainingPipeline(model=model, save_path=path, dir_alpha=10/7,
                                tb_active=True, resume=False, buffer=buf,
                                lr_mul=1/1.5 ** 1, temp_cutoff=15,
                                playouts=400, kl_tgt=15e-4)
    pipeline.run(0)


if __name__ == '__main__':
    main()
