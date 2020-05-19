import os
import pickle
import random
from collections import deque
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model, load_model
from keras.utils import to_categorical

# import dnn
import dnn
from selfplay import do_selfplay


class TrainingPipeline:
    """
    Training pipeline for the game.
    Uses TRPO (Trust Region Policy Optimisation) for retraining model
    """

    def __init__(self, playouts: int = 200,
                 history: int = 1, c_puct: int = 5, dir_alpha: float = 0.16,
                 buffer: deque = None, buffer_len: int = 10000,
                 model: Model = None, save_path: str = None,
                 resume: bool = False, lr_mul: float = 1,
                 tb_active: bool = False, kl_tgt: float = 2e-3,
                 temp_cutoff: int = 32, minibatch_size: int = 256,
                 n_sp: int = 1, mcts_batch_size: int = 10) -> None:
        """
        Parameters
        ----------
        playouts: `int`
            The amount of playouts for each MCTS search in training games,
            defaulting to 200
        history: `int`
            Default 1. The amount of previous board states (including current)
            to include in the input planes
        c_puct: `float`
            Default 5. The constant controlling exploration. This is slightly
            changed in the way it affects MCTS in the second version, which
            include virtual loss, and a non-constant c_puct. In this case, it
            acts as c_puctbase
        dir_alpha: `float`
            Default 0.16. The alpha to use for diriclet noise in training games
        buffer: `collections.deque`
            Default None. The training buffer of past positions. If no buffer
            is given, a new one is automatically instantiated
        buffer_len: `int`
            Default 10000. The number of past positions to store
        model: `keras.models.Model`
            Default None. The neural network to load from. If no model is
            given, a new one is automatically instantiated
        save_path: str
            Default 'tmp0'. The save directory.
        resume: `bool`
            Default False. Set to True if resuming training from a save
        lr_mul: `float`
            Default 1. The learning rate multiplier. (This will be
            automatically controlled once training starts)
        tb_active: `bool`
            Default False. Whether to write training metrics to tensorboard or
            not.
        kl_tgt: `float`
            Default 2e-3. The KL divergence target.
        temp_cutoff: `int`
            Default 12. The amount of moves in training games where randomness
            is applied in move selection
        minibatch_size: `int`
            Default 256. The batch size used for retraining the neural network
        n_sp: `int`
            Default 1. The amout of self-play games to play before each step
        mcts_batch_size: `int`
            Default 10. The level of parallisation in MCTS
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
        self.mcts_batch_size = mcts_batch_size
        self.dir_alpha = dir_alpha
        self.learning_rate = 2e-3  # 0.002
        self.lr_multiplier = lr_mul  # beta
        self.minibatch_size = minibatch_size
        self.n_sp = n_sp
        self.train_epochs = 5
        self.kl_tgt = kl_tgt  # 0.15 by default
        self.temp_cutoff = temp_cutoff
        self.data_buffer = buffer if buffer else deque(maxlen=buffer_len)
        self.model = (model if model else
                      dnn.create_model(history * 2 + 1))
        if tb_active:
            self.tf_writer = tf.summary.FileWriter(os.path.join(
                self.save_path, 'logs'), K.get_session().graph)
        else:
            self.tf_writer = None
        print('Training Pipeline fueled and ready for liftoff!'
              '\nSummary:\n'
              f'Saving to: {self.save_path}\n'
              f'Board parameters:\nSearch parameters:'
              f'\nPlayouts={self.playouts} | History per player={self.history}'
              f' | CPUCT={self.c_puct} | Training dirichlet={self.dir_alpha}'
              f' | MCTS batch size={self.mcts_batch_size}\n'
              f'Training parameters:\nLR={self.learning_rate} | Minibatch size'
              f'={self.minibatch_size} | Training epochs={self.train_epochs} |'
              f' LR Multiplier={self.lr_multiplier} | KL Target={self.kl_tgt}'
              f'\nSP games per step={self.n_sp}'
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
            equivalent.append((state[::-1],
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
        gen = do_selfplay(self.n_sp, self.playouts,
                          self.c_puct, self.model,
                          self.dir_alpha, self.temp_cutoff,
                          self.mcts_batch_size)
        for states, result, moves, mvisits in gen:  # this is next gen stuff
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
        e: int
            The epoch
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
            if self.tf_writer and not i:
                train_hist = self.model.fit(x=states, y=[results, mvisits],
                                            batch_size=self.minibatch_size,
                                            verbose=False)
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
        elif kl < self.kl_tgt / 2 and self.lr_multiplier < 40:
            self.lr_multiplier *= 1.5
        print(f'Retrained network successfully. kl:{round(float(kl), 5)}, '
              f'lr_mul:{round(self.lr_multiplier, 3)}')
        # make the summary
        summary = tf.Summary()
        for key, value in train_hist.history.items():
            summary.value.add(tag=key, simple_value=value[0])
        # we have some custom scalar(s) to add
        summary.value.add(tag='lr',
                          simple_value=self.lr_multiplier)
        self.tf_writer.add_summary(summary, e)
        self.tf_writer.flush()

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
            self.gen_sp_data()
            print(f'INFO: cycle={cycle}, datapoints={len(self.data_buffer)}')
            if len(self.data_buffer) >= self.minibatch_size * 2:
                self.update_network(cycle)
            if cycle % 1:
                continue
            self.model.save(os.path.join(self.save_path, f'save_{cycle}.ntwk'))
            pickle.dump(self.data_buffer, open(os.path.join(self.save_path,
                                                            'data_buffer.dbuf'
                                                            ), 'wb'))


def main() -> None:
    model = dnn.create_model(3)
    # if loading:
    # model = load_model('./SAVE_PATH/save_XYZ.ntwk')
    path = './SAVE_PATH'
    buf = None
    # if loading:
    # buf = pickle.load(open('./SAVE_PATH/data_buffer.dbuf', 'rb'))
    # some hyperparmaters I prepared earlier
    pipeline = TrainingPipeline(model=model, save_path=path, dir_alpha=0.8,
                                tb_active=True, resume=True, buffer=buf,
                                lr_mul=1/1.5**-1, temp_cutoff=12,
                                playouts=600, kl_tgt=1e-3, c_puct=3,
                                buffer_len=100000, n_sp=10, minibatch_size=512,
                                mcts_batch_size=10)
    pipeline.run(0)
    # if loading:
    # pipeline.run(XYZ)


if __name__ == '__main__':
    main()
