import keras.backend as K
from keras.layers import (Conv2D, Dense, Flatten, Input,
                          Activation)
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2


def create_model(planes: int) -> Model:
    """
    Parameters
    ----------
    planes: `int`
        The amount of input planes for a given position
    """
    reg_term = 1e-4  # L2 reg term
    board_in = Input(shape=(7, 6, planes), name='Board_Input')
    x = Conv2D(16, (3, 3), padding='same',
               kernel_regularizer=l2(reg_term), name='Conv2D_1')(board_in)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same',
               kernel_regularizer=l2(reg_term), name='Conv2D_2')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=l2(reg_term), name='Conv2D_3')(x)
    x = Activation('relu')(x)

    # value head
    with K.name_scope('Value_Head'):
        v = Conv2D(1, (1, 1), padding='same',
                   kernel_regularizer=l2(reg_term))(x)
        # padding = same does not matter (see above)
        v = Activation('relu')(v)
        v = Flatten()(v)
        v = Dense(32, activation='relu',
                  kernel_regularizer=l2(reg_term))(v)
        value_head = Dense(1, activation='tanh',
                           kernel_regularizer=l2(reg_term),
                           name='Value_Out')(v)

    # policy head
    with K.name_scope('Policy_Head'):
        # again it does not matter
        p = Conv2D(4, (1, 1), padding='same',
                   kernel_regularizer=l2(reg_term))(x)
        p = Activation('relu')(p)
        p = Flatten()(p)
        policy_head = Dense(7,
                            activation='softmax',
                            kernel_regularizer=l2(reg_term),
                            name='Policy_Out')(p)

    deep_neural_network = Model(inputs=[board_in],
                                outputs=[value_head, policy_head])
    deep_neural_network.compile(Adam(lr=2e-3),
                                loss=['mse', 'categorical_crossentropy']
                                )
    return deep_neural_network
