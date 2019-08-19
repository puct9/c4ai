import keras.backend as K
from keras.layers import (Conv2D, Dense, Flatten, Input, Layer,
                          BatchNormalization, Add, Activation)
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2


def azero_loss(y_true, y_pred):
    return K.mean(-y_true * K.log(y_pred))


def create_model(planes: int) -> Model:
    """
    Parameters
    ----------
    planes: `int`
        The amount of input planes for a given position
    """
    reg_term = 1e-4  # L2 reg term
    board_in = Input(shape=(7, 6, planes), name='Board_Input')
    x = Conv2D(32, (3, 3), padding='same',
               kernel_regularizer=l2(reg_term), name='Conv2D_1')(board_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # residual layers/conns
    x = add_residual(x, 32, reg_term, 1)
    # x = add_residual(x, 32, reg_term, 2)

    # value head
    with K.name_scope('Value_Head'):
        v = Conv2D(1, (1, 1), padding='same',
                   kernel_regularizer=l2(reg_term))(x)
        # padding = same does not matter (see above)
        v = BatchNormalization()(v)
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
        p = BatchNormalization()(p)
        p = Activation('relu')(p)
        p = Flatten()(p)
        policy_head = Dense(7,
                            activation='softmax',
                            kernel_regularizer=l2(reg_term),
                            name='Policy_Out')(p)

    deep_neural_network = Model(inputs=[board_in],
                                outputs=[value_head, policy_head])
    deep_neural_network.compile(SGD(momentum=0.9),
                                loss=['mse', azero_loss]
                                )
    return deep_neural_network


def add_residual(x: Layer, cnnfilters: int, reg_term: float, i: int):
    """
    Adds a complete residual layer before returning the result

    Comprises of cnnfilters 3x3 conv, BN, relu, cnnfilters 3x3 conv, BN, res+,
    relu
    """
    # "a" is the new layer being applied to the former
    with K.name_scope(f'Residual_Layer_{i}'):
        a = Conv2D(cnnfilters, (3, 3), padding='same',
                   kernel_regularizer=l2(reg_term),
                   name=f'Residual_{i}_Conv2D_1')(x)
        a = BatchNormalization()(a)
        a = Activation('relu')(a)
        a = Conv2D(cnnfilters, (3, 3), padding='same',
                   kernel_regularizer=l2(reg_term),
                   name=f'Residual_{i}_Conv2D_2')(a)
        a = BatchNormalization()(a)
        a = Add()([a, x])
        a = Activation('relu')(a)
    return a
