"""
Convert a Keras model to the ONNX file format.
Usage: python onnx_converter.py FILE_IN FILE_OUT
"""
import sys

import onnxmltools
from keras.models import load_model

from dnn import azero_loss


print(sys.argv)


if len(sys.argv) != 3:
    print('Usage: python onnx_converter.py FILE_IN FILE_OUT')
    sys.exit()


mdl = load_model(sys.argv[1], custom_objects={
    'azero_loss': azero_loss
})

convert = onnxmltools.convert_keras(mdl)
onnxmltools.save_model(convert, sys.argv[2])
