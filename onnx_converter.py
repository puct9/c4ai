"""
Convert a Keras model to the ONNX file format.
Usage: python onnx_converter.py FILE_IN FILE_OUT
"""
import sys

import onnxmltools
from keras.models import load_model


def convert_and_save(fin, fout):
    mdl = load_model(fin)
    save(mdl, fout)


def save(mdl, fout):
    convert = onnxmltools.convert_keras(mdl)
    onnxmltools.save_model(convert, fout)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python onnx_converter.py FILE_IN FILE_OUT')
        sys.exit()

    convert_and_save(sys.argv[1], sys.argv[2])
