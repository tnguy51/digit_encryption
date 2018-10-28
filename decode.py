#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('font', size=15)
mpl.rc('figure', figsize=(8, 5))

from keras.models import load_model

from tkinter import *
from tkinter.ttk import Combobox
import tkinter.filedialog as tkFileDialog
from PIL import Image

def decode_image(encoded, decoder, logo, weight):
    """ Take in 4 images of digits, encode and embed the UR logo

    Arguments:
    ----------
    encoded: array of shape (16, 16, 3)
    decoder: convolutional decoder
    logo: UR logo image. array of shape (16, 16, 3)

    Returns:
    --------
    digits: array of shape (4, 28, 28, 1)

    """
    encoded = encoded*(weight+1) - weight*logo
    digit_1 = encoded[:8, :8, :]
    digit_2 = encoded[:8, 8:16, :]
    digit_3 = encoded[8:16, :8, :]
    digit_4 = encoded[8:16, 8:16, :]

    encoded = np.stack([digit_1, digit_2, digit_3, digit_4], axis=0)
    digits = decoder.predict(encoded)

    return digits

def main():
    window = Tk()
    window.title("Decode")
    window.geometry("400x400")

    label = Label(window, text = "")
    label.grid(column = 0, row = 0)

    labelUpload = Label(window, text = "Upload Image here: ", font=("Times New Roman", 15))
    labelUpload.grid(column = 0, row = 5)

    def clicked():
        image_fname = tkFileDialog.askopenfilename(
            parent=window, title='Image file to decrypt')

        # parameters
        weight = 5
        logo = plt.imread('logo.png')

        # import encoder and decoder
        decoder = load_model('models/decoder.h5')

        # create password image and decode
        digits_encoded = plt.imread(image_fname)
        digits_decoded = decode_image(digits_encoded, decoder, logo, weight=weight)

        # show and save decoded image vs original
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i in range(4):
            ax = axes.flatten()[i]
            ax.imshow(digits_decoded[i].reshape(28, 28))
            ax.axis('off')
        plt.savefig('digits_decoded.png', dpi=100, bbox_inches='tight')

        classifier = load_model('models/classifier.h5')
        digits = np.argmax(classifier.predict(digits_decoded), axis=1)
        digits_str = '%d%d%d%d' % (digits[0], digits[1], digits[2], digits[3])
        labelPin = Label (window, text = "%s" % (digits_str), font=("Times New Roman", 15))
        labelPin.grid(column = 5, row = 10)

    btn = Button(window, text = "Choose File",font=("Times New Roman", 10), command=clicked, width=15)
    btn.grid (column = 6, row = 5)

    labelPin = Label (window, text = "Pin: ", font=("Times New Roman", 15))
    labelPin.grid(column = 0, row = 10)


    window.mainloop()

if __name__ == '__main__':
    main()