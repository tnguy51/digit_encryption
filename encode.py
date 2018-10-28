#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.misc import imsave

mpl.rc('font', size=15)
mpl.rc('figure', figsize=(8, 5))

from keras.datasets import mnist
from keras.models import load_model

from tkinter import *
from tkinter.ttk import Combobox
import tkinter.filedialog as tkFileDialog
from PIL import Image


def randomizer(digits_str, x, y):

    digits = [int(r) for r in digits_str]

    image_list = []
    for digit in digits:
        mask = (y == digit)
        image_list.append(x[mask][np.random.randint(np.sum(mask), size=1)])

    image_list = np.array(image_list)
    image_list = image_list.reshape(len(digits_str), 28, 28, 1)
    return image_list

def encode_image(digits, encoder, logo, weight=100):
    """ Take in 4 images of digits, encode and embed the UR logo

    Arguments:
    ----------
    digits: array of shape (4, 28, 28, 1)
    encoder: convolutional encoder
    logo: UR logo image. array of shape (16, 16, 3)
    weight: weight of logo image

    Returns:
    --------
    encoded: array of shape (16, 16, 3)

    """

    digits = digits.reshape(4, 28, 28, 1)
    encoded_0 = encoder.predict(digits)
    encoded_1 = np.concatenate([encoded_0[0], encoded_0[1]], axis=1)
    encoded_2 = np.concatenate([encoded_0[2], encoded_0[3]], axis=1)
    encoded = np.concatenate([encoded_1, encoded_2], axis=0)
    encoded = (weight*logo + encoded)/(weight+1)

    return encoded

def main():
    window = Tk()
    window.title("Encode")
    window.geometry("400x400")

    label1 = Label(window, text = "1st Digit: ", font = ("Times New Roman",15))
    label1.grid(column = 0, row = 0)

    combo1 = Combobox(window, state="readonly")
    combo1['values'] = (0,1,2,3,4,5,6,7,8,9)
    combo1.current()
    combo1.grid(column = 5, row = 0)

    label2 = Label(window, text="2nd Digit: ", font=("Times New Roman", 15))
    label2.grid(column= 0, row=5)

    combo2 = Combobox(window, state="readonly")
    combo2['values'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    combo2.current()
    combo2.grid(column= 5 , row=5)

    label3 = Label(window, text="3rd Digit: ", font=("Times New Roman", 15))
    label3.grid(column=0, row=10)

    combo3 = Combobox(window, state="readonly")
    combo3['values'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    combo3.current()
    combo3.grid(column=5, row=10)

    label4 = Label(window, text="4th Digit: ", font=("Times New Roman", 15))
    label4.grid(column=0, row=15)

    combo4 = Combobox(window, state="readonly")
    combo4['values'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    combo4.current()
    combo4.grid(column=5, row=15)

    def clicked():
        digits_str = (str(combo1.get()) + str(combo2.get()) +
                      str(combo3.get()) + str(combo4.get()))

        # import MNIST data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype(float) / 255.
        x_test = x_test.astype(float) / 255.
        x_train = x_train.reshape(*x_train.shape, 1)
        x_test = x_test.reshape(*x_test.shape, 1)

        # parameters
        weight = 5
        logo = plt.imread('logo.png')
        digits = randomizer(digits_str, x=x_test, y=y_test)

        # import encoder and decoder
        encoder = load_model('models/encoder.h5')

        # create password image and decode
        digits_encoded = encode_image(digits, encoder, logo, weight=weight)

        # show and save decoded image vs original
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i in range(4):
            ax = axes.flatten()[i]
            ax.imshow(digits[i].reshape(28, 28))
            ax.axis('off')
        plt.savefig('digits.png', dpi=100, bbox_inches='tight')

        # save password image
        imsave('digits_encoded.png', digits_encoded)
        sys.exit()


    btn = Button(window, text = "GO", command = clicked,width = 21,height = 3, bd = 5)
    btn.grid(column = 5, row = 20)

    window.mainloop()

if __name__ == "__main__":
    main()

