#!/usr/bin/env python
import PySimpleGUI as sg
from PIL import Image
import cv2 as cv
import io




def play_veido(frame):
    sg.theme('Black')
    layout = [[sg.Text('Yolov5 Demo', size=(15, 1), font='Helvetica 20')],
              [sg.Image(filename='', key='-image-')],
              [sg.Button('Exit', size=(7, 1), pad=((600, 0), 3), font='Helvetica 14')]]

    window = sg.Window('Demo Application - YoloV5 Integration',
                       layout, no_titlebar=False, location=(0, 0))
    image_elem = window['-image-']
    event, values = window.read(timeout=0)
    imgbytes = cv.imencode('.png', frame)[1].tobytes()  # ditto
    image_elem.update(data=imgbytes)


