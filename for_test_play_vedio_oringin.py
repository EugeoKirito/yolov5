#!/usr/bin/env python
import PySimpleGUI as sg
from PIL import Image
import cv2 as cv
import io

def play_veido(frame):
    # ---===--- Get the filename --- #
    # 调用popup_get_file，读取视频文件，popup_get_file返回值为文件名，若文件名为空，则返回

    #调用VideoCapture读取视频文件

    vidFile = cv.VideoCapture(0)
    # ---===--- Get some Stats --- #
    #获取视频文件的帧数和帧率
    num_frames = vidFile.get(cv.CAP_PROP_FRAME_COUNT)
    fps = vidFile.get(cv.CAP_PROP_FPS)

    sg.theme('Black')

    # ---===--- define the window layout --- #
    #界面布局，按钮默认提交event
    layout=[[sg.Text('Yolov5 Demo', size=(15, 1), font='Helvetica 20')],
            [sg.Image(filename='', key='-image-')],

            [sg.Button('Exit', size=(7, 1), pad=((600, 0), 3), font='Helvetica 14')]]

    # create the window and show it without the plot
    window = sg.Window('Demo Application - YoloV5 Integration',
              layout, no_titlebar=False, location=(0, 0))

    # locate the elements we'll be updating. Does the search only 1 time
    #变量替换，为了后续写起来简单点
    image_elem = window['-image-']


    # ---===--- LOOP through video file by frame --- #
    #若视频视频打开，则进入下面的处理循环
    cur_frame = 0
    while vidFile.isOpened():
        #界面消息读取，event为界面操作事件，若某一控件的enable_events=True，
        # 当操作此控件时，消息队列可读取到该event；values带有消息内容
        event, values = window.read(timeout=0)
        if event in ('Exit', None):
            break
        ret, frame = vidFile.read()
        if not ret:  # if out of data stop looping
            break
        # if someone moved the slider manually, the jump to that frame

        cur_frame += 1


        #将视频帧转化为Image组件可以显示的数据，并送出显示
        imgbytes = cv.imencode('.png', frame)[1].tobytes()  # ditto
        image_elem.update(data=imgbytes)

