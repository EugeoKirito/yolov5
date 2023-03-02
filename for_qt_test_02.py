




#files = r'c087Woman_safe.png'
# files = None
#
# def get_in(return_fileName):
#     global files
#     files  = return_fileName
# print('-------qt----------------------',files)



#捕获 文件夹的内容 一旦有更新
#便把 照片给捕获进来



import PySimpleGUI as sg
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import  matplotlib.pyplot
from PIL import Image
import cv2 as cv
import io
"""
    Dashboard using blocks of information.

    Copyright 2020 PySimpleGUI.org
"""

'''matplot init'''
files = r'C:\Users\Administrator\Desktop\087Woman_safe.png'
from PIL import Image
im = Image.open(files)
image=im.resize((433,220))
image.save(files)

'''matplot init'''
matplotlib.use('TkAgg')
fig = matplotlib.figure.Figure(figsize=(4.3, 2.6), dpi=100)
t = [1,2,3,4,5,6,7]

count_for_abnormal = [1,0,3,0,5,0,4]
from datetime import datetime, date
dayOfWeek = datetime.now().weekday()
fig.add_subplot(111).plot(t, count_for_abnormal)

'''end matplot init'''

theme_dict = {'BACKGROUND': '#2B475D',
                'TEXT': '#FFFFFF',
                'INPUT': '#F2EFE8',
                'TEXT_INPUT': '#000000',
                'SCROLL': '#F2EFE8',
                'BUTTON': ('#000000', '#C2D4D8'),
                'PROGRESS': ('#FFFFFF', '#C7D5E0'),
                'BORDER': 1,'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}
sg.LOOK_AND_FEEL_TABLE['Dashboard'] = theme_dict
sg.theme('Dashboard')
BORDER_COLOR = '#C7D5E0'
DARK_HEADER_COLOR = '#1B2838'
BPAD_TOP = ((20,20), (20, 10))
BPAD_LEFT = ((20,10), (0, 10))
BPAD_LEFT_INSIDE = (0, 10)
BPAD_RIGHT = ((10,20), (10, 20))
top_banner = [[sg.Text('S02_1E_BIM'+ ' '*64, font='Any 20', background_color=DARK_HEADER_COLOR)
               ]]
top  = [[sg.Text('                Smart Detect Result', size=(50,50), justification='c', pad=BPAD_TOP, font='Any 20')],
            #[sg.T(f'{i*25}-{i*34}') for i in range(7)],]
        ]

block_3 = [[sg.Text('pic', font='Any 20')],
           [sg.Image(filename=files,key="-abnormal-image-")],
              ]
block_2 = [[sg.Text('违规次数统计图', font='Any 20')],
           [sg.Canvas(key='-CANVAS-')]
            ]
block_5 = [[sg.Text('Yolov5 Demo', size=(15, 1), font='Helvetica 20')],
            [sg.Image(filename='', key='-image-')],
           ]

layout = [[sg.Column(top_banner, size=(1050, 38), pad=(0,0), background_color=DARK_HEADER_COLOR)],
          [sg.Column(top, size=(1010, 80), pad=BPAD_TOP)],

          [sg.Column([[sg.Column(block_2, size=(450,300), pad=BPAD_LEFT_INSIDE)],

          [sg.Column(block_3, size=(450,250),  pad=BPAD_LEFT_INSIDE)]],
            pad=BPAD_LEFT, background_color=BORDER_COLOR),

           sg.Column(block_5, size=(540, 570), pad=BPAD_RIGHT)

           ]]

window = sg.Window('Smart Detect Result Dashboard ', layout, margins=(0,0), background_color=BORDER_COLOR, no_titlebar=True, grab_anywhere=True,finalize=True)

image_elem = window['-image-']
pic_elem = window['-abnormal-image-']


figure_canvas_agg = FigureCanvasTkAgg(fig, window['-CANVAS-'].TKCanvas)
figure_canvas_agg.draw()
figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)

vidFile = cv.VideoCapture(0)
cur_frame = 0

 #更新pic
abnormal_pic = img = cv.imread(r'C:\Users\Administrator\Desktop\025Woman_US_UM_UG.png')
imgbytes2 = cv.imencode('.png',abnormal_pic)[1].tobytes()
pic_elem.update(data=imgbytes2)

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








