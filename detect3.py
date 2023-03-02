# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import time
import torch
import numpy as np

import threading
from for_draw_staistic import draw_statistic





try:
    import sys

    sys.path.append('/opt/nvidia/jetson-gpio/lib/python/')
    sys.path.append('/opt/nvidia/jetson-gpio/lib/python/Jetson/GPIO')
    import Jetson.GPIO as GPIO
except Exception as e:
    print(e)
from multiprocessing import Process

led1 = 15
led2 = 16
led3 = 18


# 设置GPIO初始口
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(led1, GPIO.OUT)
# GPIO.setup(led2, GPIO.OUT)
# GPIO.setup(led3, GPIO.OUT)


def gloves():
    print("----------带手套----------------")

def mask():
    print("-----------带口罩---------")

def  safe_glasses():
    print("-----戴眼镜-------")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

Judgment_Continue_Gloves = []
Judgment_Continue_Mask = []
Judgment_Continue_Safe_Glasses = []
Judgment_Continue_Open_Door = []
count = 0
count_frame_gloves = 0
count_frame_mask = 0
count_frame_glasses = 0
count_abnormal = 0




import PySimpleGUI as sg
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import  matplotlib.pyplot
from PIL import Image
import cv2 as cv
import io
#得设置 while True在def run函数里并设置multiplication   至于更新控件在im0处

'''matplot init'''
from datetime import datetime, date
dayOfWeek = datetime.now().weekday()
'''end matplot init'''
from PIL import Image
import pandas as pd
data_info_csv = pd.read_csv('./info.csv')



#外层先 设置layout与 主题
theme_dict = {'BACKGROUND': '#2B475D',
                'TEXT': '#FFFFFF',
                'INPUT': '#F2EFE8',
                'TEXT_INPUT': '#000000',
                'SCROLL': '#F2EFE8',
                'BUTTON': ('#000000', '#C2D4D8'),
                'PROGRESS': ('#FFFFFF', '#C7D5E0'),
                'BORDER': 1,'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}
sg.LOOK_AND_FEEL_TABLE['Dashboard'] = theme_dict
BORDER_COLOR = '#C7D5E0'
DARK_HEADER_COLOR = '#1B2838'
BPAD_TOP = ((20,20), (20, 10))
BPAD_LEFT = ((20,10), (0, 10))
BPAD_LEFT_INSIDE = (0, 10)
BPAD_RIGHT = ((10,20), (10, 20))

top_banner = [[sg.Text('S02_1E_BIM'+ ' '*64, font='Any 20', background_color=DARK_HEADER_COLOR),
               sg.Text('Tuesday 9 June 2020', font='Any 20', background_color=DARK_HEADER_COLOR)]]

top  = [[sg.Text('                Smart Detect Result', size=(50,50), justification='c', pad=BPAD_TOP, font='Any 20')]]
block_2 = [[sg.Text('近期违规图片', font='Any 20')],
           [sg.Image(filename='',key="-abnormal-image-")],
              ]
block_3 = [[sg.Text('违规次数统计图', font='Any 20')],
           [sg.Image(key='-static-image-')]
            ]
block_4 = [[sg.Text('Yolov5 Demo', size=(15, 1), font='Helvetica 20')],
              [sg.Image(filename='', key='-image-')],
              [sg.Button('Exit', size=(7, 1), pad=((600, 0), 3), font='Helvetica 14')]]
layout = [[sg.Column(top_banner, size=(1050, 38), pad=(0,0), background_color=DARK_HEADER_COLOR)],
          [sg.Column(top, size=(1010, 80), pad=BPAD_TOP)],
          [sg.Column([[sg.Column(block_3, size=(450,300), pad=BPAD_LEFT_INSIDE)],
                      [sg.Column(block_2, size=(450,250),  pad=BPAD_LEFT_INSIDE)]], pad=BPAD_LEFT, background_color=BORDER_COLOR),
           sg.Column(block_4, size=(540, 570), pad=BPAD_RIGHT)]
          ]



@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5x.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)


    window = sg.Window('Smart Detect Result Dashboard ', layout, margins=(0, 0), background_color=BORDER_COLOR,
                       no_titlebar=True, grab_anywhere=True, )
    '''Draw  直接使用图片了'''

    '''End Draw'''

    image_elem = window['-image-'] #Vedio Element
    pic_elem = window['-abnormal-image-'] #Pic Element

    statis_elem = window['-static-image-']

    '''resize statistic picture'''
    files = './save_pic/statistic.png'
    im = Image.open(files)
    image = im.resize((433, 260))
    image.save(files)
    statis_pic = cv.imread('./save_pic/statistic.png')
    img_sta = cv.imencode('.png', statis_pic)[1].tobytes()


    def gui_run():
        while True:
            event, values = window.read()

    t1 = threading.Thread(target=gui_run)
    t1.start()

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())


    for path, im, im0s, vid_cap, s in dataset:

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions



        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop

            #imc is vedio play


            annotator = Annotator(im0, line_width=line_thickness, example=str(names))


            if len(det):

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    vidcut_dir = str(save_dir) + '/vidcut'
                    if not os.path.exists(vidcut_dir):
                        os.makedirs(vidcut_dir)
                    vidcut_path = vidcut_dir + '\\' + str(p.stem) + (
                        '' if dataset.mode == 'image' else f'_{frame}')
                try:
                    # 把label列表 送入note_save 中
                    def test_for_process(t):
                        while True:
                            print(t)

                    statis_elem.update(data=img_sta)

                    def note_and_save(label_li):  # 已实现
                        global dayOfWeek
                        global count
                        global Judgment_Continue_Gloves
                        global Judgment_Continue_Mask
                        global Judgment_Continue_Safe_Glasses
                        global Judgment_Continue_Open_Door
                        global count_frame_gloves
                        global count_frame_mask
                        global count_frame_glasses
                        global count_abnormal
                        #30帧之后 count30后  reset 这些计数器
                        if count  % 30 == 0:
                            count_frame_glasses = 0
                            count_frame_gloves = 0
                            count_frame_mask = 0
                        print("G",len(Judgment_Continue_Gloves))
                        print('O',len(Judgment_Continue_Open_Door))
                        print('M',len(Judgment_Continue_Mask))
                        print('S_G',len(Judgment_Continue_Safe_Glasses))
                        print('count',count,'+++++++++++++++++')
                        print('Gloves  ',Judgment_Continue_Gloves)
                        #10帧都是门开了
                        if count<25:
                            if 7 in label_li and 2 in label_li:
                                print("Append-------------------------")
                                Judgment_Continue_Open_Door.append(0)
                            elif 7 in label_li:
                                Judgment_Continue_Open_Door.append(1)
                            else:
                                Judgment_Continue_Open_Door.append(0)
                        if count>25:
                            print('Delete.......................')
                            del Judgment_Continue_Open_Door[0]   #删除第一个
                            if 7 in label_li and 2 in label_li:
                                Judgment_Continue_Open_Door.append(0)
                            elif 7 in label_li:
                                Judgment_Continue_Open_Door.append(1)
                            else:
                                Judgment_Continue_Open_Door.append(0)


                        # 先拿到每一帧的检测结果
                        # label_li 结果 [tensor(3.), tensor(0.), tensor(3.)]  这张图片 检测出来 3 0 3 手套 护目镜 手套
                        # 由于模型问题  某些帧会检测错误
                        # 即 4 与 3 同时检测到 那么废弃这一帧
                        # 以 ug 为 1 作为判断依据
                        # 30帧里面有一半是OK的 我都算他OK
                        # 我的判断逻辑很宽松
                        if count < 20:
                            if 4 in label_li and 3 in label_li:
                                Judgment_Continue_Gloves.append(0)  # 两个同时出现 我也算他带了
                            elif 3 in label_li:
                                Judgment_Continue_Gloves.append(0)  # 带了
                            elif 4 in label_li:
                                Judgment_Continue_Gloves.append(1)  # 没带
                            # 这一帧里没这个标签 我也算他带了
                            elif 4 not in label_li and 3 not in label_li:
                                Judgment_Continue_Gloves.append(0)

                            # 口罩也是相同的逻辑
                            if 1 in label_li and 6 in label_li:
                                Judgment_Continue_Mask.append(0)
                            elif 1 in label_li:
                                Judgment_Continue_Mask.append(0)
                            elif 6 in label_li:
                                Judgment_Continue_Mask.append(1)
                            elif 1 not in label_li and 6 not in label_li:
                                Judgment_Continue_Mask.append(0)

                            # 眼镜 0s   5us
                            if 0 in label_li and 5 in label_li:
                                Judgment_Continue_Safe_Glasses.append(0)
                            elif 0 in label_li:
                                Judgment_Continue_Safe_Glasses.append(0)
                            elif 5 in label_li:
                                Judgment_Continue_Safe_Glasses.append(1)
                            elif 0 not in label_li and 5 not in label_li:
                                Judgment_Continue_Safe_Glasses.append(0)



                        if count > 20:  # 开始栈 删除第一个元素 添加最后一个
                            del Judgment_Continue_Gloves[0]  # 删除第一个
                            del Judgment_Continue_Safe_Glasses[0]
                            del Judgment_Continue_Mask[0]

                            # 像尾巴处添加新的  与 之前一样
                            # 手套
                            if 4 in label_li and 3 in label_li:
                                Judgment_Continue_Gloves.append(0)  # 两个同时出现 我也算他带了
                            elif 3 in label_li:
                                Judgment_Continue_Gloves.append(0)  # 带了
                            elif 4 in label_li:
                                Judgment_Continue_Gloves.append(1)  # 没带
                            # 这一帧里没这个标签 我也算他带了
                            elif 4 not in label_li and 3 not in label_li:
                                Judgment_Continue_Gloves.append(0)

                            # 口罩
                            if 1 in label_li and 6 in label_li:
                                Judgment_Continue_Mask.append(0)
                            elif 1 in label_li:
                                Judgment_Continue_Mask.append(0)
                            elif 6 in label_li:
                                Judgment_Continue_Mask.append(1)
                            elif 1 not in label_li and 6 not in label_li:
                                Judgment_Continue_Mask.append(0)

                            # 眼镜
                            if 0 in label_li and 5 in label_li:
                                Judgment_Continue_Safe_Glasses.append(0)
                            elif 0 in label_li:
                                Judgment_Continue_Safe_Glasses.append(0)
                            elif 5 in label_li:
                                Judgment_Continue_Safe_Glasses.append(1)
                            elif 0 not in label_li and 5 not in label_li:
                                Judgment_Continue_Safe_Glasses.append(0)

                        # 开门的一瞬间 判断前30帧 是不是有15帧是OK的
                        # 存储图片

                        #这个地方有问题   老是open 与 close 一起出来
                        #且  语音提示应该设置延迟  读完 一句话 才进行下一句
                        #这两个问题得 大修改
                        #连续10帧门都是开的 才判断门是开的
                        #在此 提示与存储图片
                        print('This is OpenDoar',Judgment_Continue_Open_Door)
                        if sum(Judgment_Continue_Open_Door) >= 15:  # 门开了
                            print("{{{{{{{{{{{{{{{{{{{{{门开了")
                            #门开了  可以提示了
                            if sum(Judgment_Continue_Mask) >= 5:
                                safe_glasses()
                                cv2.imwrite('./save_pic/' + 'UM_01.png', im0)


                            if sum(Judgment_Continue_Gloves) >= 5:
                                gloves()
                                cv2.imwrite('./save_pic/' + 'UG_01.png', im0)
                            if sum(Judgment_Continue_Safe_Glasses) >= 5:
                                safe_glasses()
                                cv2.imwrite('./save_pic/' + 'US_01.png', im0)

                        #有门挡住了  那么如何精确判断？  恐怕要计算 数组大于5的次数了
                        #超过多少次 判定违规
                        if sum(Judgment_Continue_Gloves) >= 5:  # 30帧有15帧以上都违规
                                count_frame_gloves += 1  #
                        if sum(Judgment_Continue_Mask) >= 5:
                                count_frame_mask  += 1
                        if sum(Judgment_Continue_Safe_Glasses)  >= 5:
                                count_frame_glasses  += 1


                        #这个地方就留保存图片 与显示图片
                        print('Frame  ======>',count_frame_gloves)

                        if count_frame_gloves >= 5:
                            #数组次数大于5 就让他截取违规图
                            count_frame_gloves = 0
                            pic = cv.resize(im0, (433, 220), interpolation=cv2.INTER_CUBIC)
                            abnormal_pic = cv.imencode('.png', pic)[1].tobytes()
                            pic_elem.update(data=abnormal_pic)
                        if count_frame_mask  >= 5:
                            count_frame_mask = 0
                            pic = cv.resize(im0, (433, 220), interpolation=cv2.INTER_CUBIC)
                            abnormal_pic = cv.imencode('.png', pic)[1].tobytes()
                            pic_elem.update(data=abnormal_pic)
                        if count_frame_glasses >= 5:
                            count_frame_glasses = 0
                            pic = cv.resize(im0, (433, 220), interpolation=cv2.INTER_CUBIC)
                            abnormal_pic = cv.imencode('.png', pic)[1].tobytes()
                        #此处的逻辑 究竟是应该 按照 count_frame来编写
                        #还是 应该 按照US  Um来编写？
                        #我觉得 应该按照 count-frame 来计数
                        #提醒的功能应该放在  门开的那里    以及保存图片的功能也应该放在那里面


                                # 存储图片
                            #     # cv2.imwrite(vidcut_dir + 'UG_01.jpg', im0)
                            #     cv2.imwrite('./save_pic/' + 'UG_01.png', im0)
                            #     pic_elem.update(data=abnormal_pic)
                            #     gloves()  # 提醒戴手套
                            #     #在此更新 GUI 界面的图片
                            #
                            # if sum(Judgment_Continue_Mask) >= 5:
                            #     # 存储图片
                            #     # cv2.imwrite(vidcut_dir + 'UM_01.jpg', im0)
                            #     cv2.imwrite('./save_pic/' + 'UM_01.png', im0)
                            #
                            #     pic_elem.update(data=abnormal_pic)
                            #     mask()  # 提醒戴口罩
                            # if sum(Judgment_Continue_Safe_Glasses) >= 5:
                            #     # 存储图片
                            #     # cv2.imwrite(vidcut_dir + 'US_01.jpg', im0)
                            #     cv2.imwrite('./save_pic/' + 'US_01.png', im0)
                            #
                            #     pic_elem.update(data=abnormal_pic)
                            #     safe_glasses()  # 提醒戴眼镜
                        count += 1  # 记录每一帧

                        return "Success Save && Note"

                    label_list = []
                    # 可能性太低的标签 也不要 ----> 评估<0.5的都是无效标签(也有可能是有效标签)
                    # ug 误判问题！   ug在某个特定帧数   误判为   0.8以上
                    # 如何修正？  ug问题  其实也不用修 因为太短了 count加一或者加二无影响



                    try:
                        #print(det)
                        for info_single_li in reversed(det):
                            label_list.append(info_single_li[-1])
                    except Exception as e:
                        label_list = []
                    #print(label_list)  # [tensor(3.), tensor(0.), tensor(3.)]  这张图片 检测出来 3 0 3 手套 护目镜 手套
                    #resolove_info = note_and_save(label_list)  # 把每一张/每一帧 pic 检测的结果 丢进函数中
                    t2 = threading.Thread(target=note_and_save,args=(label_list,))
                    t2.start()


                except Exception as e:
                    # 如果那一帧没有检测到 label 就打印None
                    print("报错: ", e)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    '''
                    x y 坐标 以及label名称
                    '''
                    x0 = (int(xyxy[0].item()) + int(xyxy[2].item())) / 2
                    y0 = (int(xyxy[1].item()) + int(xyxy[3].item())) / 2  # 中心点坐标(x0, y0)
                    chang = int(xyxy[2].item()) - int(xyxy[0].item())
                    kuan = int(xyxy[3].item()) - int(xyxy[1].item())
                    label = int(cls)  # 对应每个物体的标签对应的数字label，如person:0
                    # print(label) #
                    # 返回处理结果 进行下一步是否要通报龙卷风

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            # im0 is the stream result
            # cv2.imshow('233',im0)
            #我在这里要做的事就是更新图片就行了 并不需要打开窗口

            imgbytes = cv.imencode('.png', im0)[1].tobytes()  # ditto
            image_elem.update(data=imgbytes)

            #在此处更新统计数据！




            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                #显示视频
                #cv2.imshow(str(p), im0)
                #cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 开门不带防护用具 保存视频，然后再把视频转化成部分帧

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]

                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp10/weights/best.pt',
                        help='model path or triton URL')
    # pic
    # parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    # cam
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob/screen/0(webcam)')

    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.60, help='NMS IoU threshold')#0.45
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results', default=True)
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos', default=True)
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":

    opt = parse_opt()
    main(opt)

