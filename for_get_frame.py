# import numpy as np
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
# import PySimpleGUI as sg
# import matplotlib.pyplot as plt
# from PIL import Image,ImageTk
# #生成一个fig
# def plot_image():
#     img = Image.open(r"C:\Users\Administrator\Desktop\Lenna.png")
#     gray = img.convert('L')
#     r,g,b = img.split()
#     # 绘图，figsize的单位为英寸
#     fig=plt.figure(facecolor ="white",figsize=(5,5))
#     plt.subplot(2,2,1)
#     plt.title('原图')
#     plt.imshow(img)
#     # plt.axis('on')
#     # plt.subplot(2,2,2)
#     # plt.title('灰度图')
#     # plt.imshow(gray,cmap='gray')
#     # plt.axis('on')
#     # plt.subplot(2,2,3)
#     # plt.title('红色通道')
#     # plt.imshow(r,cmap='gray')
#     # plt.axis('on')
#     # plt.subplot(2,2,4)
#     # plt.title('绿色通道')
#     # plt.imshow(g,cmap='gray')
#     plt.axis('on')
#     # 调整间距
#     plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.3,hspace=0.3)
#     # 用来正常显示中文标签
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     # 用来正常显示负号
#     plt.rcParams['axes.unicode_minus'] = False
#     return fig
# # 画在canvas上，包括工具条
# def draw_figure(canvas,canvas_toolbar, figure):
#     figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
#     figure_canvas_agg.draw()
#     toolbar = NavigationToolbar2Tk(figure_canvas_agg,canvas_toolbar)
#     toolbar.update()
#     figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
#     return figure_canvas_agg
# # 布局
# layout = [[sg.Canvas(key='-TOOLBAR-')],
# [sg.Canvas(key='-CANVAS-')]]
# window = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI',
# layout, finalize=True, element_justification='center',
# font='Helvetica 18')
# # 调用绘图函数
# fig = plot_image()
# fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, window['-TOOLBAR-'].TKCanvas,fig)
# #等待界面输入
# event, values = window.read()
# #关闭窗体
# window.close()


from datetime import datetime, date
dayOfWeek = datetime.now().weekday()
print(dayOfWeek)