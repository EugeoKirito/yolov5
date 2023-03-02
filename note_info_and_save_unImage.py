from PIL import ImageGrab
import time
# def note_and_save(label_li):
#     # 存在着路过的问题 只有停下来才发出警报
#     # 那么如何判断不是路过的人？
#     #并不需要每一帧都检测 ------> 如何实现？
#     #确实修改不了每一帧  只能用计数器计数 当count累积到20左右 语音提示
#     global count
#     if  4 in label_li  or  5 in  label_li or 6 in  label_li :
#         count +=  1
#     print(count,'ssssssssss')
#
#     if count > 20 and count<40:
#         if 4 in label_li  :  # ug   没带手套
#             # 蜂鸣器提示带手套
#             print("请佩戴手套")
#         if 5 in  label_li :  # us  没带护目镜
#             print("请佩戴护目镜")
#
#         if 6 in label_li :  # um  没带KN95
#             print("请佩戴 N92")
#
#
#
#     # if label_li == 7:  # 门开了  可以再次判断到底有没有带上
#     #     pass
#     return "Note and Save  Success!"






