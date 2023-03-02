Judgment_Continue_Gloves = []
Judgment_Continue_Mask = []
Judgment_Continue_Safe_Glasses = []
open_door_flag = False
count = 0
#读取开门前20帧
#list pop第一个元素  append 最后新来的元素
#如此反复进行下去
def gloves():
    print("带手套")

def mask():
    print("带口罩")

def  safe_glasses():
    print("戴眼镜")


# 各个状态对应的标签 0: s，1: m  ，2: c ，3: g ，4: ug ，5: us ，6: um ，7: o
def note_and_save(label_li):  # 已实现
    global count
    global Judgment_Continue_Gloves
    global Judgment_Continue_Mask
    global Judgment_Continue_Safe_Glasses
    #先拿到每一帧的检测结果
    #label_li 结果 [tensor(3.), tensor(0.), tensor(3.)]  这张图片 检测出来 3 0 3 手套 护目镜 手套
    #由于模型问题  某些帧会检测错误
    # 即 4 与 3 同时检测到 那么废弃这一帧
    #以 ug 为 1 作为判断依据
    # 30帧里面有一半是OK的 我都算他OK
    #我的判断逻辑很宽松
    if count <  30:
        if  4 in label_li  and  3  in  label_li:
            Judgment_Continue_Gloves.append(0) #两个同时出现 我也算他带了
        elif  3 in label_li:
            Judgment_Continue_Gloves.append(0)  #带了
        elif 4 in label_li:
            Judgment_Continue_Gloves.append(1)  #没带
        #这一帧里没这个标签 我也算他带了
        elif 4  not in label_li and  3 not in label_li:
            Judgment_Continue_Gloves.append(0)

        #口罩也是相同的逻辑
        if 1 in label_li and 6 in label_li:
            Judgment_Continue_Mask.append(0)
        elif 1 in label_li:
            Judgment_Continue_Mask.append(0)
        elif 6 in label_li:
            Judgment_Continue_Mask.append(1)
        elif 1 not in label_li and 6 not in label_li:
            Judgment_Continue_Mask.append(0)

        #眼镜 0s   5us
        if 0 in label_li and 5 in label_li:
            Judgment_Continue_Safe_Glasses.append(0)
        elif 0 in label_li:
            Judgment_Continue_Safe_Glasses.append(0)
        elif 5 in label_li:
            Judgment_Continue_Safe_Glasses.append(1)
        elif  0 not in label_li and 5 not in label_li:
            Judgment_Continue_Safe_Glasses.append(0)



    else:#开始栈 删除第一个元素 添加最后一个
        Judgment_Continue_Gloves.pop(0)#删除第一个
        Judgment_Continue_Safe_Glasses.pop(0)
        Judgment_Continue_Mask.pop(0)

        #像尾巴处添加新的  与 之前一样
        #手套
        if  4 in label_li  and  3  in  label_li:
            Judgment_Continue_Gloves.append(0) #两个同时出现 我也算他带了
        elif  3 in label_li:
            Judgment_Continue_Gloves.append(0)  #带了
        elif 4 in label_li:
            Judgment_Continue_Gloves.append(1)  #没带
        #这一帧里没这个标签 我也算他带了
        elif 4  not in label_li and  3 not in label_li:
            Judgment_Continue_Gloves.append(0)

        #口罩
        if 1 in label_li and 6 in label_li:
            Judgment_Continue_Mask.append(0)
        elif 1 in label_li:
            Judgment_Continue_Mask.append(0)
        elif 6 in label_li:
            Judgment_Continue_Mask.append(1)
        elif 1 not in label_li and 6 not in label_li:
            Judgment_Continue_Mask.append(0)

        #眼镜
        if 0 in label_li and 5 in label_li:
            Judgment_Continue_Safe_Glasses.append(0)
        elif 0 in label_li:
            Judgment_Continue_Safe_Glasses.append(0)
        elif 5 in label_li:
            Judgment_Continue_Safe_Glasses.append(1)
        elif  0 not in label_li and 5 not in label_li:
            Judgment_Continue_Safe_Glasses.append(0)


    #开门的一瞬间 判断前30帧 是不是有15帧是OK的
    #存储图片

    if 7 in label_li: #门开了
        if sum(Judgment_Continue_Gloves) >= 15: #30帧有15帧以上都违规
            #存储图片
            cv2.imwrite(vidcut_dir + 'UG_01.jpg', im0)
            gloves() #提醒戴手套
        if sum(Judgment_Continue_Mask) >= 15:
            #存储图片
            mask() #提醒戴口罩
        if sum(Judgment_Continue_Safe_Glasses) >= 15:
            #存储图片
            safe_glasses()#提醒戴眼镜

    count += 1 #记录每一帧
    return "Success Save && Note"


for i in range(0,30):
    note_and_save([1,2,3,4,5,6,7])

r = note_and_save([2,4,5,6,7])
print(r)


