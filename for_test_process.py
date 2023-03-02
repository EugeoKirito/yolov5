#
Judgment_Continue_Mask = []
Judgment_Continue_Safe_Glasses = []
Judgment_Continue_Gloves = []
open_door_flag = None
count = 0
save_count = 0
def note_and_save(label_li):  # 已实现
    count += 1
    # 各个状态对应的标签 0: s，1: m  ，2: c ，3: g ，4: ug ，5: us ，6: um ，7: o
    #判断开门状态  2 &&  7

    #连续10帧判断  最后sum(list) >= 8 那么确实是 没带
    if 4 in label_li:
        Judgment_Continue_Gloves.append(1)
    else:
        Judgment_Continue_Gloves.append(0)

    if 5 in label_li:
        Judgment_Continue_Safe_Glasses.append(1)
    else:
        Judgment_Continue_Safe_Glasses.append(0)

    if 6 in label_li:
        Judgment_Continue_Mask.append(1)
    else:
        Judgment_Continue_Mask.append(0)

    if count // 10 : # 10帧后还原判断
        Judgment_Continue_Safe_Glasses = []
        Judgment_Continue_Mask = []
        Judgment_Continue_Gloves = []
        count = 0


    if 7 in label_li: #门开了  检测与报警 8~10帧数 报警一次
        #手套没带 10帧里面有8帧
        save_count += 1
        if sum(Judgment_Continue_Gloves) >= 8:
            #保存 手套没带
            if save_count > 1:
                cv2.imwrite(vidcut_path + '01.jpg', im0)
                print("Save Pic01 Success")
            try:
                glove()
            except KeyboardInterrupt:
                pass
        if sum(Judgment_Continue_Safe_Glasses) >= 8:
            if save_count >1:
                cv2.imwrite(vidcut_path + '02.jpg', im0)
            try:
                safe_glasses()
            except KeyboardInterrupt:
                pass
        if sum(Judgment_Continue_Safe_Glasses) >= 8:
            if save_count > 1:
                cv2.imwrite(vidcut_path + '03.jpg', im0)
            try:
                mask()
            except KeyboardInterrupt:
                pass
    else:
        save_count = 0

















