import os, glob
import pandas as pd


def regex2pairs(oriinfiles, projectpath):
    # 正则列出
    compare_lenth = len(oriinfiles)
    # 1. 精简文件
    strip_infiles = [i1.strip("*") for i1 in oriinfiles]
    f_index = 0
    b_index = 0
    f_flag = 0
    b_flag = 0
    for i1 in range(0, len(strip_infiles[0])):
        for i2 in range(1, compare_lenth):
            if f_flag == 0:
                if strip_infiles[0][i1] != strip_infiles[i2][i1]:
                    f_flag = 1
                else:
                    f_index += 1
            if b_flag == 0:
                if strip_infiles[0][-i1 - 1] != strip_infiles[i2][-i1 - 1]:
                    b_flag = 1
                else:
                    b_index += 1
            if f_flag == 1 and b_flag == 1:
                break
    extract_file = [i1[f_index:len(i1) - b_index] for i1 in strip_infiles]
    extract_head = strip_infiles[0][0:f_index]
    extract_tail = strip_infiles[0][len(strip_infiles[0]) - b_index:]
    # 2. 原始分割对比列表
    comparelist = []
    for i1 in oriinfiles:
        tkey = i1.strip("*")
        tmatch = glob.glob(os.path.join(projectpath, i1))
        tsplit = [i2.split(tkey) for i2 in tmatch]
        tkey_list = []
        for id2, i2 in enumerate(tsplit):
            tkey_list.append([tkey, tkey.join(i2[0:-1]), i2[-1]])
        comparelist.append(tkey_list)
    # 3. 匹配合成
    matchstrlist = []
    for i1 in comparelist[0]:
        match_n = 0
        for i2 in range(1, compare_lenth):
            for i3 in comparelist[i2]:
                if i1[1] == i3[1] and i1[2] == i3[2]:
                    # 有匹配，退出当前循环，计数，找下一个关键词 数组
                    match_n += 1
                    break
            if match_n != i2:
                # 未找到匹配，停止当前的 i1的循环，找下一个i1的
                break
        if match_n == compare_lenth - 1:
            matchstrlist.append([i1[1] + extract_head, extract_file, extract_tail + i1[2]])
    # 4. 数据加载
    pdobjlist = []
    for i1 in matchstrlist:
        pdobjs = []
        for ex_file in extract_file:
            tpd = pd.read_csv(i1[0] + ex_file + i1[2], header=0, index_col=0, encoding="utf8")
            pdobjs.append(tpd)
        pdobjlist.append(pdobjs)
    return pdobjlist, matchstrlist
