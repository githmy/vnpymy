from surf.script_tab import keytab
import os, json, time, re, codecs
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging.handlers
import pandas as pd

mpl.rcParams[u'font.sans-serif'] = u'SimHei'
mpl.rcParams[u'axes.unicode_minus'] = False

baspath = os.path.join("..", "..", "nocode", "vnpy")
projectpath = None

funcmap = {

}


def check_keytab(injson):
    " 输入检测格式函数 "
    erromessage = None
    for linejson in injson:
        for key, val in linejson.items():
            if key not in keytab["功能"]:
                erromessage = f'{key} not in {keytab["功能"]}'
                return erromessage
            funckey = keytab[key].keys()
            for com, vv in val.items():
                if com not in funckey:
                    erromessage = f'{com} not in {keytab[com]}'
                    return erromessage
                if not isinstance(vv, type(keytab[key][com])):
                    erromessage = f'the type of {com} value {vv} is not {type(keytab[key][com])}'
                    return erromessage
    return erromessage


def func_implement(injson):
    " 按脚本执行 map 函数 "
    logger1.info("input: {}".format(injson))
    # 1. 找出重启语句
    restart_name = None
    restart_order = None
    for line in injson:
        if "重启" in line.keys():
            restart_name = line["重启"]["功能名"]
            restart_order = line["重启"]["排序号"]
            break
    # 2. 找到其实行
    find_counter = 0
    start_id = 0
    for idn, line in enumerate(injson):
        if restart_name in line.keys():
            find_counter += 1
            if restart_order == find_counter:
                start_id = idn
                break
    # 3. 按顺序加载执行功能
    # 数据文件只处理csv， 因为excel 长度受限。
    for idn, line in enumerate(injson):
        if idn < start_id:
            logger1.info("pass command {}".format(line))
            continue
        logger1.info("running command {}".format(line))
        part_name = list(line.keys())[0]
        if part_name == "数据处理":
            commands = line[part_name]
            indatafilelist = commands["输入数据"]
            for indatafile in indatafilelist:
                pdobj = pd.read_csv(os.path.join(projectpath,), header=0, encoding="utf8")
            indatafilelist = commands["处理方法"]
            indatafilelist = commands["输出数据"]
            indatafilelist = commands["输出性能"]

        elif part_name == "训练拟合":
            commands = line[part_name]
            indatafilelist = commands["输入数据"]
        elif part_name == "数据预测":
            commands = line[part_name]
            indatafilelist = commands["输入数据"]
        elif part_name == "回测分析":
            commands = line[part_name]
            indatafilelist = commands["输入数据"]
        elif part_name == "图形展示":
            commands = line[part_name]
            indatafilelist = commands["输入数据"]
        else:
            pass

    return None


def finish_info(filepath, use_time, runinfo):
    " 结束通知的函数 "
    logger1.info(f"use_time:{use_time}mins.")
    return None
    title = f"{filepath}完成。use_time:{use_time}mins. \n{runinfo}"
    plt.figure(facecolor='w')
    plt.plot([], [])
    plt.title(title)
    plt.show()


def main(filepath):
    " 程序框架 "
    # 1. 加载脚本
    starttime = time.time()
    inhand = codecs.open(filepath, "r", "utf8")
    incont = inhand.readlines()
    finalstr = ""
    for line in incont:
        tstr = re.split('//', line)
        finalstr += tstr[0]
    injson = json.loads(finalstr, encoding="utf8")
    # 2. 检查格式
    checkres = check_keytab(injson)
    if checkres is not None:
        logger1.info(checkres)
        raise Exception(checkres)
    # 3. 功能映射
    runinfo = func_implement(injson)
    # 4. 完成提示
    use_time = (time.time() - starttime) / 60
    finish_info(filepath, use_time, runinfo)


if __name__ == '__main__':
    filepath = os.path.join("demo_script1.json")
    # 1. 全局日志
    filehead = ".".join(filepath.split(".")[:-1])
    projectpath = os.path.join(baspath, filehead)
    datalogfile = os.path.join(projectpath, f'{filehead }.log')
    if not os.path.exists(projectpath):
        os.makedirs(projectpath)
    if os.path.isfile(datalogfile):
        os.remove(datalogfile)
    logger1 = logging.getLogger('log')
    logger1.setLevel(logging.DEBUG)
    fh = logging.handlers.RotatingFileHandler(datalogfile, maxBytes=104857600, backupCount=10)
    ch = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger1.addHandler(fh)
    logger1.addHandler(ch)

    # 2. 功能实现
    main(filepath)
