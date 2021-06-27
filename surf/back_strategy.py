# coding: utf-8
from surf.script_tab import keytab
import os, json, time, re, codecs, glob
from surf.surf_tool import regex2pairs
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging.handlers
import pandas as pd
import itertools
import numpy as np
import lightgbm as lgb
import ffn


class BackFunc(object):
    # todo: enforce 需要分2部分，预测在当前，训练在训练模块。，回测

    def __init__(self):
        """各策略：只输出 close、比重、是否操作标记"""
        self.funcmap = {
            "cross": self.cross,
            "enforce": self.cross,
            "turtle": None,
        }

    def cross(self, dataobj, params, strategy_head):
        cn1, cn2 = params["close"]
        # vn1,vn2=params["volume"]
        dataobj["rise1"] = dataobj["close"] / dataobj["close"].shift(1) - 1
        dataobj["meanc1"] = dataobj["close"].rolling(window=cn1, center=False).mean()
        dataobj["meanc2"] = dataobj["close"].rolling(window=cn2, center=False).mean()
        oper_sig = "{}_oper_sig".format(strategy_head)
        oper_ratio = "{}_oper_ratio".format(strategy_head)
        dataobj[oper_sig] = 0
        dataobj[oper_ratio] = 0.0
        for id1 in range(1, len(dataobj)):
            hn = dataobj["meanc1"][id1]
            ho = dataobj["meanc1"][id1 - 1]
            ln = dataobj["meanc2"][id1]
            lo = dataobj["meanc2"][id1 - 1]
            if lo > ho and ln < hn:
                dataobj[oper_sig][id1] = 1
                # 简单的涨幅 转 概率
                dataobj[oper_ratio][id1] = dataobj["rise1"][id1] * 5 + 0.5
            elif lo < ho and ln > hn:
                dataobj[oper_sig][id1] = 1
                # 简单的涨幅 转 概率
                dataobj[oper_ratio][id1] = dataobj["rise1"][id1] * 5 + 0.5
            else:
                pass
        return dataobj

    def get_rewards(self, dataobj, strategy_head, trade_type=1):
        # trade_type 1 为单边涨， 2 为双向涨跌
        oper_sig = "{}_oper_sig".format(strategy_head)
        oper_ratio = "{}_oper_ratio".format(strategy_head)
        newcap = 1.0
        newmount = 0.0
        newreward = 1.0
        reward_list = [1.0]
        for id1 in range(1, len(dataobj)):
            oldreward = newreward
            if dataobj.iloc[id1][oper_sig] == 1:
                t_ratio = dataobj.iloc[id1][oper_ratio]
                if trade_type == 1 and t_ratio < 0:
                    t_ratio = 0.0
                newmount = oldreward * t_ratio / dataobj.iloc[id1]["close"]
                newcap = oldreward * (1 - t_ratio)
            newreward = newcap + newmount * dataobj.iloc[id1]["close"]
            reward_list.append(newreward)
        dataobj["{}_wealth".format(strategy_head)] = reward_list
        return dataobj

    def __call__(self, oriinfiles, commands, outhead):
        # 1. 只有两个文件
        strategy_names = [command[list(command.keys())[0]]["strategy_name"] for command in commands]
        outjson = {"filename": []}
        outjson.update({i1 + "年化": [] for i1 in strategy_names})
        outjson.update({i1 + "累计": [] for i1 in strategy_names})
        outjson.update({i1 + "回撤": [] for i1 in strategy_names})
        outjson.update({i1 + "夏普": [] for i1 in strategy_names})
        for infile in oriinfiles:
            pdobj = pd.read_csv(infile, header=0, index_col="date", encoding="utf8")
            for command in commands:
                tkey = list(command.keys())[0]
                tval = list(command.values())[0]
                strategy_head = command[tkey]["strategy_name"]
                pdobj = self.funcmap[tkey](pdobj, tval, strategy_head)
                pdobj = self.get_rewards(pdobj, strategy_head)
            (filepath, tfilename) = os.path.split(infile)
            outjson["filename"].append(tfilename)
            fname = os.path.join(filepath, "{}{}".format(outhead, tfilename))
            pdobj.to_csv(fname, index=True, header=True, encoding="utf-8")
            pdlenth = len(pdobj)
            wealth_cols = [i2 for i2 in pdobj.columns if re.search("_wealth$", i2)]
            for id2, colname in enumerate(wealth_cols):
                annRet = np.power(pdobj[colname][-1], 252 / pdlenth)
                sharpe = ffn.calc_risk_return_ratio(pdobj[colname])
                outjson[strategy_names[id2] + "年化"].append(annRet)
                outjson[strategy_names[id2] + "累计"].append(pdobj[colname][-1])
                outjson[strategy_names[id2] + "夏普"].append(sharpe)
                outjson[strategy_names[id2] + "回撤"].append(ffn.calc_max_drawdown(pdobj[colname]))
        pdobjout = pd.DataFrame(outjson)
        pdobjout.set_index("filename", inplace=True)
        (filepath, _) = os.path.split(oriinfiles[0])
        fname = os.path.join(filepath, "{}回测统计.csv".format(outhead))
        pdobjout.to_csv(fname, index=True, header=True, encoding="utf-8")
        return None


back_func = {
    "回测分析": BackFunc(),
}
