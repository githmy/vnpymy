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


class TrainFunc(object):
    def __init__(self):
        self.funcmap = {
            "lgboost": self.lgboost,
            "enforce_tab": self.enforce_tabf,
            "enforce_net": self.enforce_netf,
            # "tcn": None,
            # "tabnet": None,
        }
    def enforce_tabf(self, dataobjs, params, outhead, projectpath):
        pass
        return loss_result

    def enforce_netf(self, dataobjs, params, outhead, projectpath):
        pass
        return loss_result

    def lgboost(self, dataobjs, params, outhead, projectpath):
        train_X = []
        train_y = []
        val_X = []
        val_y = []
        collist = dataobjs[0][0].columns
        colchar = [i1 for i1 in collist if not re.search("^label_", i1, re.M)]
        collabel = [i1 for i1 in collist if re.search("^label_", i1, re.M)]
        for ttrain, tval in dataobjs:
            ttrain.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
            tval.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
            train_X.append(ttrain[colchar])
            train_y.append(ttrain[collabel])
            val_X.append(tval[colchar])
            val_y.append(tval[collabel])
        train_X = pd.concat(train_X, axis=0)
        train_y = pd.concat(train_y, axis=0)
        val_X = pd.concat(val_X, axis=0)
        val_y = pd.concat(val_y, axis=0)
        loss_result = {}
        for id1, i1 in enumerate(collabel):
            print("training:", i1)
            evals_result = {}
            lgtrain = lgb.Dataset(train_X, label=train_y.iloc[:, id1])
            lgval = lgb.Dataset(val_X, label=val_y.iloc[:, id1])
            model = lgb.train(params, lgtrain, 1000, valid_sets=lgval, early_stopping_rounds=100,
                              verbose_eval=100, evals_result=evals_result)
            fulpath = os.path.join(projectpath, "{}lgboost_{}.txt".format(outhead, i1))
            print("saving ", i1)
            print(fulpath)
            model.save_model(fulpath)
            # fig2 = plt.figure(figsize=(20, 20))
            # ax = fig2.subplots()
            # lgb.plot_tree(model, tree_index=1, ax=ax)
            # plt.show()
            # lgb.create_tree_digraph(model, tree_index=1)
            # print('画出训练结果...')
            # # lgb.plot_metric(evals_result, metric='auc')
            # lgb.plot_metric(evals_result, metric='rmse')
            # plt.show()
            # print('画特征重要性排序...')
            # ax = lgb.plot_importance(model, max_num_features=20)
            # # max_features表示最多展示出前10个重要性特征，可以自行设置
            # plt.show()
            loss_result[i1] = evals_result['valid_0']["rmse"][-1]
        return loss_result

    def __call__(self, oriinfiles, commands, outhead, projectpath):
        # 1. 只有两个文件
        print(oriinfiles, commands, outhead, projectpath)
        pdobjlist, matchstrlist = regex2pairs(oriinfiles, projectpath)
        outfilelist = [[i1[0] + i1[1][0] + i1[2], i1[0] + i1[1][1] + i1[2]] for i1 in matchstrlist]
        print(outfilelist)
        collist = pdobjlist[0][0].columns
        collabel = [i1 for i1 in collist if re.search("^label_", i1, re.M)]
        outjson = {i1: [] for i1 in collabel}
        outjson["model"] = []
        for command in commands:
            tkey = list(command.keys())[0]
            toutjson = self.funcmap[tkey](pdobjlist, command[tkey], outhead, projectpath)
            outjson["model"].append(tkey)
            [outjson[ik2].append(iv2) for ik2, iv2 in toutjson.items()]
        pdobjout = pd.DataFrame(outjson)
        pdobjout.set_index("model", inplace=True)
        return pdobjout


class PredictFunc(object):
    def __init__(self):
        self.funcmap = {
            "lgboost": self.lgboost,
            # "tcn": None,
            # "tabnet": None,
        }

    def lgboost(self, dataobj, modelhead, labelname, projectpath):
        outpdlist = []
        for i1 in labelname:
            # 模型加载
            modelpath = os.path.join(projectpath, "{}lgboost_{}.txt".format(modelhead, i1))
            try:
                model = lgb.Booster(model_file=modelpath)
                # 如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测
                pred_pd = model.predict(dataobj, num_iteration=model.best_iteration)
                outpdlist.append(pred_pd)
            except Exception as e:
                outpdlist.append(np.zeros(len(dataobj)))
                print(e)
        return outpdlist

    def __call__(self, oriinfiles, modelhead, commands, outhead, labelfile, projectpath):
        print(oriinfiles, commands, outhead, labelfile, projectpath)
        anylabel = glob.glob(os.path.join(projectpath, labelfile))[0]
        pdobj = pd.read_csv(os.path.join(projectpath, anylabel), header=0, encoding="utf8")
        pdobj.set_index("date", inplace=True)
        labelname = [i1 for i1 in pdobj.columns if re.search("^label_", i1, re.M)]
        labellenth = len(labelname)
        infiles = [glob.glob(os.path.join(projectpath, i2)) for i2 in oriinfiles]  # 正则列出
        infiles = list(set(itertools.chain(*infiles)))  # 展开去重
        for infile in infiles:
            # 为了便于集成学习，不同模型的同一类型存储到一个文件
            pdobj = pd.read_csv(infile, header=0, encoding="utf8")
            pdobj.set_index("date", inplace=True)
            pdobj = pdobj[[i3 for i3 in pdobj.columns if not re.search("^label_", i3, re.M)]]
            tpdlist = [[] for i2 in range(labellenth)]
            comkeys = [list(i2.keys())[0] for i2 in commands]
            for tkey in comkeys:
                outpdlist = self.funcmap[tkey](pdobj, modelhead, labelname, projectpath)
                [tpdlist[i3].append(outpdlist[i3]) for i3 in range(labellenth)]
            for id2, lbname in enumerate(labelname):
                tjson = {"{}_{}".format(lbname, tkey): tpdlist[id2][id3] for id3, tkey in enumerate(comkeys)}
                tmpoutpd = pd.DataFrame(tjson, index=pdobj.index)
                (filepath, tfilename) = os.path.split(infile)
                fname = os.path.join(filepath, "{}{}_{}".format(outhead, lbname, tfilename))
                tmpoutpd.to_csv(fname, index=True, header=True, encoding="utf-8")
        return None


train_func = {
    "训练拟合": TrainFunc(),
}

predict_func = {
    "数据预测": PredictFunc(),
}
