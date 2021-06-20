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


# from surf.surfing import logger1

# import kerastuner as kt


def pipe_train(dataobj, paras={}):
    outdata = dataobj
    return outdata


# class CVTuner(kt.engine.tuner.Tuner):
#     def run_trial(self, trial, X, y, splits, batch_size=32, epochs=1, callbacks=None):
#         val_losses = []
#         for train_indices, test_indices in splits:
#             X_train, X_test = [x[train_indices] for x in X], [x[test_indices] for x in X]
#             y_train, y_test = [a[train_indices] for a in y], [a[test_indices] for a in y]
#             if len(X_train) < 2:
#                 X_train = X_train[0]
#                 X_test = X_test[0]
#             if len(y_train) < 2:
#                 y_train = y_train[0]
#                 y_test = y_test[0]
#
#             model = self.hypermodel.build(trial.hyperparameters)
#             hist = model.fit(X_train, y_train,
#                              validation_data=(X_test, y_test),
#                              epochs=epochs,
#                              batch_size=batch_size,
#                              callbacks=callbacks)
#
#             val_losses.append([hist.history[k][-1] for k in hist.history])
#         val_losses = np.asarray(val_losses)
#         self.oracle.update_trial(trial.trial_id,
#                                  {k: np.mean(val_losses[:, i]) for i, k in enumerate(hist.history.keys())})
#         self.save_model(trial.trial_id, model)

class TrainFunc(object):
    def __init__(self):
        self.funcmap = {
            "lgboost": self.lgboost,
            # "tcn": None,
            # "tabnet": None,
        }

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

    def lgboost(self, dataobjs, params, outhead, projectpath):
        # todo: 预测，模型自定义，策略; 回测
        test_X = []
        for ttest in dataobjs:
            ttest.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
            test_X.append(ttest.values)
        test_X = np.concatenate(test_X, axis=0)
        # 模型加载
        # todo: 统配得出 collabel
        for id1, i1 in enumerate(collabel):
            print("training:", i1)
            fulpath = os.path.join(projectpath, "{}lgboost_{}.txt".format(outhead, i1))
        model = lgb.Booster(model_file=fulpath)
        # 如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测
        pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
        return pred_test_y

    def __call__(self, oriinfiles, commands, outhead, projectpath):
        # 1. 只有两个文件
        print(oriinfiles, commands, outhead, projectpath)
        pdobjlist, matchstrlist = regex2pairs(oriinfiles, projectpath)
        outfilelist = [[i1[0] + i1[1][0] + i1[2], i1[0] + i1[1][1] + i1[2]] for i1 in matchstrlist]
        print(outfilelist)
        outjson = {}
        for command in commands:
            tkey = list(command.keys())[0]
            outjson[command] = self.funcmap[tkey](pdobjlist, command[tkey], outhead, projectpath)
        pdobjout = pd.DataFrame(outjson)
        print(pdobjout)
        return pdobjout


train_func = {
    "训练拟合": TrainFunc(),
}

predict_func = {
    "数据预测": PredictFunc(),
}
