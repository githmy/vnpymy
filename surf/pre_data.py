from surf.script_tab import keytab
from surf.surf_tool import regex2pairs
import os, json, time, re, codecs, glob, shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging.handlers
import pandas as pd
import itertools
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args


# 拆分时间序列的类
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]

                train_array = np.sort(np.unique(
                    np.concatenate((train_array,
                                    train_array_tmp)),
                    axis=None), axis=None)

            train_end = train_array.size

            for test_group_idx in unique_groups[group_test_start:
                    group_test_start +
                    group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                    np.concatenate((test_array,
                                    test_array_tmp)),
                    axis=None), axis=None)

            test_array = test_array[group_gap:]

            if self.verbose > 0:
                pass

            yield [int(i) for i in train_array], [int(i) for i in test_array]


# 拆分时间序列的类
class PurgedGroupTimeSeriesSplitStacking(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    stacking_mode : bool, default=True
        Whether to provide an additional set to test a stacking classifier or not. 
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    max_val_group_size : int, default=Inf
        Maximum group size for a single validation set.
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split, if stacking_mode = True and None 
        it defaults to max_val_group_size.
    val_group_gap : int, default=None
        Gap between train and validation
    test_group_gap : int, default=None
        Gap between validation and test, if stacking_mode = True and None 
        it defaults to val_group_gap.
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 stacking_mode=True,
                 max_train_group_size=np.inf,
                 max_val_group_size=np.inf,
                 max_test_group_size=np.inf,
                 val_group_gap=None,
                 test_group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.max_val_group_size = max_val_group_size
        self.max_test_group_size = max_test_group_size
        self.val_group_gap = val_group_gap
        self.test_group_gap = test_group_gap
        self.verbose = verbose
        self.stacking_mode = stacking_mode

    def split(self, X, y=None, groups=None):
        if self.stacking_mode:
            return self.split_ensemble(X, y, groups)
        else:
            return self.split_standard(X, y, groups)

    def split_standard(self, X, y=None, groups=None):
        """Generate indices to split data into training and validation set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/validation set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        val : ndarray
            The validation set indices for that split.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_splits = self.n_splits
        group_gap = self.val_group_gap
        max_val_group_size = self.max_val_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds, n_groups))

        group_val_size = min(n_groups // n_folds, max_val_group_size)
        group_val_starts = range(n_groups - n_splits * group_val_size, n_groups, group_val_size)
        for group_val_start in group_val_starts:
            train_array = []
            val_array = []

            group_st = max(0, group_val_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_val_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(np.unique(np.concatenate((train_array, train_array_tmp)), axis=None), axis=None)

            train_end = train_array.size

            for val_group_idx in unique_groups[group_val_start: group_val_start + group_val_size]:
                val_array_tmp = group_dict[val_group_idx]
                val_array = np.sort(np.unique(np.concatenate((val_array, val_array_tmp)), axis=None), axis=None)

            val_array = val_array[group_gap:]

            if self.verbose > 0:
                pass

            yield [int(i) for i in train_array], [int(i) for i in val_array]

    def split_ensemble(self, X, y=None, groups=None):
        """Generate indices to split data into training, validation and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        val : ndarray
            The validation set indices for that split (testing indices for base classifiers).
        test : ndarray
            The testing set indices for that split (testing indices for final classifier)
        """

        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_splits = self.n_splits
        val_group_gap = self.val_group_gap
        test_group_gap = self.test_group_gap
        if test_group_gap is None:
            test_group_gap = val_group_gap
        max_train_group_size = self.max_train_group_size
        max_val_group_size = self.max_val_group_size
        max_test_group_size = self.max_test_group_size
        if max_test_group_size is None:
            max_test_group_size = max_val_group_size

        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)

        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(("Cannot have number of folds={0} greater than"
                              " the number of groups={1}").format(n_folds, n_groups))

        group_val_size = min(n_groups // n_folds, max_val_group_size)
        group_test_size = min(n_groups // n_folds, max_test_group_size)

        group_test_starts = range(n_groups - n_splits * group_test_size, n_groups, group_test_size)
        train_indices = []
        val_indices = []
        test_indices = []

        for group_test_start in group_test_starts:
            train_array = []
            val_array = []
            test_array = []
            val_group_st = max(max_train_group_size + val_group_gap,
                               group_test_start - test_group_gap - max_val_group_size)
            train_group_st = max(0, val_group_st - val_group_gap - max_train_group_size)

            for train_group_idx in unique_groups[train_group_st:(val_group_st - val_group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(np.unique(np.concatenate((train_array, train_array_tmp)), axis=None), axis=None)

            train_end = train_array.size
            for val_group_idx in unique_groups[val_group_st:(group_test_start - test_group_gap)]:
                val_array_tmp = group_dict[val_group_idx]
                val_array = np.sort(np.unique(np.concatenate((val_array, val_array_tmp)), axis=None), axis=None)

            val_array = val_array[val_group_gap:]
            for test_group_idx in unique_groups[group_test_start:(group_test_start + group_test_size)]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(np.concatenate((test_array, test_array_tmp)), axis=None), axis=None)

            test_array = test_array[test_group_gap:]
            yield [int(i) for i in train_array], [int(i) for i in val_array], [int(i) for i in test_array]


def sharp_ratio(data, base_ratio=0.0):
    num = len(data)
    t_return = (data.shift(-1) - data) / data
    std = t_return.std()
    sharpratio = (t_return.mean() - base_ratio) * (np.sqrt(num)) / std
    return sharpratio


class Pre_data(object):
    def __init__(self):
        self.funcmap = {
            "种子": self.set_all_seeds,
            "填充": self.pipe_pad,
            # 一个数组栏，一个dataframe
            "取列": self.split_columns,
            "取行": self.split_rows,
        }

    def set_all_seeds(self, dataobj, seed):
        np.random.seed(seed)
        random.seed(seed)
        # tf.random.set_seed(seed)
        return dataobj

    def pipe_pad(self, dataobj, paras={}):
        if paras["值"] is None:
            if paras["方式"] == "向前":
                # 再向上填充
                dataobj.fillna(method='bfill', inplace=True)
            elif paras["方式"] == "向后":
                # 先向下填充
                dataobj.fillna(method='ffill', inplace=True)
            else:
                raise Exception("paras error {}".format(paras))
        else:
            dataobj.fillna(value=paras["值"], inplace=True)
        return dataobj

    def split_columns(self, dataobj, paras):
        return dataobj[paras]

    def split_rows(self, dataobj, paras):
        if isinstance(paras[0], str):
            outdata = dataobj.loc[paras[0]:]
        elif isinstance(paras[0], int):
            outdata = dataobj.iloc[paras[0]:]
        else:
            raise Exception("type error {}".format(paras))
        if isinstance(paras[1], str):
            outdata = outdata.loc[:paras[1]]
        elif isinstance(paras[1], int):
            outdata = outdata.iloc[:paras[1]]
        else:
            raise Exception("type error {}".format(paras))
        return outdata

    def __call__(self, infiles, commands):
        outdata = []
        for infile in infiles:
            pdobj = pd.read_csv(infile, header=0, encoding="utf8")
            pdobj.set_index("date", inplace=True)
            # 顺序处理
            for command in commands:
                tkey = list(command.keys())[0]
                pdobj = self.funcmap[tkey](pdobj, command[tkey])
            outdata.append(pdobj)
        return outdata


class Train_split(object):
    def __init__(self):
        self.funcmap = {
            # 一个数组栏，一个dataframe
            "拆分": self.split_train_test,
        }

    def split_train_test(self, dataobj, paras):
        outlist = []
        if isinstance(paras[0], str):
            outlist.append(dataobj.loc[:paras[0]])
            if len(paras) > 1:
                outlist.append(dataobj.loc[paras[0]:paras[1]])
                outlist.append(dataobj.loc[paras[1]:])
            else:
                outlist.append(dataobj.loc[paras[0]:])
        elif isinstance(paras[0], int):
            outlist.append(dataobj.iloc[:paras[0]])
            if len(paras) > 1:
                outlist.append(dataobj.iloc[paras[0]:paras[1]])
                outlist.append(dataobj.iloc[paras[1]:])
            else:
                outlist.append(dataobj.iloc[paras[0]:])
        elif isinstance(paras[0], float):
            tsplit = len(dataobj)
            tsplit1 = int(tsplit * paras[0])
            outlist.append(dataobj.iloc[:tsplit1])
            if len(paras) > 1:
                tsplit2 = int(tsplit * sum(paras))
                outlist.append(dataobj.iloc[tsplit1:tsplit2])
                outlist.append(dataobj.iloc[tsplit2:])
            else:
                outlist.append(dataobj.iloc[tsplit1:])
        else:
            raise Exception("type error {}".format(paras))
        return outlist

    def __call__(self, infiles, commands):
        outdata = []
        for infile in infiles:
            pdobj = pd.read_csv(infile, header=0, encoding="utf8")
            pdobj.set_index("date", inplace=True)
            # 顺序处理
            for command in commands:
                tkey = list(command.keys())[0]
                pdobj = self.funcmap[tkey](pdobj, command[tkey])
            outdata.append(pdobj)
        return outdata


class SequenceChara(object):
    def __init__(self):
        self.funcmap = {
            "均值n": self.mean_n,
            "标准差n": self.std_n,
            "涨幅比n": self.ratio_n,
            "回撤n": self.draw_n,
            "最涨n": self.maxrise_n,
            "夏普n": self.sharp_n,
            "label_最大n": self.l_max_n,
            "label_最小n": self.l_min_n,
            "label_回撤n": self.l_draw_n,
            "label_最涨n": self.l_maxrise_n,
        }

    def mean_n(self, dataobj, n):
        outdata = dataobj.iloc[:, 0].rolling(window=n, center=False).mean()
        return outdata

    def std_n(self, dataobj, n):
        outdata = dataobj.iloc[:, 0].rolling(window=n, center=False).std()
        return outdata

    def ratio_n(self, dataobj, n):
        outdata = dataobj.iloc[:, 0].rolling(window=n, center=False).apply(lambda x: x[-1] / x[0])
        return outdata

    def draw_n(self, dataobj, n):
        pricepd = dataobj.iloc[:, 0]
        maxfallret = pd.Series(index=pricepd.index)
        for i in range(0, len(dataobj) - n):
            tmpsec = pricepd[i + 1:i + n + 1]
            tmpmax = pricepd[i]
            tmpmin = pricepd[i]
            tmpdrawdown = [1.0]
            for t in range(0, n):
                if tmpsec[t] > tmpmax:
                    tmpmax = tmpsec[t]
                    tmpdrawdown.append(tmpdrawdown[-1])
                elif tmpsec[t] <= tmpmin:
                    tmpmin = tmpsec[t]
                    tmpdrawdown.append(tmpmin / tmpmax)
                else:
                    pass
            maxfallret[i] = min(tmpdrawdown)
        return maxfallret

    def maxrise_n(self, dataobj, n):
        pricepd = dataobj.iloc[:, 0]
        maxraiseret = pd.Series(index=pricepd.index)
        for i in range(0, len(dataobj) - n):
            tmpsec = pricepd[i + 1:i + n + 1]
            tmpmax = pricepd[i]
            tmpmin = pricepd[i]
            tmpdrawup = [1.0]
            for t in range(0, n):
                if tmpsec[t] > tmpmax:
                    tmpmax = tmpsec[t]
                    tmpdrawup.append(tmpmax / tmpmin)
                elif tmpsec[t] <= tmpmin:
                    tmpmin = tmpsec[t]
                    tmpdrawup.append(tmpdrawup[-1])
                else:
                    pass
            maxraiseret[i] = max(tmpdrawup)
        return maxraiseret

    def sharp_n(self, dataobj, n):
        outdata = dataobj.iloc[:, 0].rolling(window=n, center=False).apply(sharp_ratio)
        return outdata

    def l_max_n(self, dataobj, n):
        outdata = dataobj.iloc[:, 0].rolling(window=n, center=False).max()
        outdata = outdata.shift(-n)
        return outdata

    def l_min_n(self, dataobj, n):
        outdata = dataobj.iloc[:, 0].rolling(window=n, center=False).min()
        outdata.shift(-n)
        return outdata

    def l_draw_n(self, dataobj, n):
        outdata = self.draw_n(dataobj, n)
        outdata.shift(-n)
        return outdata

    def l_maxrise_n(self, dataobj, n):
        outdata = self.maxrise_n(dataobj, n)
        outdata.shift(-n)
        return outdata

    def __call__(self, infiles, commands):
        outdata = []
        colhead = []
        for infile in infiles:
            pdobj = pd.read_csv(infile, header=0, encoding="utf8")
            pdobj.set_index("date", inplace=True)
            delhead = pdobj.columns[0]
            colhead.append(delhead)
            # 并行处理
            toutd = []
            for command in commands:
                tkey = list(command.keys())[0]
                outobj = self.funcmap[tkey](pdobj, command[tkey])
                toutd.append(outobj)
            outdata.append(toutd)
        return outdata, colhead


class CharaExtract(object):
    def __init__(self):
        self.funcmap = {
            "profit_avelog": self.profit_avelog,
            "胜率": self.win_ratio,
            "回撤": self.draw_n,
            "最涨": self.rise_n,
            "夏普": self.sharp_n,
        }

    def profit_avelog(self, dataobj):
        return np.log(dataobj.iloc[-1, 0] / dataobj.iloc[0, 0]) / len(dataobj)

    def win_ratio(self, dataobj):
        pricepd = dataobj.diff()
        pricepd = np.array(pricepd.iloc[:, 0])
        posinum = len(pricepd[pricepd > 0])
        allnum = len(pricepd[~np.isnan(pricepd)])
        return float(posinum) / allnum

    def draw_n(self, dataobj):
        pricepd = dataobj.iloc[:, 0]
        n = len(dataobj)
        tmpsec = pricepd[0:n]
        tmpmax = pricepd[0]
        tmpmin = pricepd[0]
        tmpdrawdown = [1.0]
        for i in range(1, n):
            if tmpsec[i] > tmpmax:
                tmpmax = tmpsec[i]
                tmpdrawdown.append(tmpdrawdown[-1])
            elif tmpsec[i] <= tmpmin:
                tmpmin = tmpsec[i]
                tmpdrawdown.append(tmpmin / tmpmax)
            else:
                pass
        return min(tmpdrawdown)

    def rise_n(self, dataobj):
        pricepd = dataobj.iloc[:, 0]
        n = len(dataobj)
        tmpsec = pricepd[0:n]
        tmpmax = pricepd[0]
        tmpmin = pricepd[0]
        tmpdrawup = [1.0]
        for i in range(1, n):
            if tmpsec[i] > tmpmax:
                tmpmax = tmpsec[i]
                tmpdrawup.append(tmpmax / tmpmin)
            elif tmpsec[i] <= tmpmin:
                tmpmin = tmpsec[i]
                tmpdrawup.append(tmpdrawup[-1])
            else:
                pass
        return max(tmpdrawup)

    def sharp_n(self, dataobj):
        tsr = sharp_ratio(dataobj)
        return tsr[0]

    def __call__(self, infiles, commands):
        outdatas = [{"filename": [], i1: []} for i1 in commands]
        for i1, command in enumerate(commands):
            # 并行处理
            for infile in infiles:
                pdobj = pd.read_csv(infile, header=0, encoding="utf8")
                pdobj.set_index("date", inplace=True)
                pdobj = pdobj[[pdobj.columns[0]]]
                outobj = self.funcmap[command](pdobj)
                ttinfile = os.path.split(infile)[1]
                outdatas[i1]["filename"].append(ttinfile)
                outdatas[i1][command].append(outobj)
        outdatapds = []
        for i1 in outdatas:
            tpd = pd.DataFrame(i1)
            tpd.set_index("filename", inplace=True)
            outdatapds.append(tpd)
        return outdatapds


class DataMerge(object):
    def __init__(self):
        pass

    def __call__(self, oriinfiles, projectpath):
        # 1. 只支持 前后统配合并，去掉前后的 *
        pdobjlist, matchstrlist = regex2pairs(oriinfiles, projectpath)
        outfilelist = [i1[0] + "_".join(["origin" if i2 == "" else i2 for i2 in i1[1]]) + i1[2] for i1 in matchstrlist]
        outpdobjlist = [pd.concat(i1, axis=1) for i1 in pdobjlist]
        return outpdobjlist, outfilelist


class DataCopy(object):
    def __init__(self):
        pass

    def __call__(self, oriinfiles, prefix, projectpath):
        infiles = [glob.glob(os.path.join(projectpath, i2)) for i2 in oriinfiles]
        infiles = set(itertools.chain(*infiles))  # 展开去重
        for infile in infiles:
            (filepath, ofile) = os.path.split(infile)
            shutil.copy(infile, os.path.join(filepath, prefix + ofile))
        return None


class DataCalc(object):
    def __init__(self):
        self.funcmap = {
            "+": self.add,
            "-": self.mins,
            "*": self.multi,
            "/": self.divide,
            "**": self.ppower,
        }
        self.symbolmap = {
            "+": "加",
            "-": "减",
            "*": "乘",
            "/": "除",
            "**": "幂",
        }

    def add(self, dataobj, commandstr, float_f=None, float_b=None):
        if float_b is None and float_f is None:
            outdata = dataobj[0].iloc[:, 0] + dataobj[1].iloc[:, 0]
            outdata = pd.DataFrame(outdata)
            outdata.rename(columns={0: "_".join([dataobj[0].columns[0], commandstr, dataobj[1].columns[0]])},
                           inplace=True)
        elif float_b is not None:
            outdata = dataobj[0].iloc[:, 0] + float_b
            outdata = pd.DataFrame(outdata)
            outdata.rename(columns={0: "_".join([dataobj[0].columns[0], commandstr, str(float_b)])}, inplace=True)
        elif float_f is not None:
            outdata = float_f + dataobj[1].iloc[:, 0]
            outdata = pd.DataFrame(outdata)
            outdata.rename(columns={0: "_".join([str(float_f), commandstr, dataobj[0].columns[0]])}, inplace=True)
        else:
            raise Exception("No such situation.float_f float_b both not None")
        return outdata

    def mins(self, dataobj, commandstr, float_f=None, float_b=None):
        colstrs = [dataobj[0].columns[0], dataobj[1].columns[0]]
        if float_b is None and float_f is None:
            outdata = dataobj[0].iloc[:, 0] - dataobj[1].iloc[:, 0]
            outdata = pd.DataFrame(outdata)
            outdata.rename(columns={0: "_".join([colstrs[0], commandstr, colstrs[1]])}, inplace=True)
        elif float_b is not None:
            outdata = dataobj[0].iloc[:, 0] - float_b
            outdata = pd.DataFrame(outdata)
            outdata.rename(columns={0: "_".join([dataobj[0].columns[0], commandstr, str(float_b)])}, inplace=True)
        elif float_f is not None:
            outdata = float_f - dataobj[1].iloc[:, 0]
            outdata = pd.DataFrame(outdata)
            outdata.rename(columns={0: "_".join([str(float_f), commandstr, dataobj[0].columns[0]])}, inplace=True)
        else:
            raise Exception("No such situation.float_f float_b both not None")
        return outdata

    def multi(self, dataobj, commandstr, float_f=None, float_b=None):
        if float_b is None and float_f is None:
            outdata = dataobj[0].iloc[:, 0] * dataobj[1].iloc[:, 0]
            outdata = pd.DataFrame(outdata)
            outdata.rename(columns={0: "_".join([dataobj[0].columns[0], commandstr, dataobj[1].columns[0]])},
                           inplace=True)
        elif float_b is not None:
            outdata = dataobj[0].iloc[:, 0] * float_b
            outdata = pd.DataFrame(outdata)
            outdata.rename(columns={0: "_".join([dataobj[0].columns[0], commandstr, str(float_b)])}, inplace=True)
        elif float_f is not None:
            outdata = float_f * dataobj[1].iloc[:, 0]
            outdata = pd.DataFrame(outdata)
            outdata.rename(columns={0: "_".join([str(float_f), commandstr, dataobj[0].columns[0]])}, inplace=True)
        else:
            raise Exception("No such situation.float_f float_b both not None")
        return outdata

    def divide(self, dataobj, commandstr, float_f=None, float_b=None):
        if float_b is None and float_f is None:
            outdata = dataobj[1].iloc[:, 0] / dataobj[1].iloc[:, 0]
            outdata = pd.DataFrame(outdata)
            outdata.rename(columns={0: "_".join([dataobj[0].columns[0], commandstr, dataobj[1].columns[0]])},
                           inplace=True)
        elif float_b is not None:
            outdata = dataobj[0].iloc[:, 0] / float_b
            outdata = pd.DataFrame(outdata)
            outdata.rename(columns={0: "_".join([dataobj[0].columns[0], commandstr, str(float_b)])}, inplace=True)
        elif float_f is not None:
            outdata = float_f / dataobj[1].iloc[:, 0]
            outdata = pd.DataFrame(outdata)
            outdata.rename(columns={0: "_".join([str(float_f), commandstr, dataobj[0].columns[0]])}, inplace=True)
        else:
            raise Exception("No such situation.float_f float_b both not None")
        return outdata

    def ppower(self, dataobj, commandstr, float_f=None, float_b=None):
        if float_b is None and float_f is None:
            outdata = dataobj[0].iloc[:, 0] ** dataobj[1].iloc[:, 0]
            outdata = pd.DataFrame(outdata)
            outdata.rename(columns={0: "_".join([dataobj[0].columns[0], commandstr, dataobj[1].columns[0]])},
                           inplace=True)
        elif float_b is not None:
            outdata = dataobj[0].iloc[:, 0] ** float_b
            outdata = pd.DataFrame(outdata)
            outdata.rename(columns={0: "_".join([dataobj[0].columns[0], commandstr, str(float_b)])}, inplace=True)
        elif float_f is not None:
            outdata = float_f ** dataobj[1].iloc[:, 0]
            outdata = pd.DataFrame(outdata)
            outdata.rename(columns={0: "_".join([str(float_f), commandstr, dataobj[0].columns[0]])}, inplace=True)
        else:
            raise Exception("No such situation.float_f float_b both not None")
        return outdata

    def __call__(self, oriinfiles, commands, projectpath):
        # 1. 只有两个文件
        outfilelist = []
        pdobjoutlist = []
        if isinstance(oriinfiles[0], str) and isinstance(oriinfiles[1], str):
            pdobjlist, matchstrlist = regex2pairs(oriinfiles, projectpath)
            for command in commands:
                for onepd, inlist in zip(pdobjlist, matchstrlist):
                    outobj = self.funcmap[command](onepd, self.symbolmap[command], float_f=None, float_b=None)
                    pdobjoutlist.append(outobj)
                    outfilelist.append(
                        inlist[0] + "_".join([inlist[1][0], self.symbolmap[command], inlist[1][1]]) + inlist[2])
        elif isinstance(oriinfiles[0], str) and not isinstance(oriinfiles[1], str):
            infiles = glob.glob(os.path.join(projectpath, oriinfiles[0]))
            strip_infile = oriinfiles[0].strip("*")
            for command in commands:
                for infile in infiles:
                    tfstrs = infile.split(strip_infile)
                    newoustr = strip_infile.join(tfstrs[0:-1]) + strip_infile + "_" + self.symbolmap[command] + "_" \
                               + str(oriinfiles[1]) + tfstrs[-1]
                    pdobj = pd.read_csv(infile, header=0, index_col=0, encoding="utf8")
                    outobj = self.funcmap[command]([pdobj], self.symbolmap[command], float_f=None,
                                                   float_b=oriinfiles[1])
                    pdobjoutlist.append(outobj)
                    outfilelist.append(newoustr)
        elif not isinstance(oriinfiles[0], str) and isinstance(oriinfiles[1], str):
            infiles = glob.glob(os.path.join(projectpath, oriinfiles[1]))
            strip_infile = oriinfiles[1].strip("*")
            for command in commands:
                for infile in infiles:
                    tfstrs = infile.split(strip_infile)
                    newoustr = strip_infile.join(tfstrs[0:-1]) + str(oriinfiles[0]) + "_" + self.symbolmap[command] \
                               + "_" + strip_infile + tfstrs[-1]
                    pdobj = pd.read_csv(infile, header=0, index_col=0, encoding="utf8")
                    outobj = self.funcmap[command]([pdobj], self.symbolmap[command], float_f=oriinfiles[0],
                                                   float_b=None)
                    pdobjoutlist.append(outobj)
                    outfilelist.append(newoustr)
        else:
            # 数之间运算无必要，忽略。
            pass
        return pdobjoutlist, outfilelist


pre_func = {
    # 返回的是列
    "数据处理": Pre_data(),
    # 返回的是列
    "训练拆分": Train_split(),
    # 返回的是列
    "序列特征": SequenceChara(),
    # 返回的是值，汇总成一个
    "数据提取": CharaExtract(),
    # 多个dataframe ,根据统配名 合并
    "数据合并": DataMerge(),
    "数据复制": DataCopy(),
    # 多个dataframe ,根据统配名 合并
    "数据运算": DataCalc(),
}
