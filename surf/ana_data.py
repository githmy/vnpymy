from surf.script_tab import keytab
import os, json, time, re, codecs
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging.handlers
import pandas as pd
import itertools


def pipe_mean(dataobj, paras={}):
    outdata = dataobj
    return outdata

ana_func = {
    "均值": pipe_mean,
    "平稳性": None,
    "相关性": None,
    "协整性": None,
}
