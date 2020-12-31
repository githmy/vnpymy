from surf.script_tab import keytab
import os, json, time, re, codecs
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging.handlers
import pandas as pd
import itertools


def pipe_show(dataobj, paras={}):
    outdata = dataobj
    return outdata

show_func = {
    "显示": pipe_show,
}
