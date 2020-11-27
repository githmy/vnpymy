from django.shortcuts import render
from django.views.generic import FormView, UpdateView
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from common.mixins import JSONResponseMixin, AdminUserRequiredMixin
from common.utils import get_object_or_none
from .. import forms
from django import forms as oforms
from django.views import View
import codecs
import xlrd
import xlwt
import uuid
import datetime
import pandas as pd
import os
from io import StringIO, BytesIO

# Create your views here.

global gdatas
# 文件名：pandas
gdatas = {}
# 数据间的关系
global rdatas
rdatas = {}
# 数据的置信度
global cdatas
cdatas = {}
# 数据的拟合
global fdatas
fdatas = {}


def index(request):
    return render(request, 'data_analy/index.html')


# @login_required
def data_index(request):
    "数据 原始输出 表头"
    context = {}
    context["collist"] = []
    if len(gdatas.keys()) > 0:
        context["collist"] = gdatas[list(gdatas.keys())[0]].columns
    return render(request, 'data_analy/data_index.html', context)


def data_list(request):
    "数据 原始输出 内容"
    if len(gdatas.keys()) > 0:
        tpd = gdatas[list(gdatas.keys())[0]]
        # print(tpd.values)
        return JsonResponse({
            'total': tpd.shape[0],
            'data': query2dict(tpd),
            '_': request.GET.get('_', 0)
        })
    else:
        return JsonResponse({})


def nindex_v(request):
    "指标 汇总输出 表头"
    context = {}
    # 1. 列名 2. 平稳性 3.
    context["collist"] = ["names", "mean", "std", "std"]
    return render(request, 'data_analy/nindex_index.html', context)


def data_nindex(request):
    "指标 汇总输出 内容"
    if len(rdatas.keys()) > 0:
        tpd = rdatas[list(rdatas.keys())[0]]
        return JsonResponse({
            'total': tpd.shape[0],
            'data': query2dict(tpd),
            '_': request.GET.get('_', 0)
        })
    else:
        return JsonResponse({})


def relation_v(request):
    "相关性 汇总输出 表头"
    context = {}
    # 1. 关系对名字 2.
    context["collist"] = ["names", "a c", "b c"]
    return render(request, 'data_analy/relation_index.html', context)


def data_relation(request):
    "相关性 汇总输出 内容"
    if len(rdatas.keys()) > 0:
        tpd = rdatas[list(rdatas.keys())[0]]
        return JsonResponse({
            'total': tpd.shape[0],
            'data': query2dict(tpd),
            '_': request.GET.get('_', 0)
        })
    else:
        return JsonResponse({})


def confidence_v(request):
    "置信度 汇总输出 表头"
    context = {}
    context["collist"] = []
    if len(cdatas.keys()) > 0:
        context["collist"] = ["a b", "a c", "b c"]
    return render(request, 'data_analy/confidence_index.html', context)


def data_confidence(request):
    "置信度 汇总输出 内容"
    if len(cdatas.keys()) > 0:
        tpd = cdatas[list(cdatas.keys())[0]]
        return JsonResponse({
            'total': tpd.shape[0],
            'data': query2dict(tpd),
            '_': request.GET.get('_', 0)
        })
    else:
        return JsonResponse({})


def fitfunc_v(request):
    "拟合 汇总输出 表头"
    context = {}
    context["collist"] = []
    # fdatas = {list(gdatas.keys())[0]:}
    # tpd = rdatas[list(rdatas.keys())[0]]
    if len(fdatas.keys()) > 0:
        context["collist"] = ["namepair", "train_test_num", "best_degree_score", "all_degree_score"]
    return render(request, 'data_analy/fit_index.html', context)


def data_fit(request):
    "拟合 汇总输出 内容"
    from common.tools.data_fit import regression_check
    if len(gdatas.keys()) > 0:
        tpd = fdatas[list(gdatas.keys())[0]]
        return JsonResponse({
            'total': tpd.shape[0],
            'data': query2dict(tpd),
            '_': request.GET.get('_', 0)
        })
    else:
        return JsonResponse({})


def data_fit_bak(request):
    "拟合 汇总输出 内容"
    if len(fdatas.keys()) > 0:
        tpd = fdatas[list(fdatas.keys())[0]]
        return JsonResponse({
            'total': tpd.shape[0],
            'data': query2dict(tpd),
            '_': request.GET.get('_', 0)
        })
    else:
        return JsonResponse({})


def data_clean(request):
    # 文件名：pandas
    global gdatas
    gdatas = {}
    # 数据间的关系
    global rdatas
    rdatas = {}
    # 数据的置信度
    global cdatas
    cdatas = {}
    # 数据的拟合
    global fdatas
    fdatas = {}
    return JsonResponse({})


def query2dict(t_pandas):
    lists = []
    for id1, values in t_pandas.iterrows():
        lists.append({"pk": id1, **values})
    return lists


class DataExportView(View):
    def get(self, request, *args, **kwargs):
        filename = os.path.join(".", "template.xlsx")
        response = HttpResponse(content_type='application/vnd.ms-excel')
        response['Content-Disposition'] = 'attachment; filename="%s"' % filename
        with codecs.open(filename, "rb") as f:
            c = f.read()
        response.write(c)
        return response

    def post(self, request, *args, **kwargs):
        return JsonResponse({'redirect': ""})


class BulkImportDataView(JSONResponseMixin, FormView):
    form_class = forms.FileForm

    def form_valid(self, form):
        file = form.cleaned_data['file']
        wb = xlrd.open_workbook(filename=None, file_contents=file.read())
        sheet1 = wb.sheet_by_index(0)  # 第一个
        header_ = sheet1.row_values(0)  # 表格头
        tsig = [1 if isinstance(field, str) else  0 for field in sheet1.row_values(1)]
        pdjson = {col: [] for col in header_}
        try:
            for i1 in range(1, sheet1.nrows):  # 遍历每行表格
                row = sheet1.row_values(i1)
                for id2, i2 in enumerate(row):
                    if tsig[id2]:  # 处理时间格式问题
                        i2 = datetime.datetime.strptime("".join(i2.split('/')), "%Y%m%d%H%M%S%f")
                        i2 = i2.strftime('%Y-%m-%d %H:%M:%S %f')
                    pdjson[header_[id2]].append(i2)
            data = {
                'created': "ok",
            }
            gdatas[file.__str__()] = pd.DataFrame(pdjson)
        except Exception as e:
            data = {
                'created': "error",
            }
        return self.render_json_response(data)
