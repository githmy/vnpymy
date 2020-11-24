from django.shortcuts import render
from django.views.generic import FormView, UpdateView
from common.mixins import JSONResponseMixin, AdminUserRequiredMixin
from common.utils import get_object_or_none
from .. import forms
import xlrd
import xlwt
import uuid
import datetime


# Create your views here.

def index(request):
    return render(request, 'data_analy/index.html')


def relation_v(request):
    context = {}
    context['hello'] = 'Hello World!'
    return render(request, 'data_analy/data_analy_index.html', context)


def confidence_v(request):
    context = {}
    context['hello'] = 'Hello World!'
    return render(request, 'data_analy/data_analy_index.html', context)


def fitfunc_v(request):
    context = {}
    context['hello'] = 'Hello World!'
    return render(request, 'data_analy/data_analy_index.html', context)


# @login_required
def data_index(request):
    print(777)
    Model = None
    sform = forms.DataSearchForm(Model)
    Model = {}
    cform = forms.DataAddForm(Model)
    return render(request, 'data_analy/index.html', {
        'searchForm': sform,
        'createForm': cform
    })


class BulkImportDataView(AdminUserRequiredMixin, JSONResponseMixin, FormView):
    form_class = forms.FileForm

    def form_valid(self, form):
        file = form.cleaned_data['file']
        wb = xlrd.open_workbook(filename=None, file_contents=file.read())
        sheet1 = wb.sheet_by_index(0)  # 第一个
        header_ = sheet1.row_values(0)  # 表格头
        # fields = [field for field in Purchase._meta.fields]
        print(999)
        print(sheet1.row_values())
        mapping_reverse = {field.verbose_name: field.name for field in fields}
        attr = [mapping_reverse.get(n, None) for n in header_]
        if None in attr:
            data = {'valid': False, 'msg': 'Must be same format as template or export file'}
            return self.render_json_response(data)

        created, updated, failed, purchase = [], [], [], []
        lines = 1
        for i in range(1, sheet1.nrows):  # 遍历每行表格
            row = sheet1.row_values(lines)
            if set(row) == {''}:
                continue
            purchase_dict = dict(zip(attr, row))
            id_ = purchase_dict.pop('id', 0)

            if id_ == '':
                try:
                    purchaseb = Purchase()
                    for k, v in purchase_dict.items():
                        if v:
                            if k == 'pr_submitdate' or k == 'receive':
                                if isinstance(v, str):
                                    v = datetime.date(int(v.split('/')[0]), int(v.split('/')[1]),
                                                      int(v.split('/')[2]))  # 处理时间格式问题
                                else:
                                    v = xlrd.xldate.xldate_as_datetime(v, 0)  # 处理时间格式问题
                                setattr(purchaseb, k, v)
                            elif k == "num":
                                setattr(purchaseb, k, str(v))
                            else:
                                setattr(purchaseb, k, v)
                    purchase.append(purchaseb)
                    created.append(purchase_dict['num'])
                except IndexError as e:
                    failed.append('%s: %s' % (purchase_dict['num'], str(e)))
            else:
                purchaseb = get_object_or_none(Purchase, id=id_)
                for k, v in purchase_dict.items():
                    if v:
                        if k == 'pr_submitdate' or k == 'receive':
                            if isinstance(v, str):
                                v = datetime.date(int(v.split('/')[0]), int(v.split('/')[1]),
                                                  int(v.split('/')[2]))  # 处理时间格式问题
                            else:
                                v = xlrd.xldate.xldate_as_datetime(v, 0)  # 处理时间格式问题
                            setattr(purchaseb, k, v)
                        elif k == "num":
                            setattr(purchaseb, k, str(v))
                        else:
                            setattr(purchaseb, k, v)
                try:
                    purchaseb.save()
                    updated.append(purchase_dict['num'])
                except Exception as e:
                    failed.append('%s: %s' % (purchase_dict['num'], str(e)))
            lines += 1
        if purchase:
            try:
                Purchase.objects.bulk_create(purchase)
            except IntegrityError as e:
                # map(lambda x: failed.append(x.num), assets)
                for i in purchase:
                    failed.append(i.num)
                created = []
        data = {
            'created': created,
            'created_info': '创建 {}'.format(len(created)),
            'updated': updated,
            'updated_info': '更新 {}'.format(len(updated)),
            'failed': failed,
            'failed_info': '失败 {}'.format(len(failed)),
            'valid': True,
            'msg': '创建: {}. 更新: {}, 失败: {}'.format(len(created), len(updated), len(failed))
        }
        return self.render_json_response(data)
