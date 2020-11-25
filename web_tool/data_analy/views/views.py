from django.shortcuts import render
from django.views.generic import FormView, UpdateView
from common.mixins import JSONResponseMixin, AdminUserRequiredMixin
from common.utils import get_object_or_none
from .. import forms
from django import forms as oforms
from django.views import View
import xlrd
import xlwt
import uuid
import datetime
import pandas as pd
import os


# Create your views here.

def index(request):
    return render(request, 'data_analy/index.html')


def relation_v(request):
    context = {}
    context['hello'] = 'Hello World!'
    return render(request, 'data_analy/data_index.html', context)


def confidence_v(request):
    context = {}
    context['hello'] = 'Hello World!'
    return render(request, 'data_analy/data_index.html', context)


def fitfunc_v(request):
    context = {}
    context['hello'] = 'Hello World!'
    return render(request, 'data_analy/data_index.html', context)

def data_list(request):
    queryset = Assets.objects.all()
    qd = btUrldecode(request.GET, url='asset')

    totals = queryset.filter(**qd['query']).count()
    devices = queryset.filter(**qd['query']).order_by(qd['orderName'])[qd['start']:qd['offset']]

    return JsonResponse({
        'total': totals,
        'data': query2dict(devices, url='asset'),
        '_': request.GET.get('_', 0)
    })

# @login_required
def data_index(request):
    tmp_path = os.path.join("..", "test.xlsx")
    data = pd.read_excel(io=tmp_path, sheet_name='Sheet1', header=0)
    print(data)
    collist = data.columns
    widgeto = {
    row: oforms.TextInput(attrs={'id': 'search_{}'.format(row), 'name': 's_{}'.format(row), 'class': 'col-sm-6'}) for
    row in collist}

    class Modeln:
        def __init__(self):
            pass

    class DataSearchFormn:
        class Meta:
            model = Modeln
            fields = collist
            widgets = widgeto
            # fields = ['purchase', 'num', 'type', 'model', 'serial', 'supplier', 'status', 'brand', 'expired']
            # widgets = {
            #     'purchase': oforms.Select(
            #         attrs={'id': 'search_purchase', 'name': 's_purchase', 'style': 'height: 38px;',
            #                'class': 'col-sm-6'}),
            #     'num': oforms.TextInput(attrs={'id': 'search_num', 'name': 's_num', 'class': 'col-sm-6'}),
            #     'type': oforms.Select(
            #         attrs={'id': 'search_type', 'name': 's_type', 'style': 'height: 38px;', 'class': 'col-sm-6'}),
            #     'model': oforms.TextInput(attrs={'id': 'search_model', 'name': 's_model', 'class': 'col-sm-6'}),
            #     'serial': oforms.TextInput(attrs={'id': 'search_serial', 'name': 's_serial', 'class': 'col-sm-6'}),
            #     'supplier': oforms.TextInput(
            #         attrs={'id': 'search_supplier', 'name': 's_supplier', 'class': 'col-sm-6'}),
            #     'status': oforms.Select(
            #         attrs={'id': 'search_status', 'name': 's_status', 'style': 'height: 38px;', 'class': 'col-sm-6'}),
            #     'brand': oforms.TextInput(attrs={'id': 'search_brand', 'name': 's_brand', 'class': 'col-sm-6'}),
            #     'expired': oforms.DateTimeInput(
            #         attrs={'id': 'search_expired', 'name': 's_expired', 'class': 'col-sm-6'}),
            # }

    sform = DataSearchFormn()
    # sform = {}
    # Model = {}
    # cform = forms.DataAddForm()
    # cform = {}
    return render(request, 'data_analy/data_index.html', {
        'searchForm': sform,
        # 'createForm': cform,
        'collist': collist,
    })

class DataExportView(View):
    def get(self, request, *args, **kwargs):
        spm = request.GET.get('spm', '')
        assets_id = cache.get(spm, [Assets.objects.first().id])  # assets_id 要导出的id列表
        assets = Assets.objects.filter(id__in=assets_id)
        fields = [
            field for field in Assets._meta.fields
            if field.name not in ['create', 'update']
        ]
        # fields 是表字段名
        filename = 'assets-{}.xls'.format(timezone.now().strftime('%Y-%m-%d_%H-%M-%S'))
        response = HttpResponse(content_type='application/vnd.ms-excel')
        response['Content-Disposition'] = 'attachment; filename="%s"' % filename

        writer = xlwt.Workbook(encoding='utf-8')
        sheet1 = writer.add_sheet(u'sheet1')  # 创建sheet1
        # 写入第一行标题
        header = [field.verbose_name for field in fields]
        for i in range(0, len(header)):
            sheet1.write(0, i, header[i])

        # 循环写入
        row = 1
        for asset in assets:
            data = [getattr(asset, field.name) for field in fields]
            for i in range(0, len(data)):
                if isinstance(data[i], datetime.datetime):
                    data[i] = data[i].strftime('%Y/%m/%d')  # 处理时间格式问题
                if isinstance(data[i], Purchase):
                    data[i] = data[i].num
                sheet1.write(row, i, data[i])
            row += 1

        output = BytesIO()
        writer.save(output)
        output.seek(0)
        response.write(output.getvalue())
        return response

    def post(self, request, *args, **kwargs):
        try:
            assets_id = json.loads(request.body).get('assets_id', [])
        except ValueError:
            return HttpResponse('Json object not valid', status=400)
        spm = uuid.uuid4().hex
        cache.set(spm, assets_id, 300)  # id列表存入redis,生成跳转链接进行上面定义的get下载
        url = reverse_lazy('assets:asset-export') + '?spm=%s' % spm
        return JsonResponse({'redirect': url})

class BulkImportDataView(AdminUserRequiredMixin, JSONResponseMixin, FormView):
    form_class = forms.FileForm

    def form_valid(self, form):
        file = form.cleaned_data['file']
        wb = xlrd.open_workbook(filename=None, file_contents=file.read())
        sheet1 = wb.sheet_by_index(0)  # 第一个
        header_ = sheet1.row_values(0)  # 表格头
        # fields = [field for field in Purchase._meta.fields]
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


# @login_required
def data_add(request):
    if request.method == 'POST':
        form = forms.AssetCreateForm(request.POST)
        if form.is_valid():
            form.save()
            return JsonResponse({'status': 'true'})
        else:
            raise form.error_class
    else:
        return HttpResponseRedirect(reverse('assets:asset-index'))
