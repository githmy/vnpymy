# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render, HttpResponse, HttpResponseRedirect, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, permission_required
from django.db import transaction
# from account.decorations import user_groups
# from django.views.decorators.csrf import csrf_exempt
from django.db.models import F, Q
from django.conf import settings
import json, os
import datetime, time
# from .models import DutyTab
# from .forms import Calendaradd, Calendarcha
# from common.utils import out_month_week
# from util.data2excel import write_data_to_excel, excel2mysql, excel2mysql_audit


# Create your views here.
def urldecode(querydict):
    # for query in querydict:
    #     if querydict[query] is not u'':
    #         print query + ':' + querydict[query]
    id = querydict.get('id', -1)  # offset 偏移位置
    type = querydict.get('type', 'month')  # 每页显示条目数
    name = querydict.get('name', 'abc')
    segm = querydict.get('segm', '早')  # 排序的字段名
    start = querydict.get('start', '2017-07-13')  # asc or desc
    date = querydict.get('date', '2099-07-13 00:00:00')  # 排序的字段名
    dateo = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").date()
    return {
        'id': id,
        'start': start,
        'date': dateo,
        'segm': segm,
        'name': name,
        'type': type,
    }


def query2dict(query):
    lists = []
    for q in query:
        lists.append({
            "title": q.segm + " : " + q.duty_name if q.segm else "  : " + q.duty_name,
            "start": q.duty_timef.strftime('%Y-%m-%d') if q.duty_timef else None,
            # "start": q.duty_timef.strftime('%Y-%m-%d %X') if q.duty_timef else None,
            "allDay": True,
            # "backgroundColor": q.hostname,
            # "borderColor": q.serial_num,
            # "color": "#0000ff",
            # "textColor": "#0000ff",
            "_id": "_fc" + str(q.id),
            # "editable": False,
        })
    return lists


# ######################1.新建部分#############################
@login_required
def web_tool(request):
    return render(request, 'op/web_tool.html', {
        'menu': '值班日历',
        'menu_url': '/op/calendar/',
    })


# @user_groups
@login_required
def duty_calendar(request):
    return render(request, 'op/calendar.html', {
        'menu': '值班日历',
        'menu_url': '/op/calendar/',
    })


# @user_groups
@login_required
def duty_data(request):
    # 1.解析参数
    querydict = urldecode(request.GET)
    # 2.获取数据
    dbdata = DutyTab.objects.filter(duty_timef__range=out_month_week(querydict['date']))
    return HttpResponse(json.dumps({
        'status': 1,
        'data': query2dict(dbdata),
    }))


# @user_groups
@login_required
def duty_show(request):
    pass
    # querydict = urldecode(request.GET)
    # current_draw = request.GET.get('draw', 0)  # draw 必须
    # list_obj = List.objects.all()  # 分表的临时数据
    # if querydict['search']:
    #     search_obj = list_obj.filter(Q(cha_name__icontains=querydict['search']) |
    #                                  Q(cha_desc__icontains=querydict['search']) |
    #                                  Q(applied_by__icontains=querydict['search']) |
    #                                  Q(approved_by__icontains=querydict['search']) |
    #                                  Q(status__icontains=querydict['search']) |
    #                                  Q(result__icontains=querydict['search']) |
    #                                  Q(cha_id__icontains=querydict['search'])).order_by('-' + querydict['order_col'])
    #     totals = search_obj.count()
    #     show_obj = search_obj[querydict['start']:querydict['offset']]
    # else:
    #     totals = list_obj.count()
    #     show_obj = list_obj.order_by('-' + querydict['order_col'])[querydict['start']:querydict['offset']]
    # return HttpResponse(json.dumps({
    #     'draw': current_draw,
    #     'recordsTotal': totals,
    #     'recordsFiltered': totals,
    #     'data': list_query2dict(show_obj),
    # }))


# @user_groups
@login_required
@permission_required('op.add_dutytab')
@transaction.atomic
def duty_add(request):
    text_list = ['duty_even']
    ret_url = '/op/calendar/'
    if request.method == 'POST':
        form = Calendaradd(request.POST)
        if form.is_valid():
            form.save()
            return redirect(ret_url)
        else:
            return HttpResponse(u'表单字段填写有误。')
    else:
        form = Calendaradd()
        return render(request, 'op/add.html',
                      {'form': form, 'text_list': text_list, 'menu_url': ret_url, 'title': '新建值日'})


# @user_groups
@transaction.atomic
@permission_required('op.change_dutytab')
@login_required
def duty_edit(request, evenid):
    ret_url = '/op/calendar/'
    del_url = '/op/calendar/' + str(evenid) + '/del/'
    text_list = ['值班事件']
    dutab = get_object_or_404(DutyTab, id=evenid)
    if request.method == 'POST':
        form = Calendarcha(request.POST.copy(), instance=dutab)
        if form.is_valid():
            form.save()
            return redirect(ret_url)
        else:
            return HttpResponse(u'表单字段填写有误。')
    else:
        form = Calendarcha(instance=dutab)
        # print form
        return render(request, 'op/edit.html',
                      {'form': form, 'text_list': text_list, 'menu_url': ret_url, 'del_url': del_url, 'title': '值日编辑'})


# @user_groups
@login_required
@permission_required('op.delete_dutytab')
@transaction.atomic
def duty_del(request, evenid):
    try:
        DutyTab.objects.filter(id=evenid).delete()
        return redirect('/op/calendar/')
    except Exception as e:
        return HttpResponse(u'不存在此条目。')


# @user_groups
@login_required
def duty_list(request):
    return render(request, 'op/tables.html', {
        'menu': '值班交接',
        'menu_url': '/op/list/',
    })


# @csrf_exempt
# @user_groups
@login_required
def duty_down(request):
    downame = request.GET.get('dname', None)
    datef = request.GET.get('datef', None)
    datet = request.GET.get('datet', None)
    ori_name = '值日列表'
    if os.path.exists('./upload/dump'):
        pass
    else:
        os.makedirs('./upload/dump')
    headarr = [
        # "id",
        "duty_name",
        "duty_timef",
        "segm",
        "duty_even",
    ]
    connecbase = ','
    connecstr = connecbase.join(headarr)
    sqlstr = 'select ' + connecstr + ' from duty_dutytab where duty_timef <= "' + datet + ' 23:59:59" and duty_timef >= "' + datef + ' 00:00:00"'
    write_data_to_excel('./upload/dump', ori_name, sqlstr, headarr)
    response = HttpResponse()
    response['Content-Disposition'] = 'attachment;filename="{0}"'.format(downame + '.xls').encode('gb2312')
    # filename.txt为将要被下载的文件名
    full_path = os.path.join('./upload/dump/', ori_name + '.xls')
    if os.path.exists(full_path):
        response['Content-Length'] = os.path.getsize(full_path)  # 可不加
        content = open(full_path, 'rb').read()
        response.write(content)
        return response
    else:
        return HttpResponse(u'文件未找到')


# @user_groups
@login_required
@permission_required('op.add_dutytab')
def duty_batchin(request):
    # 1.初始准备
    ret_url = '/op/calendar/'
    if os.path.exists('./upload/input'):
        pass
    else:
        os.makedirs('./upload/input')
    inputname = os.path.join("./upload/input", "值日_导入.xls")

    # 2.上传文件
    if request.method == 'POST':
        # request.FILES["myfile"]或者request.FILES.get("myfile", None)
        myFile = request.FILES.get("file", None)  # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return HttpResponse("no files for upload!")
        destination = open(inputname, 'wb+')  # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():  # 分块写入文件
            destination.write(chunk)
        destination.close()
        # return redirect(ret_url)
    else:
        return render(request, 'upfile.html', {'menu_url': ret_url, 'title': '编辑'})

    # 3.导入文件
    tablename = 'duty_dutytab'
    # 忽略 字符 数组
    ignarr = ['duty_timet', 'duty_even']
    # 添加 字段和默认值 数组
    addarr = [[], []]
    # 执行转换
    try:
        res = excel2mysql(inputname, tablename, ignarr, addarr)
        if res is True:
            return redirect(ret_url)
        else:
            return HttpResponse(res)
    except Exception as e:
        return HttpResponse(e)
