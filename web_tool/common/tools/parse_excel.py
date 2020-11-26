# -*- coding: utf-8 -*-
import xlrd
import xlwt
from datetime import date, datetime

from server.models import Virtual, Summary


def read_excel():
    # 打开文件
    workbook = xlrd.open_workbook(r'v2.xlsx')
    # 获取所有sheet
    print(workbook.sheet_names())  # [u'sheet1', u'sheet2']
    # sheet2_name = workbook.sheet_names()[0]

    # 根据sheet索引或者名称获取sheet内容
    # sheet2 = workbook.sheet_by_index(0)  # sheet索引从0开始
    sheet2 = workbook.sheet_by_name('Sheet1')

    # sheet的名称，行数，列数
    print(sheet2.name, sheet2.nrows, sheet2.ncols)

    # 获取整行和整列的值（数组）
    # rows = sheet2.row_values(3)  # 获取第四行内容
    # cols = sheet2.col_values(2)  # 获取第三列内容
    # print rows
    # print cols

    # 获取单元格内容
    # print sheet2.cell(1, 0).value.encode('utf-8')
    # print sheet2.cell_value(1, 0).encode('utf-8')
    # print sheet2.row(1)[0].value.encode('utf-8')

    # 获取单元格内容的数据类型
    # print sheet2.cell(1, 0).ctype


def parse_excel(path):
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_name('Sheet1')
    # sheet的名称，行数，列数
    # print sheet.name, sheet.nrows, sheet.ncols
    name_cols = sheet.col_values(0)  # 获取第1列内容
    tenant_cols = sheet.col_values(1)  # 获取第2列内容
    role_cols = sheet.col_values(2)  # 获取第3列内容
    env_cols = sheet.col_values(3)  # 获取第4列内容
    zone_cols = sheet.col_values(4)  # 获取第5列内容
    for name, tenant, role, env, zone in zip(name_cols, tenant_cols, role_cols, env_cols, zone_cols):
        Virtual.objects.filter(username=name, zone=zone).update(tenant=tenant, env=env, role=role)


def parse_excel_3(path):
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_name('Sheet1')
    # sheet的名称，行数，列数
    # print sheet.name, sheet.nrows, sheet.ncols
    company_cols = sheet.col_values(6)  # 获取第1列内容
    role_cols = sheet.col_values(7)  # 获取第2列内容
    contact_cols = sheet.col_values(8)  # 获取第3列内容
    for company, role, contact in zip(company_cols, role_cols, contact_cols):
        Summary.objects.filter(company=company, role=role).update(contact=contact)
        # "update server_summary set contact='{}' where company='{}' and role='{}';".format(contact, company, role)


def run(path):
    parse_excel(path)
    parse_excel_3(path)


if __name__ == '__main__':
    run('v3.xlsx')
