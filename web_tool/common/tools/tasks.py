# -*- coding: utf-8 -*-

from cmdb.celery import app
from server.tools.parse_excel import run as excel_run
from server.tools.parse_op import run as op_run
from server.tools.parse_virtual import parse
from cmdb.settings import BASE_DIR


@app.task(bind=True, max_retries=None)
def openStackTask(self):
    print('1.openstack task start: {0}'.format(self.request.id))
    op_run()
    print('2.parse excel.')
    excel_run(BASE_DIR+r'/server/tools/v3.xlsx')
    print('3.virtual into summary')
    parse()