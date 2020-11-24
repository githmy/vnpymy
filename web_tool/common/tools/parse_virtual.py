# -*- coding: utf-8 -*-
from server.models import Virtual
from server.models import Summary
from ast import literal_eval


def parse():
    # 1.删除老表type=vMachine
    Summary.objects.filter(type='vMachine').delete()

    # 2.解析两表
    bulklist = []
    for item in Virtual.objects.all():
        bulklist.append(Summary(
            physical_machine=item.physical,
            hostname=item.virtual,
            region=item.zone,
            date=item.create.strftime('%Y-%m-%d'),
            status=item.status,
            ipaddress=item.ip + "," + item.elasticIP,
            cpu=item.cpu,
            mem=item.mem,
            os=item.osType,
            role=item.role,
            hdisk=cooking(literal_eval(item.dataVolume)),
            filesystem=item.sysVolumeCapability,
            env=item.env,
            company=item.tenant,
            type='vMachine',
            location='JQ',
            field='Server',
            vm=1
        ))
    # 3.插入
    Summary.objects.bulk_create(bulklist)


def cooking(datas):
    if datas:
        volume = ''
        for vo in datas:
            volume += '+' + str(vo['capability'])
        return volume
    else:
        return None


if __name__ == '__main__':
    parse()
