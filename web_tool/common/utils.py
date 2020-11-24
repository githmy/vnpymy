# -*- coding: utf-8 -*-

from collections import OrderedDict
from six import string_types
import base64
import os
from itertools import chain
import string
import logging
import datetime
import json
import time
import hashlib
from email.utils import formatdate
import calendar
import threading
from six import StringIO
# from itsdangerous import TimedJSONWebSignatureSerializer, JSONWebSignatureSerializer, \
#     BadSignature, SignatureExpired
from django.shortcuts import reverse as dj_reverse
from django.conf import settings
from django.utils import timezone

from datetime import datetime, date
from datetime import timedelta

SECRET_KEY = settings.SECRET_KEY


def reverse(view_name, urlconf=None, args=None, kwargs=None,
            current_app=None, external=False):
    url = dj_reverse(view_name, urlconf=urlconf, args=args,
                     kwargs=kwargs, current_app=current_app)

    if external:
        url = settings.SITE_URL.strip('/') + url
    return url


def get_object_or_none(model, **kwargs):
    try:
        obj = model.objects.get(**kwargs)
    except model.DoesNotExist:
        return None
    return obj


class Signer(object):
    """用来加密,解密,和基于时间戳的方式验证token"""

    def __init__(self, secret_key=SECRET_KEY):
        self.secret_key = secret_key

    def sign(self, value):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        s = JSONWebSignatureSerializer(self.secret_key)
        return s.dumps(value)

    def unsign(self, value):
        s = JSONWebSignatureSerializer(self.secret_key)
        try:
            return s.loads(value)
        except BadSignature:
            return {}

    def sign_t(self, value, expires_in=3600):
        s = TimedJSONWebSignatureSerializer(self.secret_key, expires_in=expires_in)
        return str(s.dumps(value), encoding="utf8")

    def unsign_t(self, value):
        s = TimedJSONWebSignatureSerializer(self.secret_key)
        try:
            return s.loads(value)
        except (BadSignature, SignatureExpired):
            return {}


def date_expired_default():
    try:
        years = int(settings.CONFIG.DEFAULT_EXPIRED_YEARS)
    except TypeError:
        years = 70
    return timezone.now() + timezone.timedelta(days=365 * years)


def combine_seq(s1, s2, callback=None):
    for s in (s1, s2):
        if not hasattr(s, '__iter__'):
            return []

    seq = chain(s1, s2)
    if callback:
        seq = map(callback, seq)
    return seq


def search_object_attr(obj, value='', attr_list=None, ignore_case=False):
    """It's provide a method to search a object attribute equal some value

    If object some attribute equal :param: value, return True else return False

    class A():
        name = 'admin'
        age = 7

    :param obj: A object
    :param value: A string match object attribute
    :param attr_list: Only match attribute in attr_list
    :param ignore_case: Ignore case
    :return: Boolean
    """
    if value == '':
        return True

    try:
        object_attr = obj.__dict__
    except AttributeError:
        return False

    if attr_list is not None:
        new_object_attr = {}
        for attr in attr_list:
            new_object_attr[attr] = object_attr.pop(attr)
        object_attr = new_object_attr

    if ignore_case:
        if not isinstance(value, string_types):
            return False

        if value.lower() in map(string.lower, map(str, object_attr.values())):
            return True
    else:
        if value in object_attr.values():
            return True
    return False


def get_logger(name=None):
    return logging.getLogger('jumpserver.%s' % name)


def int_seq(seq):
    try:
        return map(int, seq)
    except ValueError:
        return seq


def timesince(dt, since='', default="just now"):
    """
    Returns string representing "time since" e.g.
    3 days, 5 hours.
    """

    if since is '':
        since = datetime.datetime.utcnow()

    if since is None:
        return default

    diff = since - dt

    periods = (
        (diff.days / 365, "year", "years"),
        (diff.days / 30, "month", "months"),
        (diff.days / 7, "week", "weeks"),
        (diff.days, "day", "days"),
        (diff.seconds / 3600, "hour", "hours"),
        (diff.seconds / 60, "minute", "minutes"),
        (diff.seconds, "second", "seconds"),
    )

    for period, singular, plural in periods:
        if period:
            return "%d %s" % (period, singular if period == 1 else plural)
    return default


def setattr_bulk(seq, key, value):
    def set_attr(obj):
        setattr(obj, key, value)
        return obj

    return map(set_attr, seq)


_STRPTIME_LOCK = threading.Lock()

_GMT_FORMAT = "%a, %d %b %Y %H:%M:%S GMT"
_ISO8601_FORMAT = "%Y-%m-%dT%H:%M:%S.000Z"


def to_unixtime(time_string, format_string):
    time_string = time_string.decode("ascii")
    with _STRPTIME_LOCK:
        return int(calendar.timegm(time.strptime(time_string, format_string)))


def http_date(timeval=None):
    """返回符合HTTP标准的GMT时间字符串，用strftime的格式表示就是"%a, %d %b %Y %H:%M:%S GMT"。
    但不能使用strftime，因为strftime的结果是和locale相关的。
    """
    return formatdate(timeval, usegmt=True)


def http_to_unixtime(time_string):
    """把HTTP Date格式的字符串转换为UNIX时间（自1970年1月1日UTC零点的秒数）。

    HTTP Date形如 `Sat, 05 Dec 2015 11:10:29 GMT` 。
    """
    return to_unixtime(time_string, _GMT_FORMAT)


def iso8601_to_unixtime(time_string):
    """把ISO8601时间字符串（形如，2012-02-24T06:07:48.000Z）转换为UNIX时间，精确到秒。"""
    return to_unixtime(time_string, _ISO8601_FORMAT)


def capacity_convert(size, expect='auto', rate=1000):
    """
    :param size: '100MB', '1G'
    :param expect: 'K, M, G, T
    :param rate: Default 1000, may be 1024
    :return:
    """
    rate_mapping = (
        ('K', rate),
        ('KB', rate),
        ('M', rate ** 2),
        ('MB', rate ** 2),
        ('G', rate ** 3),
        ('GB', rate ** 3),
        ('T', rate ** 4),
        ('TB', rate ** 4),
    )

    rate_mapping = OrderedDict(rate_mapping)

    std_size = 0  # To KB
    for unit in rate_mapping:
        if size.endswith(unit):
            try:
                std_size = float(size.strip(unit).strip()) * rate_mapping[unit]
            except ValueError:
                pass

    if expect == 'auto':
        for unit, rate_ in rate_mapping.items():
            if rate > std_size / rate_ > 1:
                expect = unit
                break
    expect_size = std_size / rate_mapping[expect]
    return expect_size, expect


def sum_capacity(cap_list):
    total = 0
    for cap in cap_list:
        size, _ = capacity_convert(cap, expect='K')
        total += size
    total = '{} K'.format(total)
    return capacity_convert(total, expect='auto')


signer = Signer()


class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, datetime.date):
            return obj.strftime("%Y-%m-%d")
        else:
            return json.JSONEncoder.default(self, obj)


# ************************** QYB start **********************************

def get_endweek_of_lastmonth(date):
    """
    获取上个月第一天的日期，然后加21天就是22号的日期
    :return: 返回日期
    """
    # today = datetime.today()
    today = date
    year = today.year
    month = today.month
    if month == 1:
        month = 12
        year -= 1
    else:
        month -= 1
    res = datetime(year, month, 1) + timedelta(days=21)
    return res.strftime('%Y-%m-%d %X')


def get_endweek_of_nextmonth(date):
    """
    获取下个月的22号的日期
    :return: 返回日期
    """
    # today = datetime.today()
    today = date
    year = today.year
    month = today.month
    if month == 12:
        month = 1
        year += 1
    else:
        month += 1
    res = datetime(year, month, 1) + timedelta(days=21)
    return res.strftime('%Y-%m-%d %X')


def get_1stweek_of_nextmonth(date):
    # today = datetime.today()
    today = date
    year = today.year
    month = today.month
    if month == 12:
        month = 1
        year += 1
    else:
        month += 1
    res = datetime(year, month, 1) + timedelta(days=6)
    return res.strftime('%Y-%m-%d %X')


def get_1stDay_Of_thismonth(date):
    # d = datetime.now()
    d = date
    # c = calendar.Calendar()

    year = d.year
    month = d.month

    if month == 1:
        month = 12
        year -= 1
    else:
        month -= 1
    days = calendar.monthrange(year, month)[1]
    return (datetime(year, month, 1) + timedelta(days=days)).strftime('%Y-%m-%d %X')


def out_month_week(date):
    return [get_endweek_of_lastmonth(date), get_1stweek_of_nextmonth(date)]


def date2str(obj):
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, date):
        return obj.strftime("%Y-%m-%d")
    else:
        return obj

# ************************** QYB end   **********************************
