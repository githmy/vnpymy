# # -- coding: utf-8 --
# from __future__ import absolute_import, unicode_literals
# import os
# from datetime import timedelta
# # from celery import Celery
# # from celery.schedules import crontab
#
# # set the default Django settings module for the 'celery' program.
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rainbow.settings')
#
# from django.conf import settings
#
# app = Celery('rainbow', broker=settings.CELERY_BROKER_URL, backend=settings.CELERY_RESULT_BACKEND)
#
# # Using a string here means the worker will not have to
# # pickle the object when using Windows.
# app.config_from_object('django.conf:settings')
# app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)
#
# app.conf.update(
#     # CELERYBEAT_SCHEDULE={
#     #     'refresh-asset-hardware-info': {
#     #         'task': 'assets.tasks.update_assets_hardware_period',
#     #         'schedule': 60*60*60*24,
#     #         'args': (),
#     #     },
#     #     'test-admin-user-connective': {
#     #         'task': 'assets.tasks.test_admin_user_connective_period',
#     #         'schedule': 60*60*60,
#     #         'args': (),
#     #     }
#     # }
#
#     # CELERYBEAT_SCHEDULE={
#     #     'add-every-5-minutes': {
#     #         'task': 'monitor.tasks.update_mysql',
#     #         'schedule': crontab(minute='3-59/5'),
#     #         'args': ()
#     #     },
#     #     'add-every-30-minutes1': {
#     #         'task': 'monitor.tasks.update_host',
#     #         'schedule': crontab(minute='5,35'),
#     #         'args': ()
#     #     },
#     #     'add-every-30-minutes2': {
#     #         'task': 'monitor.tasks.update_tend',
#     #         'schedule': crontab(minute='10,40'),
#     #         'args': ()
#     #     },
#     #     'add-every-1-hour': {
#     #         'task': 'monitor.tasks.update_alert',
#     #         'schedule': crontab(minute='45'),
#     #         'args': ()
#     #     },
#     #     'add-every-1-day': {
#     #         'task': 'monitor.tasks.update_day',
#     #         'schedule': crontab(minute='50', hour='23'),
#     #         'args': ()
#     #     },
#     #     'add-every-1-week': {
#     #         'task': 'monitor.tasks.update_week',
#     #         'schedule': crontab(minute='55', hour='23', day_of_week='0'),
#     #         'args': ()
#     #     },
#     #     'add-every-1-month': {
#     #         'task': 'monitor.tasks.update_month',
#     #         'schedule': crontab(minute='5', hour='0', day_of_month='1'),
#     #         'args': ()
#     #     },
#     # }
# )
