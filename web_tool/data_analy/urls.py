
"""Mebius URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url

from . import views


urlpatterns = [
    url(r'^web_tool/$', views.web_tool, name='web_tool'),
    url(r'^calendar/$', views.duty_calendar, name='calendar'),
    url(r'^calendar/data', views.duty_data, name='calend_data'),
    url(r'^calendar/add$', views.duty_add, name='add'),
    url(r'^calendar/edit/(?P<evenid>\d+)/$', views.duty_edit, name='edit'),
    # url(r'^(?P<evenid>\d+)/change/$', views.duty_change, name='change'),
    # url(r'^(?P<evenid>\d+)/show/$', views.duty_show, name='show'),
    url(r'^calendar/(?P<evenid>\d+)/del/$', views.duty_del, name='del'),
    url(r'^list/$', views.duty_list, name='list'),
    url(r'^down$', views.duty_down, name='duty_down'),
    url(r'^batchin$', views.duty_batchin, name='duty_batchin'),

]
