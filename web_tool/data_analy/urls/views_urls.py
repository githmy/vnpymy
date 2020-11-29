from django.conf.urls import url, include
from ..views import views

urlpatterns = [
    # 数据分析管理
    url(r'^$', views.index, name='data_analy-index'),
    url(r'^show/$', views.data_index, name='data_show'),
    url(r'^prob_check/$', views.prob_check_v, name='prob_check_views'),
    url(r'^relation/$', views.relation_v, name='relation_views'),
    url(r'^confidence/$', views.confidence_v, name='confidence_views'),
    url(r'^fitfunc/$', views.fitfunc_v, name='fitfunc_views'),
    url(r'^api-data_list/$', views.data_list, name='api-data-list'),
    url(r'^api-data_fit/$', views.data_fit, name='api-data-fit'),
    url(r'^api-data_relation/$', views.data_relation, name='api-data-relation'),
    url(r'^api-data_confidence/$', views.data_confidence, name='api-data-confidence'),
    url(r'^clean/$', views.data_clean, name='clean'),
    # url(r'^data/add/$', views.data_add, name='data-add'),
    # url(r'^purchase/(?P<pk>[0-9]+)/change/$', views.PurchaseUpdateView.as_view(), name='purchase-change'),
    # url(r'^purchase/(?P<pk>[0-9]+)/delete/$', views.purchase_delete, name='purchase-delete'),
    # url(r'^purchase/export/$', views.PurchaseExportView.as_view(), name='purchase-export'),
    url(r'^import/$', views.BulkImportDataView.as_view(), name='data-import'),
    url(r'^export/$', views.DataExportView.as_view(), name='data-export'),
]
