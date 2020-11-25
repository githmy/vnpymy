from django.conf.urls import url, include
from ..views import views

urlpatterns = [
    # 采购单管理
    url(r'^$', views.index, name='data_analy-index'),
    url(r'^show/$', views.data_index, name='data_show'),
    url(r'^relation/$', views.relation_v, name='relation_views'),
    url(r'^confidence/$', views.confidence_v, name='confidence_views'),
    url(r'^fitfunc/$', views.fitfunc_v, name='fitfunc_views'),
    url(r'^api-data_list/$', views.data_list, name='api-data-list'),
    url(r'^data/add/$', views.data_add, name='data-add'),
    # url(r'^purchase/(?P<pk>[0-9]+)/change/$', views.PurchaseUpdateView.as_view(), name='purchase-change'),
    # url(r'^purchase/(?P<pk>[0-9]+)/delete/$', views.purchase_delete, name='purchase-delete'),
    # url(r'^purchase/export/$', views.PurchaseExportView.as_view(), name='purchase-export'),
    url(r'^import/$', views.BulkImportDataView.as_view(), name='data-import'),
    url(r'^export/$', views.DataExportView.as_view(), name='data-export'),
    #
    # # 资产管理
    # url(r'^asset/$', views.asset_index, name='asset-index'),
    # url(r'^asset/add/$', views.asset_add, name='asset-add'),
    # url(r'^asset/(?P<pk>[0-9]+)/$', views.asset_show, name='asset-show'),
    # url(r'^asset/(?P<pk>[0-9]+)/change/$', views.AssetUpdateView.as_view(), name='asset-change'),
    # url(r'^asset/(?P<pk>[0-9]+)/delete/$', views.asset_delete, name='asset-delete'),
    # url(r'^asset/export/$', views.AssetExportView.as_view(), name='asset-export'),
    # url(r'^asset/import/$', views.BulkImportAssetView.as_view(), name='asset-import'),
    #
    # # 物理服务器管理
    # url(r'^physics/$', views.physics_index, name='physics-index'),
    # url(r'^physics/add/$', views.physics_add, name='physics-add'),
    # url(r'^physics/(?P<pk>[0-9]+)/$', views.physics_show, name='physics-show'),
    # url(r'^physics/(?P<pk>[0-9]+)/change/$', views.PhysicsUpdateView.as_view(), name='physics-change'),
    # url(r'^physics/(?P<pk>[0-9]+)/delete/$', views.physics_delete, name='physics-delete'),
    # url(r'^physics/export/$', views.PhysicsExportView.as_view(), name='physics-export'),
    # url(r'^physics/import/$', views.BulkImportPhysicsView.as_view(), name='physics-import'),
    #
    # # 虚拟服务器管理
    # url(r'^virtual/$', views.virtual_index, name='virtual-index'),
    # # url(r'^virtual/add/$', views.virtual_add, name='virtual-add'),
    # url(r'^virtual/(?P<pk>[0-9]+)/$', views.virtual_show, name='virtual-show'),
    # url(r'^virtual/(?P<pk>[0-9]+)/change/$', views.VirtualUpdateView.as_view(), name='virtual-change'),
    # url(r'^virtual/(?P<pk>[0-9]+)/delete/$', views.virtual_delete, name='virtual-delete'),
    # url(r'^virtual/export/$', views.VirtualExportView.as_view(), name='virtual-export'),
    # # url(r'^virtual/import/$', views.BulkImportVirtualView.as_view(), name='virtual-import'),
    #
    # # 服务器管理
    # url(r'^server/$', views.ServerListView.as_view(), name='server-index'),
    #
    # # 网络设备管理
    # url(r'^network/$', views.network_index, name='network-index'),
    # url(r'^network/(?P<pk>[0-9]+)/$', views.network_show, name='network-show'),
    # url(r'^network/add/$', views.network_add, name='network-add'),
    # url(r'^network/(?P<pk>[0-9]+)/delete/$', views.network_delete, name='network-delete'),
    # url(r'^network/(?P<pk>[0-9]+)/change/$', views.NetworkUpdateView.as_view(), name='network-change'),
    # url(r'^network/export/$', views.NetworkExportView.as_view(), name='network-export'),
    # url(r'^network/import/$', views.BulkImportNetworkView.as_view(), name='network-import'),
]
