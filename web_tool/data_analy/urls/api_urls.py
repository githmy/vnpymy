# from django.conf.urls import url, include
# from ..views import api_views
# # --- rest api ---
# from rest_framework import routers
#
# # 使用URL路由来管理我们的API
# router = routers.DefaultRouter()
# router.register(r'v1/users', api_views.UserViewSet, 'users')
# router.register(r'v1/groups', api_views.GroupViewSet, 'groups')
# router.register(r'v1/assets', api_views.AssetViewSet, 'assets')
# router.register(r'v1/idc', api_views.DataCenterViewSet, 'idc')
# router.register(r'v1/cabinet', api_views.CabinetViewSet, 'cabinet')
#
# urlpatterns = [
#     url(r'^v1/purchase/list', api_views.purchase_list, name='purchase-list'),
#     url(r'^v1/assets/list', api_views.asset_list, name='asset-list'),
#     url(r'^v1/physics/list', api_views.physics_list, name='physics-list'),
#     url(r'^v1/network/list', api_views.network_list, name='network-list'),
#     url(r'^v1/virtual/list', api_views.virtual_list, name='virtual-list'),
#     url(r'^v1/server/list', api_views.server_list, name='server-list'),
#     url(r'^v1/idc/(?P<pk>\d+)/cabinet/$', api_views.IDCabinetApi.as_view()),  # 查看idc下的机柜
# ]
# urlpatterns += router.urls
