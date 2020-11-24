from django import forms
from django.utils.translation import ugettext_lazy as _
# from .models import Assets, PhysicsServers, Company, NetworkDevices, Purchase, VirtualServers, Region
from django.core.exceptions import ValidationError


class FileForm(forms.Form):
    file = forms.FileField()


class DataSearchForm(forms.ModelForm):
    def __init__(self, Model):
        # model = PhysicsServers
        model = Model
        fields = ['ip', 'hostname', 'tenant', 'region', 'cabinet', 'asset']
        widgets = {
            'ip': forms.TextInput(attrs={'class': 'col-sm-6', 'id': 'search_ip', 'name': 's_ip'}),
            'hostname': forms.TextInput(attrs={'class': 'col-sm-6', 'id': 'search_hostname', 'name': 's_hostname'}),
            # 'tenant': forms.Select(
            #     attrs={'style': 'height: 38px;', 'class': 'col-sm-6', 'id': 'search_tenant', 'name': 's_tenant'}),
            'region': forms.Select(
                attrs={'style': 'height: 38px;', 'class': 'col-sm-6', 'id': 'search_region', 'name': 's_region'}),
            'cabinet': forms.Select(
                attrs={'style': 'height: 38px;', 'class': 'col-sm-6', 'id': 'search_cabinet', 'name': 's_cabinet'}),
            'asset': forms.Select(
                attrs={'style': 'height: 38px;', 'class': 'col-sm-6', 'id': 'search_asset', 'name': 's_asset'}),
        }
        # help_texts = {
        #
        # }
        # error_messages = {
        #
        # }


class DataAddForm(forms.ModelForm):
    chosestatus = forms.ChoiceField(choices=[('线上已分配', '线上已分配'), ('线上未分配', '线上未分配')],
                                    required=True,
                                    help_text='* required',
                                    widget=forms.Select(
                                        attrs={
                                            'style': 'height: 25px;width: 150px;'
                                        }))
    chosecenter = forms.ChoiceField(choices=[('JQ', 'JQ'), ('PK', 'PK')],
                                    required=True,
                                    help_text='* required',
                                    widget=forms.Select(
                                        attrs={
                                            'style': 'height: 25px;width: 150px;'
                                        }))

    def __init__(self, Model):
        # model = PhysicsServers
        model = Model
        fields = "__all__"

        widgets = {
            'hostname': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
            'gigabit_ip': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
            'ten_gigabit_ip': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
            'management_ip': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
            'gigabit_port': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
            'ten_gigabit_port': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
            'cpu': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
            'mem': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
            'u_num': forms.TextInput(attrs={'style': 'width: 50px'}),
            'content': forms.TextInput(attrs={'style': 'width: 410px'}),

            'disk_capacity': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
            'disk_count': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
            'disk_raid': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
            'os_capacity': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
            'os_count': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
            'os_raid': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
            'ssd_capacity': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
            'ssd_count': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
            'ssd_raid': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),

            'asset': forms.Select(attrs={'style': 'height: 25px;width: 150px;'}),
            'cabinet': forms.Select(attrs={'style': 'height: 25px;width:50px;'}),
            'env': forms.Select(attrs={'style': 'height: 25px;width: 150px;'}),
            'region': forms.Select(attrs={'style': 'height: 25px;width: 150px;'}),
            'system': forms.Select(attrs={'style': 'height: 25px;width: 150px;'}),
            'tenant': forms.Select(attrs={'style': 'height: 25px;width: 150px;'}),
            'role': forms.Select(attrs={'style': 'height: 25px;width: 150px;'}),
        }

    def clean(self):
        if PhysicsServers.objects.filter(cabinet=self.cleaned_data['cabinet'],
                                         cabinet__dc__num=self.cleaned_data['chosecenter']):
            cleaned_data = super(ServerAddForm, self).clean()
        else:
            raise forms.ValidationError(u"机房机柜编号不匹配.")
        return cleaned_data

    def save(self, commit=True):
        cleaned_data = {k: v for k, v in self.cleaned_data.items() if v is not None}
        cleaned_data.pop('chosecenter')
        asset = cleaned_data['asset']
        asset.status = cleaned_data.pop('chosestatus')
        asset.save()
        return PhysicsServers.objects.create(**cleaned_data)

# class PhysicsServersForm(forms.ModelForm):
#     class Meta:
#         model = PhysicsServers
#         fields = "__all__"
#
#
# class NetworkUpdateForm(forms.ModelForm):
#     class Meta:
#         model = NetworkDevices
#         fields = "__all__"
#         widgets = {
#             'type': forms.Select(attrs={'style': 'width: 186px;height: 25px;'}),
#             'assets': forms.Select(attrs={'style': 'width: 186px;height: 25px;'}),
#             'cabinet': forms.Select(attrs={'style': 'width: 186px;height: 25px;'}),
#         }
#
#
# class ServerUpdateForm(forms.ModelForm):
#     class Meta:
#         model = PhysicsServers
#         fields = "__all__"
#         widgets = {
#             'system': forms.Select(attrs={'style': 'width: 175px;height: 25px;'}),
#             'env': forms.Select(attrs={'style': 'width: 175px;height: 25px;'}),
#             'asset': forms.Select(attrs={'style': 'width: 175px;height: 25px;'}),
#             'cabinet': forms.Select(attrs={'style': 'width: 175px;height: 25px;'}),
#             'tenant': forms.Select(attrs={'style': 'width: 175px;height: 25px;'}),
#             'region': forms.Select(attrs={'style': 'width: 175px;height: 25px;'}),
#             'role': forms.Select(attrs={'style': 'width: 175px;height: 25px;'}),
#
#             'hostname': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'u_num': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'cpu': forms.NumberInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'mem': forms.NumberInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'ten_gigabit_ip': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'gigabit_ip': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'management_ip': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'gigabit_port': forms.NumberInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'ten_gigabit_port': forms.NumberInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'disk_capacity': forms.NumberInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'disk_count': forms.NumberInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'disk_raid': forms.NumberInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'os_capacity': forms.NumberInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'os_count': forms.NumberInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'os_raid': forms.NumberInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'ssd_capacity': forms.NumberInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'ssd_count': forms.NumberInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'ssd_raid': forms.NumberInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'content': forms.TextInput(attrs={'style': 'margin-left: 70px;width: 600px;height: 30px'}),
#         }
#         # widgets = {}
#         # help_texts = {}
#
#
# class AssetUpdateForm(forms.ModelForm):
#     class Meta:
#         model = Assets
#         fields = "__all__"
#         widgets = {
#             'purchase': forms.Select(attrs={'style': 'width: 175px;height: 25px;'}),
#             'num': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'type': forms.Select(attrs={'style': 'width: 175px;height: 25px;'}),
#             'status': forms.Select(attrs={'style': 'width: 175px;height: 25px;'}),
#             'model': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'brand': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'serial': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'supplier': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'expired': forms.TextInput(attrs={'style': 'width: 150px;height: 25px'}),
#             'content': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#         }
#         # widgets = {}
#         # help_texts = {}
#
#
# class VirtualUpdateForm(forms.ModelForm):
#     class Meta:
#         model = VirtualServers
#         fields = "__all__"
#         widgets = {
#
#         }
#
#
# class PurchaseSearchForm(forms.ModelForm):
#     class Meta:
#         model = Purchase
#         fields = ['num', 'pr_type', 'pr_status', 'project_code', 'applicant']
#         widgets = {
#             'num': forms.TextInput(attrs={'id': 'search_num', 'name': 's_num', 'class': 'col-sm-6'}),
#             'pr_type': forms.Select(
#                 attrs={'id': 'search_pr_type', 'name': 's_pr_type', 'style': 'height: 38px;', 'class': 'col-sm-6'}),
#             'pr_status': forms.Select(
#                 attrs={'id': 'search_pr_status', 'name': 's_pr_status', 'style': 'height: 38px;', 'class': 'col-sm-6'}),
#             'project_code': forms.TextInput(
#                 attrs={'id': 'search_project_code', 'name': 's_project_code', 'class': 'col-sm-6'}),
#             'applicant': forms.TextInput(attrs={'id': 'search_applicant', 'name': 's_applicant', 'class': 'col-sm-6'}),
#         }
#
#
# class PurchaseCreateForm(forms.ModelForm, forms.Field):
#     class Meta:
#         model = Purchase
#         fields = "__all__"
#
#
# class PurchaseUpdateForm(forms.ModelForm):
#     class Meta:
#         model = Purchase
#         fields = "__all__"
#         widgets = {
#             'num': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'pr_type': forms.Select(
#                 attrs={'id': 'pr_type', 'name': 's_pr_type', 'style': 'width: 175px;height: 25px;',
#                        'class': 'col-sm-6'}),
#             'pr_status': forms.Select(
#                 attrs={'id': 'search_pr_status', 'name': 's_pr_status', 'style': 'width: 175px;height: 25px;',
#                        'class': 'col-sm-6'}),
#             'pr_name': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'pr_num': forms.NumberInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'pr_submitdate': forms.TextInput(attrs={'style': 'width: 150px;height: 25px;'}),
#             'project_code': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'project_name': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'applicant': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'expense': forms.NumberInput(attrs={'style': 'width: 175px;height: 25px;'}),
#             'receive': forms.TextInput(attrs={'style': 'width: 150px;height: 25px'}),
#             'content': forms.TextInput(attrs={'style': 'width: 175px;height: 25px;'}),
#         }
#         # widgets = {}
#         # help_texts = {}
#
#
# class AssetSearchForm(forms.ModelForm):
#     class Meta:
#         model = Assets
#         fields = ['purchase', 'num', 'type', 'model', 'serial', 'supplier', 'status', 'brand', 'expired']
#         widgets = {
#             'purchase': forms.Select(
#                 attrs={'id': 'search_purchase', 'name': 's_purchase', 'style': 'height: 38px;', 'class': 'col-sm-6'}),
#             'num': forms.TextInput(attrs={'id': 'search_num', 'name': 's_num', 'class': 'col-sm-6'}),
#             'type': forms.Select(
#                 attrs={'id': 'search_type', 'name': 's_type', 'style': 'height: 38px;', 'class': 'col-sm-6'}),
#             'model': forms.TextInput(attrs={'id': 'search_model', 'name': 's_model', 'class': 'col-sm-6'}),
#             'serial': forms.TextInput(attrs={'id': 'search_serial', 'name': 's_serial', 'class': 'col-sm-6'}),
#             'supplier': forms.TextInput(attrs={'id': 'search_supplier', 'name': 's_supplier', 'class': 'col-sm-6'}),
#             'status': forms.Select(
#                 attrs={'id': 'search_status', 'name': 's_status', 'style': 'height: 38px;', 'class': 'col-sm-6'}),
#             'brand': forms.TextInput(attrs={'id': 'search_brand', 'name': 's_brand', 'class': 'col-sm-6'}),
#             'expired': forms.DateTimeInput(attrs={'id': 'search_expired', 'name': 's_expired', 'class': 'col-sm-6'}),
#         }
#
#
# class AssetCreateForm(forms.ModelForm):
#     class Meta:
#         model = Assets
#         fields = "__all__"
#         widgets = {
#             'purchase': forms.Select(attrs={'style': 'width: 300px;height: 25px'}),
#             'num': forms.TextInput(attrs={'placeholder': '资产编号', 'style': 'width: 300px;height: 25px'}),
#             'type': forms.Select(attrs={'style': 'width: 300px;height: 25px'}),
#             'model': forms.TextInput(attrs={'style': 'width: 300px'}),
#             'brand': forms.TextInput(attrs={'style': 'width: 300px'}),
#             'serial': forms.TextInput(attrs={'style': 'width: 300px'}),
#             'supplier': forms.TextInput(attrs={'style': 'width: 300px'}),
#             'status': forms.Select(attrs={'style': 'width: 300px;height: 25px'}),
#             'content': forms.TextInput(attrs={'style': 'width: 300px'}),
#         }
#         help_texts = {
#             'purchase': _('对应的采购编号(PR).'),
#         }
#
#
# class NetworkSearchForm(forms.ModelForm):
#     class Meta:
#         model = NetworkDevices
#         fields = ['cabinet']
#         widgets = {
#             'cabinet': forms.Select(
#                 attrs={'style': 'height: 38px;', 'class': 'col-sm-6', 'id': 'search_cabinet'}),
#         }
#
#
# class ServerSearchForm(forms.ModelForm):
#     class Meta:
#         model = PhysicsServers
#         fields = ['ip', 'hostname', 'tenant', 'region', 'cabinet', 'asset']
#         widgets = {
#             'ip': forms.TextInput(attrs={'class': 'col-sm-6', 'id': 'search_ip', 'name': 's_ip'}),
#             'hostname': forms.TextInput(attrs={'class': 'col-sm-6', 'id': 'search_hostname', 'name': 's_hostname'}),
#             # 'tenant': forms.Select(
#             #     attrs={'style': 'height: 38px;', 'class': 'col-sm-6', 'id': 'search_tenant', 'name': 's_tenant'}),
#             'region': forms.Select(
#                 attrs={'style': 'height: 38px;', 'class': 'col-sm-6', 'id': 'search_region', 'name': 's_region'}),
#             'cabinet': forms.Select(
#                 attrs={'style': 'height: 38px;', 'class': 'col-sm-6', 'id': 'search_cabinet', 'name': 's_cabinet'}),
#             'asset': forms.Select(
#                 attrs={'style': 'height: 38px;', 'class': 'col-sm-6', 'id': 'search_asset', 'name': 's_asset'}),
#         }
#         # help_texts = {
#         #
#         # }
#         # error_messages = {
#         #
#         # }
#
#
# class NetworkAddForm(forms.ModelForm):
#     chosecenter = forms.ChoiceField(choices=[('JQ', 'JQ'), ('PK', 'PK')],
#                                     required=True,
#                                     help_text='* required',
#                                     widget=forms.Select(
#                                         attrs={
#                                             'style': 'height: 25px;width: 150px;'
#                                         }))
#     chosestatus = forms.ChoiceField(choices=[('线上已分配', '线上已分配'), ('线上未分配', '线上未分配')],
#                                     required=True,
#                                     help_text='* required',
#                                     widget=forms.Select(
#                                         attrs={
#                                             'style': 'height: 25px;width: 150px;'
#                                         }))
#
#     class Meta:
#         model = NetworkDevices
#         fields = "__all__"
#
#         widgets = {
#             'hostname': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
#             'ip': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
#             'ports': forms.NumberInput(attrs={'style': 'height: 25px;width: 150px;'}),
#             'unit': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
#             'firmware': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
#             'owner': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
#
#             'cabinet': forms.Select(attrs={'style': 'height: 25px;width: 150px;'}),
#             'type': forms.Select(attrs={'style': 'height: 25px;width: 150px;'}),
#             'assets': forms.Select(attrs={'style': 'height: 25px;width: 150px;'}),
#         }
#
#     def save(self, commit=True):
#         cleaned_data = {k: v for k, v in self.cleaned_data.items() if v is not None}
#         cleaned_data.pop('chosecenter')
#         asset = cleaned_data['assets']
#         asset.status = cleaned_data.pop('chosestatus')
#         asset.save()
#         return NetworkDevices.objects.create(**cleaned_data)
#
#
# class VirtualSearchForm(forms.ModelForm):
#     region = forms.ModelChoiceField(
#         queryset=Region.objects.all(),
#         required=False,
#         widget=forms.Select(attrs={'style': 'height: 38px;width: 100px;', 'id': 'search_region'}))
#
#     class Meta:
#         model = VirtualServers
#         fields = "__all__"
#         widgets = {
#             'physical': forms.Select(attrs={'style': 'height: 38px;width: 100px;', 'id': 'search_physical'}),
#             'tenant': forms.Select(attrs={'style': 'height: 38px;width: 100px;', 'id': 'search_tenant'})
#         }
#
#
# class ServerAddForm(forms.ModelForm):
#     chosestatus = forms.ChoiceField(choices=[('线上已分配', '线上已分配'), ('线上未分配', '线上未分配')],
#                                     required=True,
#                                     help_text='* required',
#                                     widget=forms.Select(
#                                         attrs={
#                                             'style': 'height: 25px;width: 150px;'
#                                         }))
#     chosecenter = forms.ChoiceField(choices=[('JQ', 'JQ'), ('PK', 'PK')],
#                                     required=True,
#                                     help_text='* required',
#                                     widget=forms.Select(
#                                         attrs={
#                                             'style': 'height: 25px;width: 150px;'
#                                         }))
#
#     class Meta:
#         model = PhysicsServers
#         fields = "__all__"
#
#         widgets = {
#             'hostname': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
#             'gigabit_ip': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
#             'ten_gigabit_ip': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
#             'management_ip': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
#             'gigabit_port': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
#             'ten_gigabit_port': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
#             'cpu': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
#             'mem': forms.TextInput(attrs={'style': 'height: 25px;width: 150px;'}),
#             'u_num': forms.TextInput(attrs={'style': 'width: 50px'}),
#             'content': forms.TextInput(attrs={'style': 'width: 410px'}),
#
#             'disk_capacity': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
#             'disk_count': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
#             'disk_raid': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
#             'os_capacity': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
#             'os_count': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
#             'os_raid': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
#             'ssd_capacity': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
#             'ssd_count': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
#             'ssd_raid': forms.TextInput(attrs={'style': 'width: 70px;height: 25px'}),
#
#             'asset': forms.Select(attrs={'style': 'height: 25px;width: 150px;'}),
#             'cabinet': forms.Select(attrs={'style': 'height: 25px;width:50px;'}),
#             'env': forms.Select(attrs={'style': 'height: 25px;width: 150px;'}),
#             'region': forms.Select(attrs={'style': 'height: 25px;width: 150px;'}),
#             'system': forms.Select(attrs={'style': 'height: 25px;width: 150px;'}),
#             'tenant': forms.Select(attrs={'style': 'height: 25px;width: 150px;'}),
#             'role': forms.Select(attrs={'style': 'height: 25px;width: 150px;'}),
#         }
#
#     def clean(self):
#         if PhysicsServers.objects.filter(cabinet=self.cleaned_data['cabinet'],
#                                          cabinet__dc__num=self.cleaned_data['chosecenter']):
#             cleaned_data = super(ServerAddForm, self).clean()
#         else:
#             raise forms.ValidationError(u"机房机柜编号不匹配.")
#         return cleaned_data
#
#     def save(self, commit=True):
#         cleaned_data = {k: v for k, v in self.cleaned_data.items() if v is not None}
#         cleaned_data.pop('chosecenter')
#         asset = cleaned_data['asset']
#         asset.status = cleaned_data.pop('chosestatus')
#         asset.save()
#         return PhysicsServers.objects.create(**cleaned_data)
#
# # # -*- coding: utf-8 -*-
# # from django import forms
# # from .models import DutyTab
# #
# #
# # class Calendaradd(forms.ModelForm):
# #     class Meta:
# #         model = DutyTab
# #         fields = [
# #             'duty_name',
# #             'duty_timef',
# #             'segm',
# #         ]
# #         # exclude = ['event_id', 'status', 'closed_by']
# #
# #     def save(self, commit=True, cha_id=None, fname=None):
# #         if self.errors:
# #             raise ValueError(
# #                 "The %s could not be %s because the data didn't validate." % (
# #                     self.instance._meta.object_name,
# #                     'created' if self.instance._state.adding else 'changed',
# #                 )
# #             )
# #         if commit:
# #             self.instance.cha_id = cha_id
# #             self.instance.file_name = fname
# #             self.instance.save()
# #             self._save_m2m()
# #         else:
# #             self.save_m2m = self._save_m2m
# #         return self.instance
# #
# # class Calendarcha(forms.ModelForm):
# #     class Meta:
# #         model = DutyTab
# #         fields = [
# #             'duty_name',
# #             'duty_timef',
# #             'segm',
# #             'duty_even',
# #         ]
# #         # exclude = ['event_id', 'status', 'closed_by']
# #
