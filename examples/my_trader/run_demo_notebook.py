#!/usr/bin/env python
# coding: utf-8

# 该Jupyter Notebook用于展示如何使用ScriptTraderApp模块，在CLI命令行下进行交易指令的调用


from vnpy.app.script_trader import init_cli_trading
from vnpy.gateway.oes import OesGateway


# 连接到服务器
setting = {
    "td_ord_server": "tcp://106.15.58.119:6101",
    "td_rpt_server": "tcp://106.15.58.119:6301",
    "td_qry_server": "tcp://106.15.58.119:6401",
    "md_tcp_server": "tcp://139.196.228.232:5103",
    "md_qry_server": "tcp://139.196.228.232:5203",
    "username": "",
    "password": "",
    "hdd_serial": "",
    "customize_ip": "",
    "customize_mac": ""
}

engine = init_cli_trading([OesGateway])
engine.connect_gateway(setting, "OES")

# 查询所有合约
engine.get_all_contracts(use_df=True)

# 查询资金
engine.get_all_accounts(use_df=True)

# 查询持仓
engine.get_all_positions(use_df=True)

# 查询活动委托
engine.get_all_active_orders(use_df=True)

# 订阅行情
engine.subscribe(["600036.SSE"])

# 查询行情
engine.get_tick("600036.SSE", use_df=True)

# 委托下单
vt_orderid = engine.buy("600036.SSE", 32, 1000)
print(789)
print(vt_orderid)

# 查询特定委托
engine.get_order(vt_orderid)

# 委托撤单
engine.cancel_order(vt_orderid)
