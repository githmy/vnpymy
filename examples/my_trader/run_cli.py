import multiprocessing
from time import sleep
from datetime import datetime, time
from logging import INFO

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine

from vnpy.gateway.ctp import CtpGateway
from vnpy.gateway.xtp import XtpGateway
from vnpy.app.cta_strategy import CtaStrategyApp
from vnpy.app.cta_strategy.base import EVENT_CTA_LOG
from vnpy.app.risk_manager import RiskManagerApp
import json

SETTINGS["log.active"] = True
SETTINGS["log.level"] = INFO
SETTINGS["log.console"] = True


# 国内期货CTP SimNow
ctp_setting = {
    "用户名": "53191002042",
    "密码": "Sbiwh1Po",
    "经纪商代码": 9999,
    "交易服务器": "120.27.164.69:6001",
    "行情服务器": "120.27.164.138:6002",
    "产品名称": "simnow_client_test",
    "授权编码": "0000000000000000",
    "产品信息": ""
}


def run_child():
    """
    Running in the child process.
    """
    SETTINGS["log.file"] = True

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    # main_engine.add_gateway(CtpGateway)
    main_engine.add_gateway(XtpGateway, gateway_name='Xtp_test')

    print(58, main_engine.gateways)
    print(main_engine.get_all_gateway_names())
    print(main_engine.get_all_gateway_status())
    for gateway_name in main_engine.gateways:
        xtp_setting = json.load(open(f"./connect_{gateway_name}.json", "r", encoding="utf8"))
        main_engine.connect(xtp_setting, gateway_name)
    sleep(2)
    print(main_engine.get_all_gateway_status())
    print(59)
    cta_engine = main_engine.add_app(CtaStrategyApp)
    main_engine.add_app(RiskManagerApp)
    main_engine.write_log("主引擎创建成功")

    log_engine = main_engine.get_engine("log")
    event_engine.register(EVENT_CTA_LOG, log_engine.process_log_event)
    main_engine.write_log("注册日志事件监听")

    # main_engine.connect(ctp_setting, "CTP")
    main_engine.write_log("连接CTP接口")

    sleep(10)

    cta_engine.init_engine()
    main_engine.write_log("CTA策略初始化完成")

    cta_engine.init_all_strategies()
    sleep(60)  # Leave enough time to complete strategy initialization
    main_engine.write_log("CTA策略全部初始化")

    cta_engine.start_all_strategies()
    main_engine.write_log("CTA策略全部启动")

    while True:
        sleep(1)


def run_parent():
    """
    Running in the parent process.
    """
    print("启动CTA策略守护父进程")

    # Chinese futures market trading period (day/night)
    DAY_START = time(8, 45)
    DAY_END = time(15, 30)

    NIGHT_START = time(20, 45)
    NIGHT_END = time(2, 45)

    child_process = None

    while True:
        current_time = datetime.now().time()
        trading = False

        # Check whether in trading period
        if (
                        (current_time >= DAY_START and current_time <= DAY_END)
                    or (current_time >= NIGHT_START)
                or (current_time <= NIGHT_END)
        ):
            trading = True

        # Start child process in trading period
        if trading and child_process is None:
            print("启动子进程")
            child_process = multiprocessing.Process(target=run_child)
            child_process.start()
            print("子进程启动成功")

        # 非记录时间则退出子进程
        if not trading and child_process is not None:
            print("关闭子进程")
            child_process.terminate()
            child_process.join()
            child_process = None
            print("子进程关闭成功")

        sleep(5)


if __name__ == "__main__":
    run_parent()
