from vnpy.trader.rqdata import rqdata_client
from vnpy.trader.tudata import tusharedata
from vnpy.trader.database import database_manager
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest
from datetime import datetime, time
from time import sleep
import tushare as ts
from vnpy.gateway.xtp import XtpGateway

def load_data(req):
    " 数据下载 "
    # data = rqdata_client.query_history(req)
    data = tusharedata.query_history(req)
    database_manager.save_bar_data(data)
    print(f"{req.symbol}历史数据下载完成")


def download_data():
    "定时下载任务"
    # 设置日期
    start_date = datetime(2005, 1, 1)
    end_date = datetime(2020, 9, 10)

    # 设置品种
    symbols = {
        "SHFE": ["cu", "al", "zn", "pb", "ni", "sn", "au", "ag", "rb", "wr", "hc", "ss", "bu", "ru", "nr", "sp", "sc",
                 "lu",
                 "fu"],
        "DCE": ["c", "cs", "a", "b", "m", "y", "p", "fb", "bb", "jd", "rr", "l", "v", "pp", "j", "jm", "i", "eg", "eb",
                "pg"],
        "CZCE": ["SR", "CF", "CY", "PM", "WH", "RI", "LR", "AP", "JR", "OI", "RS", "RM", "TA", "MA", "FG", "SF", "ZC",
                 "SM",
                 "UR", "SA", "CL"],
        "CFFEX": ["IH", "IC", "IF", "TF", "T", "TS"]
    }

    symbol_type = "99"

    # 批量下载
    for exchange, symbols_list in symbols.items():
        for s in symbols_list:
            # 先查询最新的，再拉取
            req = HistoryRequest(
                symbol=s + symbol_type,
                exchange=Exchange(exchange),
                start=start_date,
                interval=Interval.DAILY,
                end=end_date,
            )
            load_data(req)


def time_misson():
    " 定时任务 每天下午5点更新指定数据"
    last_dt = datetime.now()
    start_time = time(17, 0)

    while True:
        dt = datetime.now()

        if dt.time() > start_time and last_dt.time <= start_time:
            download_data()
        else:
            sleep(60)
        last_dt = dt


if __name__ == '__main__':
    tt = ts.get_industry_classified()
    print(tt)
    download_data()
    # time_misson()
