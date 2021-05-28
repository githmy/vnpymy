import math


def option_pricing(price_now, price_force, rise_up, rise_dn, rate, t):
    """
    根据当前price_now，执行价格，涨幅, 跌幅，利率，时间差 定出期权价格
    a = rise_up - 1 
    b = 1 - rise_dn
    proba_up / proba_dn = a / b
    price_expection_future = price_now * math.e(rate * t)
    price_expection_future = (proba_up * rise_up + proba_dn * rise_dn) * price_now
    variance_future = (proba_up * rise_up ** 2 + proba_dn * rise_dn ** 2) * price_now ** 2 - price_expection_future ** 2
    
    proba_up = (math.e(rate * t)-rise_dn)/(rise_up-rise_dn)  
    """
    proba_up = (math.e(rate * t) - rise_dn) / (rise_up - rise_dn)
    proba_dn = 1 - proba_up
    price_up = price_now * rise_up
    price_dn = price_now * rise_dn
    # 截止日之前，跌多了可以高卖
    fall_american = False
    # 只有截止日，跌多了可以高卖
    fall_europe = False
    # 截止日之前，涨高了可以低买
    rise_american = False
    # 只有截止日，涨高了可以低买
    rise_europe = True
    if rise_europe:
        profit_high = price_up - price_force if price_up - price_force > 0 else 0
        profit_low = price_dn - price_force if price_dn - price_force > 0 else 0
        option4rise_europe = (proba_up * profit_high + proba_dn * profit_low) * math.e(rate * t)
        return option4rise_europe
    elif rise_american:
        option4rise_american = None
        return option4rise_american
    elif fall_american:
        profit_high = price_force - price_up if price_force - price_up > 0 else 0
        profit_low = price_force - price_dn if price_force - price_dn > 0 else 0
        option4fall_american = (proba_up * profit_high + proba_dn * profit_low) * math.e(rate * t)
        return option4fall_american
    elif fall_europe:
        option4fall_europe = None
        return option4fall_europe


def main():
    pass


if __name__ == '__main__':
    main()
