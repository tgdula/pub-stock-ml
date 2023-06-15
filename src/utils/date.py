import datetime as dt

date_format = "%Y-%m-%d"

one_month = 21
one_year = 12 * one_month

def adjust_days(date_string:str, days:int) -> str:
    date = dt.datetime.strptime(date_string, date_format)
    adjusted = date+dt.timedelta(days=days)
    return adjusted.strftime(date_format)

def to_date(date_string:str) -> dt.datetime:
    return dt.datetime.strptime(date_string, date_format)