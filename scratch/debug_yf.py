import yfinance as yf
t = yf.Ticker("BBCA.JK")
divs = t.dividends
print("Type:", type(divs))
print("Index:", divs.index)
print("Series head:\n", divs.head())
