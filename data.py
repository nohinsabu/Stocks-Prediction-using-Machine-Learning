# import external pandas_datareader library with alias of web
import pandas_datareader as web

# import datetime internal datetime module
# datetime is a Python module
import datetime

# datetime.datetime is a data type within the datetime module
# which allows access to Gregorian dates and today function

# datetime.date is another data type within the datetime module
# which permits arithmetic with Gregorian date components

# definition of end with datetime.datetime type must precede
# definition of start with datetime.date type

end = datetime.datetime.today()
start = datetime.date(end.year-2, 1, 1)

# DataReader method name is case sensitive

df = web.DataReader("nvda", 'yahoo', start, end)

# invoke to_csv for df dataframe object from
# DataReader method in the pandas_datareader library

# ..\second_yahoo_prices_to_csv_demo.csv must not
# be open in another app, such as Excel

# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html
# specification does not work as documented,
# but these workarounds are ok to use
# notice use of forward slashes, instead of backward slashes, for path_out

path_out = 'c:/python_programs_output/'
df.to_csv(path_out+'second_yahoo_prices_volumes_to_csv_demo.csv')
