import pandas as pd
from ydata_profiling import ProfileReport

data = pd.read_csv("heart.csv")
report = ProfileReport(data)
report.to_file("heart.html")