import bs4 as bs
import urllib.request
import pandas as pd

import io 

source = urllib.request.urlopen('https://www.stats.ox.ac.uk/pub/PRNN/synth.tr').read()
soup = bs.BeautifulSoup(source,'lxml')

table_train = soup.find_all('p')
df_train = pd.read_csv(io.StringIO(table_train[0].string))

source2 = urllib.request.urlopen('https://www.stats.ox.ac.uk/pub/PRNN/synth.te').read()
soup2 = bs.BeautifulSoup(source2,'lxml')

table_test = soup2.find_all('p')
df_test = pd.read_csv(io.StringIO(table_test[0].string))

