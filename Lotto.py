import numpy as np
import pandas as pd
import requests
import sys
import bs4
import re
import urllib.request

search_url = "https://dhlottery.co.kr/gameResult.do?method=byWin&drwNo={page}"


def one_lotto_number(page):
    response = urllib.request.urlopen(search_url.format(page=page))
    lotto_data = response.read()

    soup = bs4.BeautifulSoup(lotto_data)
    ret = []
    newret = []
    for winnums in soup.findAll('div', attrs={'class': 'nums'}):
        winnum = winnums.findAll('span')
        ret.append(winnum)
    ret = ret[0]
    for i in ret:
        string = str(i)
        onlynum = re.sub('<.+?>', '', string, 0, re.I | re.S)
        newret.append(onlynum)
        newret = list(map(int, newret))
    return newret


one_lotto_number(892)

lotto = []

for i in range(892):
    lotto_num = one_lotto_number(i + 1)
    lotto.append(lotto_num)

lotto.head()