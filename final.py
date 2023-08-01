# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:50:38 2023

@author: yije
"""

#洪宏吉 D0869492 資訊四乙
#馬宇哲 D0809311 資訊四丙
#洪正軒 D0827022 資訊四乙

#homework 1
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

import pandas as pd
import requests as req
import json    
import matplotlib.pyplot as plt
import numpy as np
import re
import warnings

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import linear_model   
from sklearn import svm   # Support Vector Regression
from sklearn import tree  # Decision Trees Regression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.cluster import KMeans
#-------------------------------------------------------------------------------

# 台中市八⼤⾏政區,順序為：東區、西區、南區、北區、中區、西屯區、北屯區、南屯區
# 101/8 ~ 111/12, 查詢類型：「房地」「房地(車)」
district=['East','West','South','North','Central','Xitun','Beitun','Nantun']
url_0='https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/158322a89004157f8d01338cdd650692?q=VTJGc2RHVmtYMThBOEU2bGk4dDQxc3pWSnJldmVmdVhIZml5U1Y5UkJGM2cyNGdXSHhzNTR6L2UrdVYzVUc3QklsRlhhWFZtWkJLN3M5N3Z0RWNuSjM3UnBMQnhUS3p3Si9vSUtDVFZjR040RHE3N3FJL2hSWVpqVE9vSXJROWp6U2JHODRnbkdjZStUcXN1bnNoc0d2SFNYMzRBc1J6NWZmL2hxaTV6T3o1ZXJkWElnUkdvRm90eEpsNXV0L0ZocFREYk9vM1YrSTRzcUd2NzVEVm13WjBXMkx2ZDFQMFRmNjlQeGxxMythTE9kQnBGdmdIQjNSa1lBSHhCTnpqYUJlQXc4UzA1Q1EzQkVsUkw3Rml0SGo4a2w5Qm8ySi9KMkdJM1IrY0ptZWNSTmwwMTNIVGFOcitiZXh2VVpvSDRzeFhRaGdPYVJuUm50OEdrT2VvSnZHU3M5Mm9PQ2Z2ei9HK3lBTlJ6eVRqUEF3VENPYnhGelRtWGJFU3ljVDd4RlJTMnBtYmZKS3VteHRLbncyRFRybS9TZmNLQjNZY09Cem1iR0ZZSytUdVNBL01vQm1Nb1V5TjNPV2V2TkhnR3BiV3JLTFlDRjFwVFVNRkNjZm9XTUNQRXljeEZMS295aVR0SUJuSFNoY1ZnWWt3WGdWdTlwakM1aElXTC93MTdqZVppV05CUWVZd1hKUllXTGhSU3pnSmd6SUY2RlROcVhyMytmU20yeVl1aDNJK3V2YWlHWUpJWGptNVM2OVErWCtjTmlTdldIMmYrRUFVUkR4bGVUT2VWL2wrVHVuZlZYQkNVbFhtdS9wVi92Yit6ek5qUFVVY1kyTmg3MWRRK1NFWUZmVjV0ZTM5YVpmOXUvUjNCNzMzQURWYkxvZnE3SklrUFl2aEh2bUk9'
url_1='https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/3bb120a05814f579d734aa15b8fe1b2e?q=VTJGc2RHVmtYMThaWlg5bEFIUTQ3MTRISVdBZzh2eXFkWjJQTGZ2TXYzaWRvS3NQbTJjdEJJRGczWnRQQkZQTkpmYlVsczBBWXZXWTJsczhkZUppVFUyQ3VNOG15M054SlZlUTJiZnFaMjJDODZ3TlNvdGsxOEJaRnl3UXp0OHRzMTVNRlo2a2RqSWs4UEdzRUdpeFJ4S1ZWeXdPS0MrSW9zS3E1SERkQlVXdEtJK094Y3B0eTlrUjJXeGpSM1M0Q2NoSWhaeFZDUmdic2dkdThQcHUwSzVCbjRhWWw3bmxLSDB4Qnpoc3lwMUl6c1pyL3RBd0lmeEU1WjJrcEtpZVc4N1ovYUtVZHJvRUpBa0k3NUxKRlVSMEhkZGFZUE5FY3lKNnJtQzJjNU5MZk9ydHk4dVZSZWpyZFdVNzVZSkFUSTkvWm9kUzVIOXVrSXBaV2NvVGt5bThxS0lUd3BhMkdGWkFHYnpncXltbWpVWTh2cVdCaUh2TExkdlk3b3ZKbTh2SWlDWVNjUVVGUGt6OXFYY0xxVHd4Ync3cFAwWGZ3UmRkemxnUFJzUnF6RHZac0VZWkJVMHBaMWovTHhXSUIxa1A5ZjE2WTc5R3dnekRQOVR2b292OUowWHhyMWNiblloTzNJNkxMREVwWDZjKzNNSnhUOU5JSnEvMm1jM2drSER4NTQ2cDI4N2NCNkY5b3RZQ0lpNVErY29jYm5oVlQ1TTJpazNlb2VhMTQ3UzVHejFEdGdFbGs0b2ZQTFNtZHdEYTZiMTFCVDdpVGxaRjZZTWdHRXMrcDVpd0tmTllBYXNmODBRdkhFNXJtTGtzTjljNGJMek5aSFdmaU01bUhmMll3eDRVRGowRzNOa3NRNzk2UGR3K013dEpoNGx5QitPeis5KzBBZU09'
url_2='https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/7fa51cfb89eb44630f512a53319a674a?q=VTJGc2RHVmtYMSs0NkdlMXZZd1p3b0JHWDQvbHpBMkowRVg5d3lnclpMYVNFRG1YV0VINVVuWVJrc3AxT1BkS3F0UkVEOXBBMGxRUWJEVmhKVmZpMHl5UUEwYXF1N0FDRkNzRXR0WGp0ZzR6cVlxMmFJMjZuczJEd2IvRmdteUZnUFlRQm1jMkF6Q0pjQ1F1bGdCOHcweWdvZ1R1SlN0VXhGeGFBUDZCZG1CNkdIMVVxK2ZWT0VpWTJRQVd0YmpNVWtHNEpDRHJBeTB5SkFnUEF2QzdJTXRvK0ZWVWs1ZlJzVkhiRVVka21BZ2sxWGJVWFBZVkVzMWl4VlgreHBLb3UxaThaWTdpT2lpK3RTc2R1VWF4cFdSUlllNDJtVHFRVm1HQlV6S2tDOE5JR3k2ejJPbXBhUS9zWE55MWNtbTVLVEFqd0JXdFJCckxxckhOcjB1SVVxZzJmN1NvMWgvUWNJN1hwU1dDM0tOblltcytLRmhZQnRNYkgrQWE1eDROVW9ONUt4T1NCRlVmT0NUTlFmcjM0azkvc2VFK2JaRy9Wdk1kM1Q4Q3JUNTNsNjZaZDZJUkUreGtiekhBU2tqZ25hbXJYOVRTTzc0TmxKWUdEc0s0WG1aekVZem9JWStQN0oyUmY1SStIaDRxTDNSb0xLNVVaNDJ6T2RkdXdVK1JhSXhlZ1U3SVAwU0kyb2dVbVptR2pLODdhNUQvTm1EcmxvOHJCT2ZZcktDdFV3YXV0UUhiN01CTG5teVltMHQzczlRYnlzczR0elkyQmtHS2E1V0NGazBxSzRWV29FOENxQlhaQUYvU1hoNUNzUnJ0UkNXZkxZblFtMjN4THBqdm9oU1hHRXFlRW5ONVdhUUk2c0ZhWURUMGcrMEpPdWpRQkRCUGw3S01wRlk9'
url_3='https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/85e0b737708b95fe686b3213809bc9df?q=VTJGc2RHVmtYMSsxYlZHMGdiTTNjNXpEN2VGM2crTWF3aWxLemNFRUZ1TThZOWJCY3gyeVl5MkdVckdnWjRYSGpPMnZvWDZvM1JOeWQxVGYrQUF3VHhsSnVod0Myd1N5cmdrZHJFMEp4cUdhckRvUzZHcGlxRVdDdVFzUzNrRDUrU3BIcVlmbzhhZGxxeWN1YVNVL1BHaFFvMXo2a2VSN2p5VDNXbk9pa040K0ZRa2dKdzNSaXl6ajZBZllRdWZtOU43NzRmWHJnSEJoZ3RpQ0YrcFJNSDRTOGViK01NN2pVc0NOUUYrSHhXN2NCWG9zNEZUSy8wSWdnYXdHOHdnSDdNTkdEb3V0ZDVKNGxSc3F5Ylg4RDFQQ29jZ0tDay9TM2c2eXZQeGt5eDhMbjZXQ01PU1RZZ1hra3l3ZXdBSU0xakZrSFdlZUZlY0xZdXNqV2tMTnNGclNLK0ZabXJyc1kxcXdHdTBwMGlralpXZjZ5VGZJQzFHbUxibElLMkhaNTJRaWNCK1laM3lJMVpMeFpNRy9nc200ckpZaWhxZnRCL2ZNaWRPZC9raDRtNy9VdVk2SnlkYzZjcGczdG5lZnNxa2hEM1VJek1OaDNaUzRsR2RmamZjd2xSVEEwdHdVZmYyRVZTR3piaXk3VnNTLytKOUhta2NTYVg1NkxEZFdSTy9VR1UxVUFzbElvNG1QcTZRZk1nOTB1aE5TNUduTm9NdXk5SHpMa1ZsOElNNjcvN1dTZk9CZ1VhU3Z4K1YzdnlTK2lEVWduSUJMM0VRV1ZIYlhOa0Q4bUhTYlh4MmF3N1RiWldKRU1LZ0lKZFdidnhpQ0xQTTJ2b0Z3UTAwUWZDZXlJWndRV0RBVC84S3hWTlhveTNBaVhXTVZwZWVkU1A1WTJ6Tjh5dkU9'
url_4='https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/42467d53f97beb936e31230ee1140c36?q=VTJGc2RHVmtYMTlnNStqVFJMV3RXUWt1K0hKclRyNzkyQTdPczJleGJhcCtOTWxQWk1IaXRlMzk0cEdHWGQraDBZSDhFVU5lZ2hzUWsvTlQ0VWF2WS9FOTVTbk1rTjFPSUdDYlQ2N2dmSkJqNnBwMzRUYkhBdkl3QlFKZWJtais4ZmZOVDk3cjdwTm1CM0RsUHFRclFsMTBNcTNjRyt3SkV4bkdPNjZncUxBNU4yMGFQb2hWNVE4UzNjN1ZNYkhxWXV2bTJaTm9sUXdFMkVXV3BuT044eDJQa3d3SzZPSi91WkVFU3ZHelVBbzJHaCs0cHl1Z2VBUEppclVZakh5dW4rb2JUalAraTZCVUhzWnh5WWsyRkRUNG9JVWFLM25JL3dJU1ZNdExJT05manVqSFRPL3NJRXhrZTVWS1F4OEgzZUpvdVVwWGMyc0NkdG10TktQTk5GOUNhTG9VUW42OWRPSWdENVJWc2JtRStQS29vaGN2UE9kbHUwSFY3a3NTMzAzRlhYMm8zSTRwR2x5YXlsV003S09EUFVOUFFucFdJdjF2SGhIem8xVXhXL1VJSXorZk5iUGRjTU5sMWsxbXhROWdrRVQ3Yk1CYVVqY2lOVXBGUmRkNnN5Mjh0SUZ3Nk5oTktpdWh6akJRa09OU2RJQmw2N2pOdSt0MlF4U3lnMWozRUx5R0hwQ1ozbHFCMlJyeW1lTzE1MDVXc05hVG9xWU9yeERQZHg4R1Nnak9BMVR1VnFyTUdLck9UOGdlSFUzYVJEV0ZOQjRwUG1DZkxsZUU1Z2FlMlFpTWo0bGtFeDFoQ1o1TmZvUEVIL2x2SXc2eEl6U1NwV25tQjFUamNDbkt4UXNldkFxRW5iTWpJNGQ1bU51YXcrY1NjdER6ZjFpWHNmbzdZQjA9'
url_5='https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/d3bf86e16d95ea9fb3a076b5422f1f0f?q=VTJGc2RHVmtYMS9SYnI2eTd1UE1CbXBaN016THhnZy90c0pKd0JpRVZnOEc5dDNnZTQ2UmVCZGs3QkNhUEhPa29Tak9xV0dzN0NHT0FxNmI0RDB1T0wvSWk3aDZyZVFvWkh3SFFRanlhWTFiQXl4MWhvMDNoMEtvbkExcG1CTVZGUjg5dHFKRTdROHdpbFIxRHE5VStqV1ZSeG1LR1RjY1RpRUVzVHp4LzN1NGtJTWRCS1FPakZNOStNbDVtdnlxZ3I5Y0VUK3FKZ1FSdCtZWm9ZdUVNRW5XYnFhWVRZYzI1b1ZKSElLcEhOZWtydyt2U1RjWmtIUXV6TWQwRC9nTVYxdmdVRkRZd3h3VFd6cmRKZHhKRXRJQzMreHlVa1I3NTlUdm00aEJRU2xRUTk5OTZnSlBqOUdRQnczWVg4eWZhTUZvSFJad3YzejUwa1RHT2VUTTA0WGVqbHp1NWFrQTlONFdKc21jc2FWdUsxVVBtRDlqWVJYUzA0b3pGakdqVURCblVtSVdTaG4wMm00VVRGS2lGekFHSUxPNGYvMjBlTE1KSmxMVVZBdk93RHlUVjN5bWVBcFptOEVJZ0VNME1uNXpHaE0yeS9Za0NZbkhkWU5nTWRGRkgrYlVSMThxSTFIYUg1bG5ieXp2WDFIRTE2a0JSQkpuWEw3M1lZMFFiZlVQUCtwaHAwY3Q2N3NqNW56eHVZVEFVV1hYeGpyWVdiTVFCdkNQSEZOM1I0YWJuWDJFc1piNEdUeFFtaVIraEVuSmZjU1dhV0F2Z282M0dEWHdRSW9XZzBqR0ZaQzRZdVlobWNrU2dNM1EzcENMRzZ3KzJhRzExcVJkc3AxaE5QYi9JUWFqa1ZWRkRSVjVwVDJ4V3ZZSzJzQWZmRVJSZFZwVkQxaU1Hemdtb3dXWUZUR0FlV2xvY2FWbW1ra3M='
url_6='https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/a7593443a4004dd8a2df660bcf9ad7b0?q=VTJGc2RHVmtYMS9KUXpNcVBTLzZkay9sSG1SQ3JSQVpESzJFZHoxdUNwRXZkU2lWenZNOGNydWRCLzBjaDY2WG1Rdk9vdWRMbU93cnhtbUhHbGUrUVloM0VybzhXWmxWRFNpQU9Vem1jR0FiS1paVmxyV2xSaEQ0U3hJTnd5amUrK0g2clhCd2M3RG9QYjEzRjBaRzZEdkVkcWV2d3pLU2Y4OWw3em10ZThSNCtTMEk2VE0yQUIyN2kzNzZrcjBuUUVOTjV3OHRuaFNtRERiMHFGSCt5bjF5WlJtQWpoQVVGZEFCdFFMZGlKOU8zZXNKdVhyYUYwNHF1N2xMQUMzTHY3alVxTHRtY2xsck9xOW9XR3M2NDhTWHlSV3ZNYXc5L0s4OFc3cXFINUthcDh4WnlmaG5keHgya1FRMEwyWmZjU1A5eC9RUDN5cmxGYjBFMlVweUIzS01WQ2I2MzlCNnI5bm9penJQaDcvZTZGdEJTb0NIcFZtYkFnOG9sYnFXYnNyVVBMZmRIYjdDcHY1Z3Q0LzFkTkhFOVpLUmhwckZaUEVEZWJ0WmxrMUREemZMS2xsbXNLYVVjbmRCVnQxN0tvZXp0bXh2UGF0emNVWHh3bjN5cVV3QkpYWGVBb3FWRk4vZm56dUhpaFp2NmZOQ0VnRmhvZ3l5UG5wajlKMjlkcFo5Z0JFYThHazJlc2UyYUFzT2NxRTArUW01Q3BybmVQcStLNk5NK3dMSisxcERrTEtnVEhLTFdjVUZyT3orQmUrM3FWcTl1ZjNXYzZSMkUzbkoxbGpPTG5ZTlA3c21LQmp5V2NxRm82cHVjZEZmTDhEZTM4NkV4alFqVmZnSHZZcytrYjJDY3ZSdWR2VmRwdUZNUThRVUNwTks1TmNpd0hSOW5rbUdLWmM9'
url_7='https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/a6cc59ea73cdd09374d9959dbbd2baf2?q=VTJGc2RHVmtYMStGOEp5RnZiTUVzNjEwUnVyajhPMTNtUjVUVXh0RVhQWmFRVUQvczk1SjY0WWFCRk5sRS9HN1Q0R01ySUhLZ2lrL1ppMmdlS3Q0bW5SalcxbjgxZFQzMldiVUpMOHBEN1c3aW9LUkl0NnJFZXc2clp2eUJyUDNmc0tJZzRXZHE0RkhJS0RKc3RuVGdyWURRbEdpa0lGSHY4TzcxSjNaWHVqZGZCTEhiT3pLYk1leHpmSTc1V1NPelpsdFFDWXRSQUFnY2hUdWdlWkhpL0VkSGEyQk94UXcrTXEyWXpUTHFSaVFGUG5IWGlvYURSd01BUTVNUGdzelkwVUp5L2pBSW9zc2VSLzhTRXlta0xhRlNWczV3cFBVWFpQaXh4MUI2alRaWW9tNlVLNDJySzVIYkhSUVpNSkZObVQzMTRKNFk3T2tycWJRRnRHQjlUTThyQS9sa2g4QWtjdW53c2NJUFFRcS9JUzV4eWRkOGtxTTNvanYzRWs2cFZBNnlaY0FVdWg0ZVBXWG42SzNFTExvR1NmdSs5b2FKQytHSjJPMVJCRjN4WDJlVVNXRFVhRUNMbEJuaTBHT2owdEErVmkwclowalVMVjh6N0RNTlNyeXhwSWFtV0lLKzhrNUM3dndwSTBYZmluK3Q0MjBrM05vemJnak1QZ2lCaUFwamh3MnFRanJ5YTRodUYxYzN2TmozOUtzQkhOYXB4bEd0eW5qV05jNHhCVnMySHNCYnhzT0ZNVGpGZ1E1L3ZVQ0tDRHplaEFhNUFQUEEvdG1iS3IxMGs1WTU5WHUrbDB1ZVZiZXhneGgyUGY5K05kTjNVVDFRTm5lNlVrSWhZcW5HNHdzcmZ3Q3dDVGo0RERraVBjN1Z4a0Q3YUpEbXArSzhzR2daYkk9'
url=[url_0,url_1,url_2,url_3,url_4,url_5,url_6,url_7]

#-------------------------------------------------------------------------------

# 使用 requests 爬取網頁 ＆ 資料轉為 JOSN 格式 (根據測試，回傳11年資料＋JSON轉換，至少需要1-2分鐘)
print('Requesting 「{0}」 district data between 2012/Aug and 2022/Dec'.format(district))
print('May take a while, be patient...')

all_cases=[]
for i in range(8):
    resp=req.get(url[i])
    try:
        resp=req.get(url[i])
        resp.raise_for_status() # Raises HTTPError, if one occurred.
    except Exception as err:
        print(err)
    
    #將資料轉為 JOSN 格式，存到一個dict中 (ie. 變數 data)
    print('Web Scraping Done !')
    print('JSON -> Python dict...(may take a while.... be patient)')
    data=json.loads(resp.text)
    print('一共抓到 {0} 筆台中市{1}區101/8 ~ 111/12間的房價'.format(len(data),district[i]))
    
    # 只取出想要的欄位，並重新命名
    for case in data : # to get all cases 
        one_case={}               # dictionary used to store single case
        one_case['district'] = district[i]               #行政區
        one_case['address'] = case['a']                  #住址
        one_case['community'] = case['bn']               #社區名稱
        one_case['lon']= case['lon']                     #經度
        one_case['lat']= case['lat']                     #緯度     
        one_case['age'] = case['g']                      #屋齡
        one_case['area'] = case['s']                     #建物面積    
        one_case['build_type'] = case['b']               #建物型態 e.g.大樓,公寓,透天厝...etc    
        one_case['main_purpose'] = case['pu']            #主要用途    
        one_case['floor'] = case['f']                    #?/?層   
        one_case['layout'] = case['v']                   #格局
        one_case['elevator'] = case['el']                #有無電梯       
        one_case['manager'] = case['m']                  #有無管理員          
        one_case['parking_num'] = case['l']              #車位個數    
        one_case['build_share1'] = case['bs']            #主建物面積占建物移轉總面積（扣除車位面積）之比例   
        one_case['build_share2'] = case['es']            #主建物面積占建物移轉總面積之比例   
        one_case['note'] = case['note']                  #附記, e.g. 頂加 
        one_case['deal_date'] = case['e']                #成交日期(tw) 
        one_case['deal_year']=(one_case['deal_date'])[:3]#成交年份(year) 
        one_case['unit_price'] = case['p']               #單價  
        one_case['price'] = case['tp']                   #總價 (萬)
        all_cases.append(one_case.copy())

#-------------------------------------------------------------------------------

#--- 將原始數據保存到 CSV 檔案
cols=['district','address','community','lon','lat',
      'age','area','build_type','main_purpose',
      'floor','layout','elevator','manager','parking_num',
      'build_share1','build_share2','note',
      'deal_date','deal_year','unit_price','price']
df_house = pd.DataFrame (all_cases, columns = cols) # 將list 轉換為 dataframe
df_house.to_csv('price_201208_202212.csv',columns=cols,index=False)         

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# 資料前置處理(過濾)
# 分析並處理所有欄位的 invalid data
cols=['district','address','community','lon','lat',
      'age','area','build_type','main_purpose',
      'floor','layout','elevator','manager','parking_num',
      'build_share1','build_share2','note',
      'deal_date','deal_year','unit_price','price']
# 讀取CSV檔案
df_house = pd.read_csv("price_201208_202212.csv")

# 以平均值取代nan值
df_house.age = df_house.age.fillna(df_house.age.dropna().mean())

# 尋找重複值
duplicates = df_house[df_house.duplicated()]

# 刪除重複值
df_house.drop_duplicates(inplace=True)

# 使用pandas的str.replace()函數刪除逗號
df_house['price'] = df_house['price'].str.replace(',', '')
df_house['area'] = df_house['area'].str.replace(',', '')

# 將'Price'和'Area'欄位的資料類型從字串轉換為浮點數
df_house['price'] = pd.to_numeric(df_house['price'])
df_house['area'] = pd.to_numeric(df_house['area'])

# 定义编码字典
elevator_manager = {'有': 1, '無': 0}

# 使用map()函数进行编码
df_house['elevator'] = df_house['elevator'].map(elevator_manager)
df_house['manager'] = df_house['manager'].map(elevator_manager)
df_house['build_type']=df_house['build_type'].str.split("(").str[0]
df_house['build_type']=df_house['build_type'].str.split("（").str[0]

#把build type 和 district 所有的類別變成float 並且以數字表示
typeOfBuild=df_house.build_type.unique()
n = 1
typeOfBuildDict={}
for i in typeOfBuild:
    typeOfBuildDict[i]=n
    n+=1

district=['East','West','South','North','Central','Xitun','Beitun','Nantun']
n = 1
typeOfDistrictDict={}
for i in district:
    typeOfDistrictDict[i]=n
    n+=1
    
df_house['build_type'] = df_house['build_type'].map(typeOfBuildDict)
df_house['district'] = df_house['district'].map(typeOfDistrictDict)

#把address 多餘的文字去掉
def findaddress(address):

    match = re.search(r'([^\d]+)\d', address)
    if match:
        result = match.group(1)
    else:
        result = address
    return(result)

n=1

addressArray=list()

for i in df_house.address:
    #df_house.loc[df_house.address == i , "address"] = findaddress(i) #太慢了
    addressArray.append(findaddress(i))
    n+=1

position = df_house.columns.get_loc('address')
df_house.insert(position, 'addressname', addressArray)

"""#無用但可能未來會用
typeOfaddress=df_house.addressname.unique()
n = 1
typeOfaddressDict={}
for i in typeOfaddress:
    typeOfaddressDict[i]=n
    n+=1
    
df_house['addressname']=df_house['addressname'].map(typeOfaddressDict)
"""
df=df_house[['district','addressname','lon','lat',
             'age','area','build_type','elevator',
             'manager','parking_num','deal_year','price']]
# 將處理後的資料儲存到新的 csv 檔案
df.to_csv('filter.csv',index=False)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# 數據切分：將數據集切分為訓練集和測試集，用於建立和評估預測模型。
# 通常,大部分數據用於訓練模型，一小部分用於測試模型。
dfai = pd.read_csv("filter.csv")

x=dfai[['area','age','build_type','deal_year','district']]
y=dfai.price

# X_train: 訓練資料集(特徵), y_train: 訓練資料集 (標籤)
# X_test:  測試資料集(特徵), y_test:  測試資料集 (標籤)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=2023)

#重新組成完整dataframe (後續前置處理)
df_train=pd.concat([X_train, y_train], axis=1)
df_test=pd.concat([X_test, y_test], axis=1)

#用plot查看有沒有離群值
"""
plt.figure(figsize=(10,6))  
plt.scatter(df_train.area, df_train.price, s=5)
plt.title('Area vs. Price (Training Dataset)')
plt.xlabel('Area')
plt.ylabel('Price (10K)')
plt.show()

plt.figure(figsize=(10,6))  
plt.scatter(df_train.age, df_train.price, s=5)
plt.title('Age vs. Price (Training Dataset)')
plt.xlabel('Area')
plt.ylabel('Price (10K)')
plt.show()

"""

#處理掉離群值
#--- DPP: check outlier function
def check_outlier_IQR(df, col):
    #--- 使用Upper Limit 判斷 outlier, 
    # upper limit= Q3+IQR*1.5, where IQR=Q3-Q1
    # lower limit= Q1-IQR*1.5, where IQR=Q3-Q1
    upper_lim=df[col].quantile(q=0.75) +\
              (df[col].quantile(q=0.75)-df[col].quantile(q=0.25))*1.5 
    lower_lim=df[col].quantile(q=0.25) -\
              (df[col].quantile(q=0.75)-df[col].quantile(q=0.25))*1.5 
    cnt=df[(df[col]>=lower_lim)&(df[col]<=upper_lim)][col].count()          
    return lower_lim, upper_lim, cnt

#build_type，year,district 不會有離群值所以做其他的就好
#---DPP: check upper limit
ll, ul, cnt=check_outlier_IQR(df_train, 'area')
print('"Area of Training data" LL={0:.2f}, UP={1:.2f}, Filter={2:,}'.format(ll, ul, cnt))
ll, ul, cnt=check_outlier_IQR(df_train, 'age')
print('"Age of Training data" LL={0:.2f}, UP={1:.2f}, Filter={2:,}'.format(ll, ul, cnt))
ll, ul, cnt=check_outlier_IQR(df_train, 'price')
print('"Price of Training data" LL={0:.2f}, UP={1:.2f}, Filter={2:,}'.format(ll, ul, cnt))


df_train_wo_outlier=df_train[(df_train.area<=103) & 
                             (df_train.price<=26295000.00) & 
                             (df_train.age<=45.00)] 

#再用plot查看有沒有離群值
"""
plt.figure(figsize=(10,6))  
plt.scatter(df_train_wo_outlier.area, df_train_wo_outlier.price, s=2)
plt.title('Area vs. Price (Training data with outlier filtered)')
plt.xlabel('Area')
plt.ylabel('Price (10K)')
plt.show()

plt.figure(figsize=(10,6))  
plt.scatter(df_train_wo_outlier.age, df_train_wo_outlier.price, s=5)
plt.title('Age vs. Price (Training Dataset)')
plt.xlabel('Area')
plt.ylabel('Price (10K)')
plt.show()
"""
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# 模型選擇和建立：根據您的預測任務和數據特點，選擇合適的機器學習或深度學習模型。 
# 使用 DPP 後的training data 準備特徵資料 (feature)

X_train=df_train_wo_outlier[['area','age','build_type','deal_year','district']]  
X_test=df_train_wo_outlier[['area','age','build_type','deal_year','district']]

# 使用 DPP 後的training data 準備標籤 (label)
y_train=df_train_wo_outlier.price     
y_test=df_train_wo_outlier.price
    
model1 = RandomForestRegressor(random_state=2023)      # Random Forest Regression

model1.fit(X_train, y_train)

# 做預測
y_pred = model1.predict(X_test)

r2=metrics.r2_score(y_test, y_pred)
print("Random Forest Regression:")
print('R2 Score：{0:.3f}'.format(r2))

# By Adj R2 score
n=len(y_test)    #樣本的個數
k=len(x.columns) #變數的個數
adj_r2 = 1-(n-1)/(n-k-1)*(1-r2)
print("Adjusted R^2 : {0:.4f}".format(adj_r2)) 

# By MAE, MSE, RMSE
mae= metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)

print('Mean Absolute Error:', mae)  
print('Mean Squared Error:', mse)  
print('Root Mean Squared Error:', rmse)  

#-------------------------------------------------------------------------------
model2 = linear_model.LinearRegression() # Linear Model: Linear regression

model2.fit(X_train, y_train)

# 做預測
y_pred = model2.predict(X_test)

r2=metrics.r2_score(y_test, y_pred)
print("Linear Model: Linear regression:")
print('R2 Score：{0:.3f}'.format(r2))

# By Adj R2 score
n=len(y_test)    #樣本的個數
k=len(x.columns) #變數的個數
adj_r2 = 1-(n-1)/(n-k-1)*(1-r2)
print("Adjusted R^2 : {0:.4f}".format(adj_r2)) 

# By MAE, MSE, RMSE
mae= metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)

print('Mean Absolute Error:', mae)  
print('Mean Squared Error:', mse)  
print('Root Mean Squared Error:', rmse)  
print()
#-------------------------------------------------------------------------------
model3 = tree.DecisionTreeRegressor(random_state=2023) # Decision Tree Regression

model3.fit(X_train, y_train)

# 做預測
y_pred = model3.predict(X_test)

r2=metrics.r2_score(y_test, y_pred)
print("Decision Tree Regression:")
print('R2 Score：{0:.3f}'.format(r2))

# By Adj R2 score
n=len(y_test)    #樣本的個數
k=len(x.columns) #變數的個數
adj_r2 = 1-(n-1)/(n-k-1)*(1-r2)
print("Adjusted R^2 : {0:.4f}".format(adj_r2)) 

# By MAE, MSE, RMSE
mae= metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)

print('Mean Absolute Error:', mae)  
print('Mean Squared Error:', mse)  
print('Root Mean Squared Error:', rmse)  
print()
#-------------------------------------------------------------------------------
model4 = linear_model.SGDRegressor(random_state=2023)     # Stochastic Gradient Descent Regression

model4.fit(X_train, y_train)

# 做預測
y_pred = model4.predict(X_test)

r2=metrics.r2_score(y_test, y_pred)
print("Stochastic Gradient Descent Regression:")
print('R2 Score：{0:.3f}'.format(r2))

# By Adj R2 score
n=len(y_test)    #樣本的個數
k=len(x.columns) #變數的個數
adj_r2 = 1-(n-1)/(n-k-1)*(1-r2)
print("Adjusted R^2 : {0:.4f}".format(adj_r2)) 

# By MAE, MSE, RMSE
mae= metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)

print('Mean Absolute Error:', mae)  
print('Mean Squared Error:', mse)  
print('Root Mean Squared Error:', rmse) 
print()

#得出結論 Decision Tree Regression 數值比較好
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# 使用最佳化後的最終模型預測,經過測試decision Tree Regression 數值較好所以使用它

model=model3

preditPrice=list()
# 禁用警告
warnings.filterwarnings("ignore")
#建立表格看8大區未來1年的交易權重
for i in dfai.district.unique():
    m=df_train_wo_outlier.loc[(df_train_wo_outlier['district'] == i)&(df_train_wo_outlier['build_type'] <= 5)
                            , 'price'].mean()/10000
    one=(( model.predict([[df_train_wo_outlier.area.median(),
                      df_train_wo_outlier.age.median(),
                      1,112,i]])[0]+
      model.predict([[df_train_wo_outlier.area.median(),
                      df_train_wo_outlier.age.median(),
                      2,112,i]])[0]+
      model.predict([[df_train_wo_outlier.area.median(),
                      df_train_wo_outlier.age.median(),
                      3,112,i]])[0]+
      model.predict([[df_train_wo_outlier.area.median(),
                      df_train_wo_outlier.age.median(),
                      4,112,i]])[0]+
      model.predict([[df_train_wo_outlier.area.median(),
                      df_train_wo_outlier.age.median(),
                      5,112,i]])[0]
      )/5/10000)
    sales=len(df_train_wo_outlier.loc[(df_train_wo_outlier['district'] == i)])
    preditPrice.append([i,m,one,sales,(one-m)*sales/10000])

# 恢复警告
warnings.filterwarnings("default")
columns=['district','current price(10k)','after one year(10k)','number of sales','(current-oneyear)*sales/10000']
dfpredict = pd.DataFrame(preditPrice, columns=columns)
#從dfpredict 中得知 district 7 的權重較高所以接下來做著重 district 7 
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#畫district 7 的分群，通過lon 和 lat 來分群，看每個區的銷售情況
dfgroup = pd.read_csv("filter.csv")

def findthegroup(dfg,districtno):
    # 初始化WCSS列表
    wcss = []
    dfkmean=dfg.loc[dfg['district'] == districtno, ['addressname','age','build_type','area','lat', 'lon','price']]
    X = dfkmean[['lat', 'lon']]
    """#用於找分群數量 做ppt的記得把這圖 加進去
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, random_state=17)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    # 绘制簇数目与WCSS曲线 並找出最佳分群數量
    plt.plot(range(1, 10), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()
    """
    # 透過 Elbow Method 演算法，將資料分成四群為最佳
    km = KMeans(n_clusters=4, random_state=17)  # 為了確定隨機性=> random_state=17
    dfkmean['Clusters'] = km.fit_predict(X)   # 先fit然後預測X中每個樣本所屬的最近群集(cluster)
    
    """#plot 圖片ppt請解鎖來看
    # data visualization
    colors = ['purple', 'blue', 'green', 'red']
    #colors=['purple', 'blue', 'green','yellow','red']
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    for i in range(km.n_clusters):
        df_cluster=dfkmean[dfkmean['Clusters']==i]
        ax.scatter(df_cluster['lat'],
                   df_cluster['lon'],
                   s=25,label='Cluster'+str(i), c=colors[i])
    ax.set_xlabel('lat', fontsize=10)
    ax.set_ylabel('lon', fontsize=10)
    ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=50,marker='^', c='red', alpha=0.7, label='Centroids')
    plt.legend()
    plt.show()
    """
    return dfkmean

findthegroup(dfgroup,7)
    
#發現 district 7 標籤的lon &lat 有異常先處理異常值
ll, ul, cnt=check_outlier_IQR(dfgroup, 'lat')
print('"Area of Training data" LL={0:.2f}, UP={1:.2f}, Filter={2:,}'.format(ll, ul, cnt))
ll, ul, cnt=check_outlier_IQR(dfgroup, 'lon')
print('"Area of Training data" LL={0:.2f}, UP={1:.2f}, Filter={2:,}'.format(ll, ul, cnt))

dfgroupfilter=dfgroup[(dfgroup.lon<=120.75) & (dfgroup.lon>=120.58) &
              (dfgroup.lat<=24.22)& (dfgroup.lat>=24.10)&
              (dfgroup.area<=103) & 
              (dfgroup.price<=26295000.00) & 
              (dfgroup.age<=45.00)]


df7filter=findthegroup(dfgroupfilter,i)

def showHouseSales(value,districtno):
    plt.figure(figsize=(10,7))  # 設定圖形大小
    plt.bar(value.index,value.values, label="EPS") #x軸:index of series , y軸:values of series
    plt.title(districtno)
    plt.xlabel('house type')
    plt.ylabel('Number Sold')
    plt.show()
    
housesales=df7filter['build_type'].value_counts()
housesales=housesales[housesales.index<=5]
housesales=housesales.sort_values(ascending = False)
dict_keys = [key for key, val in typeOfDistrictDict.items() if val == 7]

showHouseSales(housesales,dict_keys)
#['華廈': 1, '住宅大樓': 2, '透天厝': 3, '套房': 4, '公寓': 5]
#圖片得知 2 也就是住宅大樓在beitun（北屯）的交易量很高
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#接下來要查看住宅大樓和面積和年齡之間的關聯
df2=df7filter[(df7filter.build_type==2)]
def findtherelation(dfgp,a,b):
    # 初始化WCSS列表
    wcss = []
    dfkmean=dfgp
    X = dfkmean[[a, b]]
    #用於找分群數量 做ppt的記得把這圖 加進去
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, random_state=17)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    # 绘制簇数目与WCSS曲线 並找出最佳分群數量
    plt.plot(range(1, 10), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()
    
    # 透過 Elbow Method 演算法，將資料分成四群為最佳
    km = KMeans(n_clusters=3, random_state=17)  # 為了確定隨機性=> random_state=17
    dfkmean['Clusters'] = km.fit_predict(X)   # 先fit然後預測X中每個樣本所屬的最近群集(cluster)
    
    #plot 圖片ppt請解鎖來看
    # data visualization
    colors = ['purple', 'blue', 'green', 'red']
    #colors=['purple', 'blue', 'green','yellow','red']
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    for i in range(km.n_clusters):
        df_cluster=dfkmean[dfkmean['Clusters']==i]
        ax.scatter(df_cluster[a],
                   df_cluster[b],
                   s=25,label='Cluster'+str(i), c=colors[i])
    ax.set_xlabel(a, fontsize=10)
    ax.set_ylabel(b, fontsize=10)
    ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=50,marker='^', c='red', alpha=0.7, label='Centroids')
    plt.legend()
    plt.show()
    
#找出area 和 age的關係 得出最多交易的area 和age
findtherelation(df2,'area','age')
result1=df2['Clusters'].value_counts()
print(result1)
#知道Clusters 2為最多所以過濾出Clusters 2的資料
dfresult=df2[df2['Clusters']==2]
print("住宅大樓交易熱門age 位於",dfresult.age.min(),"至",dfresult.age.max(),"之間")
print("住宅大樓交易熱門area 位於",dfresult.area.min(),"至",dfresult.area.max(),"之間")
print("住宅大樓交易熱門price(10k) 位於",dfresult.price.min()/10000,"至",dfresult.price.max()/10000,"之間")
rlonmax,rlonmin=dfresult.lon.max(),dfresult.lon.max()
rlatmax,rlatmin=dfresult.lat.max(),dfresult.lat.max()

import folium

# 計算中心點
center_lat = (rlatmax + rlatmin) / 2
center_lon = (rlonmax + rlonmin) / 2

# 創建地圖
m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# 繪製圓圈
folium.Circle(
    radius=1000,  # 設定半徑，單位是米
    location=[center_lat, center_lon],  # 設定中心點坐標
    color='blue',
    fill=True,
    fill_color='blue'
).add_to(m)

# 保存地圖為HTML文件
m.save('map.html')
