#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 08:25:15 2022

@author: denizaycan
"""


import wbdata
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import regfunction


#checked https://wbdata.readthedocs.io/en/stable/ 
#chose sustainable development goals module for the analysis
#print(wbdata.get_indicator(source=46))
turkey = [i['id'] for i in wbdata.get_country(country_id=("TUR"))]

sourcecode = {"NY.GNP.MKTP.PC.CD":"GNI per Capita", "SL.UEM.TOTL.FE.NE.ZS": "National Female Unemployment"}

datas = wbdata.get_dataframe(sourcecode, country=turkey)

#with dropna I determined the range of years for the analysis
datas.dropna(inplace=True)
print (datas)

datas.to_csv('HW2aycande.csv')

#formed x and y variables from the dataset and added columns of 1
Y = np.array(datas.iloc[:, :1]).reshape((datas.shape[0], 1))
X = np.hstack((np.ones((datas.shape[0],1)),np.array(datas.iloc[:,1:])))

results = regfunction.regfunction(X, Y)

print(results)
print("The regression fitted as GNI per Capita = ", results[0][0], "+", results[0][1], " * National Female Unemployment")
print("The results are statistically insignificant.")

#visualised with the help of seaborn and matplotlib
graph= seaborn.lmplot(x= "National Female Unemployment", y="GNI per Capita", data=datas)
plt.title("National Female Unemployment against GNI per Capita")
plt.savefig("NFU-GNIPC-Regression")