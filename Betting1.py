from scipy.stats import pearsonr
from scipy.stats import chisquare
import pandas as pd
import numpy as np
from numpy import column_stack
from random import random
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


headers = ['Home_Team_Win_percentages']
row = [x for x in range(1,13)]
data = np.array([[100*random()],
                 [100*random()],
                 [100*random()],
                 [100*random()],
                 [100*random()],
                 [100*random()],
                 [100*random()],
                 [100*random()],
                 [100*random()],
                 [100*random()],
                 [100*random()],
                 [100*random()]])
headers2 = ['Feature_1','Feature_2','Feature_3', 'Feature_4','Feature_5']
row2 = [x for x in range(1,13)]
data2 = np.array([[10*random(),10*random(),10*random(),10*random(),10*random()],
                 [10*random(),10*random(),10*random(),10*random(),10*random()],
                 [10*random(),10*random(),10*random(),10*random(),10*random()],
                 [10*random(),10*random(),10*random(),10*random(),10*random()],
                 [10*random(),10*random(),10*random(),10*random(),10*random()],
                  [10*random(),10*random(),10*random(),10*random(),10*random()],
                  [10*random(),10*random(),10*random(),10*random(),10*random()],
                  [10*random(),10*random(),10*random(),10*random(),10*random()],
                  [10*random(),10*random(),10*random(),10*random(),10*random()],
                  [10*random(),10*random(),10*random(),10*random(),10*random()],
                  [10*random(),10*random(),10*random(),10*random(),10*random()],
                  [10*random(),10*random(),10*random(),10*random(),10*random()]])
headers3 = ['Outcome']
row3 = [x for x in range(1,13)]
data3 = np.array([[1],[2],[2],[1],[1],[1],[2],[2],[2],[1],[1],[2]])

table = pd.DataFrame(data,row,headers)
table2 = pd.DataFrame(data2,row2,headers2)
table3 = pd.DataFrame(data3,row3,headers3)


def pearsons(x,y):
    X = np.array([pearsonr(x,y)])
    ZP = pd.DataFrame(X, ['Pearsons Correlation'],['Z-value','P-value'])
    print ZP
    return ZP

def MIS(x,y):
    Z = mutual_info_score(x,y)
    print 'Mutual Info Score:', Z
    return Z

def recurssive(x,y,model,n):
    rfe = RFE(model, n)
    rfe = rfe.fit(x, y)
    # summarize the selection of the attributes
    support = rfe.support_
    ranking = rfe.ranking_
    data = np.array([ranking])
    column_label = list(x.columns.values)
    ranking = pd.DataFrame(data,['Ranking'],column_label)
    print ranking
    return ranking

print chisquare(table2.Feature_1)
recurssive(table2, table3, LogisticRegression(), 1)
pearsons(table2.Feature_1, table.Home_Team_Win_percentages)
MIS(table2.Feature_1, table.Home_Team_Win_percentages)
