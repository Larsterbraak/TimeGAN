"""
MSc Thesis Quantitative Finance
Title: Interest rate risk due to EONIA-ESTER transition
Author: Lars ter Braak (larsterbraak@gmail.com)

Last updated: May 26th 2020
Code Author: Lars ter Braak (larsterbraak@gmail.com)

-----------------------------

Plotting 
- Code that helps to visualize the short rate data
 
Inputs
(1) EONIA, pre-ESTER & ESTER dataset

Outputs
(1) Histogram of daily differences for EONIA, pre-ESTER and ESTER
(2) Short rates EONIA, pre-ESTER & ESTER over the complete history
(3) ESTER during pre-ESTER period and the 4 additional characteristics:
    [25th percentile, 75th percentile, Transact volume and Nr. of transactions]
"""

# Necesarry packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

# More efficient and elegant pyplot
plt.style.use(['science', 'no-latex'])

def plot(hist = False, history = False, pre_ester=False):
    df_pre_ESTER = pd.read_csv('data/pre_ester.csv', sep = ';')
    df_pre_ESTER = df_pre_ESTER.iloc[::-1] # Reverse df to be chronological
    df_EONIA = pd.read_csv("data/EONIA.csv", sep=";")
    df_ESTER = pd.read_csv("data/ESTER.csv", sep=";")
    df_ESTER = df_ESTER.iloc[::-1] # Reverse df to be chronological
    
    if hist: # Show the histogram of the daily differences
        plt.figure(figsize=(12,8))
        plt.hist(np.diff(df_EONIA.EONIA.values[4662:5426]), bins = 75, 
                 facecolor = '#0C5DA5', edgecolor = '#169acf',
                 linewidth=0.5, density = True, alpha = 0.7,
                 label = r'EONIA')
        
        plt.hist(np.append(np.diff(df_pre_ESTER.WT), 
                           np.diff(df_ESTER.WT.values)),
                 bins = 75, facecolor = '#00B945', edgecolor = '#169acf',
                 linewidth=0.1, density = True, alpha = 0.7, 
                 label = r'pre-€STER & €STER')
        
        plt.title(r'''Histogram daily difference in $r_t$ during pre-€STER and €STER period''')
        plt.xlabel(r'Daily difference in $r_t$')
        plt.ylabel(r'P(X = x)')
        plt.legend(fontsize = 'xx-large')
        plt.show()
        
    if history: # Show the history of the EONIA and ESTER since inception        
        dates_EONIA = np.ravel(df_EONIA.Date.values).astype(str)
        dates_EONIA = [datetime.datetime.strptime(d,"%d-%m-%Y").date()
                       for d in dates_EONIA]
        dates_pre_ESTER = np.ravel(df_pre_ESTER.Date.values).astype(str)
        dates_pre_ESTER = [datetime.datetime.strptime(d,"%d-%m-%Y").date()
                           for d in dates_pre_ESTER]
        dates_ESTER = np.ravel(df_ESTER.Date.values).astype(str)
        dates_ESTER = [datetime.datetime.strptime(d,"%d-%m-%Y").date()
                           for d in dates_ESTER]
        
        plt.figure(figsize=(12,8))
        plt.plot_date(dates_EONIA, df_EONIA.EONIA, 'b-', color = '#0C5DA5')
        plt.plot_date(dates_pre_ESTER, df_pre_ESTER.WT.values, 'b-', color = '#00B945')
        plt.plot_date(dates_ESTER, df_ESTER.WT.values, 'b-', color = '#FF9500')
        ax = plt.gca()
        
        ax.set_xlim(datetime.date(2017, 9, 30), datetime.date(2020, 3, 12))
        ax.set_ylim((-.60, -.20))
        plt.legend(('EONIA', 'pre-€STER', '€STER'), fontsize = 'xx-large')
        plt.ylabel(r'Short rate $r_t$ [%]')
        plt.xlabel(r'Time $t$')
        plt.title(r'Short rates $r_t$ over time')
        plt.show()
    
    if pre_ester: # Show all characteristics of ESTER during pre-ESTER time
        plt.figure(figsize=(12,8))    
        
        # The top plot consisting of daily closing prices
        top = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
        top.plot(dates_pre_ESTER, df_pre_ESTER.R75, 
                 label = r'$75^{th}$ percentile €STER')
        top.plot(dates_pre_ESTER, df_pre_ESTER.WT, 
                 label = r'Weighted trimmed mean €STER')
        top.plot(dates_pre_ESTER, df_pre_ESTER.R25, 
                 label = r'$25^{th}$ percentile €STER')
        plt.title('€STER during pre-€STER period')
        plt.xlabel(r'Time $t$')
        plt.ylabel(r'Short rate $r_t$ [$\%$]')
        top.legend(loc = 'best', fontsize = 'xx-large')
        
        # The bottom plot consisting of daily trading volume
        bottom = plt.subplot2grid((4, 4), (3,0), rowspan=1, colspan=4)
        bottom.plot(dates_pre_ESTER, df_pre_ESTER.TT *1e6, 
                    color = '#FF2C00', label = r'Transaction volume')
        bottom.plot(dates_pre_ESTER, df_pre_ESTER.NT *1e8, 
                    color = '#845B97', label = r'Nr. of transactions')
        
        def to_transactions(x):
            return x * 1e-8
        
        def to_volume(x):
            return x * 1e8
        
        ax = plt.gca()
        secaxy = ax.secondary_yaxis('right', functions = (to_transactions, to_volume))
        secaxy.set_ylabel(r'Nr. of transactions')
        
        plt.title('Transaction volume during pre-€STER period')
        plt.xlabel(r'Time $t$')
        plt.ylabel(r'Volume $[€]$')
        plt.ylim((2e10, 9e10))
        plt.legend(loc = 'upper right')
        
        plt.gcf().set_size_inches(12, 8)
        plt.subplots_adjust(hspace=0.75)