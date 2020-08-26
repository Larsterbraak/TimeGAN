import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

os.chdir('C://Users/s157148/Documents/Github/TimeGAN')

# More efficient and elegant pyplot
plt.style.use(['science', 'no-latex'])

df_spreads = pd.read_csv('data/Euribor_spreads.csv', sep = ';')
plt.figure(figsize=(12,8))
dates_ESTER = [datetime.datetime.strptime(d,"%d-%m-%Y").date()
                           for d in df_spreads.Date2[:-1]]
plt.plot_date(dates_ESTER, df_spreads.iloc[:-1,1].values, 'b-', color = '#0C5DA5')
plt.plot_date(dates_ESTER, df_spreads.iloc[:-1,2].values, 'b-', color = '#00B945')
plt.plot_date(dates_ESTER, df_spreads.iloc[:-1,3].values, 'b-', color ='#FF9500')
plt.plot_date(dates_ESTER, df_spreads.iloc[:-1,4].values, 'b-', color = '#FF2C00')
plt.plot_date(dates_ESTER, df_spreads.iloc[:-1,5].values, 'b-', color = '#845B97')
plt.plot_date(dates_ESTER, df_spreads.iloc[:-1,6].values, 'b-', color = '#474747')
plt.plot_date(dates_ESTER, df_spreads.iloc[:-1,7].values, 'b-', color = '#9e9e9e')
plt.plot_date(dates_ESTER, df_spreads.iloc[:-1,8].values, 'b-')
plt.title(r'''$Spread_t(\tau)$ during EONIA and €STER period''')
plt.xlabel(r'Time $t$')
plt.ylim((0, 2))
plt.ylabel(r'$Spread_t(\tau)$')
plt.legend(('1W', '2W', '1M', '2M', '3M', '6M', '9M', '12M'), fontsize = 'xx-large')
plt.show()

df_infl_real_gdp = pd.read_csv('data/Inflation_real_GDP_combined.csv', sep=';')
df_infl_real_gdp = df_infl_real_gdp[['Month', 'Real_GDP_growth.1', 'Inflation']]
        
plt.figure(figsize=(12,8))
dates_ESTER = [datetime.datetime.strptime(d,"%Y-%m").date()
                           for d in df_infl_real_gdp.Month]
plt.plot_date(dates_ESTER, df_infl_real_gdp['Real_GDP_growth.1'].values, 'b-', color = '#0C5DA5')
plt.plot_date(dates_ESTER, df_infl_real_gdp.Inflation.values, 'b-', color = '#00B945')
plt.title(r'''Rate of inflation ($\pi_t$) and Real GDP growth ($y_t$) during EONIA and €STER period''')
plt.xlabel(r'Time $t$')
plt.ylabel(r'$Percentage$')
plt.legend(('Real GDP growth', 'Inflation'), fontsize = 'xx-large')
plt.show()