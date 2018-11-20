""" 
Plotting annual time Trends in 5 variables: 
Early repayment rate, default rate, choice of loan-terms (ratio), dti, income.

defcount: number of defaults in period
ERcount : no. of early repayment in period
Schcount : no of repaid on time per schedule
Current: no. still current (no default or repayment)

The su of all 4 is exhaustive states:   
	Def_rate : thus decount/sum, Imly, ER_rate

#For loans issued in a period
DTI: debt to income ratio (0-100)
short/longcount, termRatio: #of term 36omnth or term 60 month loans issued in period

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('data/merge0.csv')

df.columns
# Index(['defcount', 'addr_state', 'ERcount', 'Schcount', 'Curcount', 'DTI',
#        'income', 'shortcount', 'longcount', 'Period', 'ER_rate', 'Def_rate',
#        'termRatio'],
#       dtype='object')

df['numLoans_issued'] = df.shortcount+df.longcount

df.head()
# 	 defcount state  ERcount  Schcount  Curcount        DTI        income  shortcount  longcount      Period   ER_rate  Def_rate  termRatio
# 0        15         AZ       36         2       353  12.287082  75486.044591         182         75  2010-01-01  0.088670  0.036946   0.412088
# 1         8         SC       19         1       182  12.605338  60300.424662          93         40  2010-01-01  0.090476  0.038095   0.430108
# 2         3         LA       12         3       179  13.863664  86128.764885          87         44  2010-01-01  0.060914  0.015228   0.505747
# 3        34         NJ       66        19       823  12.656628  73944.853339         442        166  2010-01-01  0.070064  0.036093   0.375566
# 4        11         VA       50         7       651  13.192647  69798.517374         355        121  2010-01-01  0.069541  0.015299   0.340845

df['Period'] = df['Period'].astype('datetime64[D]')
df['ER_rate'] = 100*df['ER_rate']
df['Def_rate'] = 100*df['Def_rate']

#Default and Repayment
timePivotDef_Rate = df.pivot(index='Period', columns='State',values='Def_rate') 
timePivotDef_Rate[['CA', 'TX', 'NY', 'FL', 'IL']].plot()
plt.show()

timePivotER_Rate = df.pivot(index='Period', columns='State',values='ER_rate') 
timePivotER_Rate[['CA', 'TX', 'NY', 'FL', 'IL']].plot()
plt.show()


# Newly issued loans

#total loans issued
timePivotIssued = df.pivot(index='Period', columns='State',values='numLoans_issued') 
timePivotIssued[['CA', 'TX', 'NY', 'FL', 'IL']].plot()
plt.show()

#term length
timePivottermRatio = df.pivot(index='Period', columns='State',values='termRatio') 
timePivottermRatio[['CA', 'TX', 'NY', 'FL', 'IL']].plot()
plt.show()

timePivotDTI = df.pivot(index='Period', columns='State',values='DTI') 
timePivotDTI[['CA', 'TX', 'NY', 'FL', 'IL']].plot()
plt.show()


##income increasing
timePivotIncome = df.pivot(index='Period', columns='State',values='income')  
timePivotIncome[['CA', 'TX', 'NY', 'FL', 'IL']].plot()
plt.show()


# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
# timePivotDef_Rate[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[0,0]); axes[0,0].set_title('A');
# timePivotER_Rate[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[0,1]); axes[0,1].set_title('B');
# timePivotDTI[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[1,0]); axes[1,0].set_title('C');
# timePivottermRatio[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[1,1]); axes[1,1].set_title('D');
# plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
timePivotDef_Rate[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[0]); axes[0].set_title('Default Rates (%)');
timePivotER_Rate[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[1]); axes[1].set_title('Early Repayment Rates (%)');
#plt.show()
plt.savefig('images/QrtlytrendsDefER.png')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
timePivotDTI[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[1]); axes[1].set_title('Debt to Income Ratios (%)');
timePivotIncome[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[0]); axes[0].set_title('Borrower Incomes');
#plt.show()
plt.savefig('images/QrtlytrendIncome.png')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
timePivotIssued[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[0]); axes[0].set_title('Number of Loans Issued');
timePivottermRatio[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[1]); axes[1].set_title('Loan Term Ratio (5yr/3yr)');
#plt.show()
plt.savefig('images/QrtlytrendloansTerms.png')



##all in one
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 8), sharex=True)
timePivotIssued[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[0,0]); axes[0,0].set_title('Number of Loans Issued');
timePivottermRatio[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[0,1]); axes[0,1].set_title('Loan Term Ratio (5yr/3yr)');
timePivotIncome[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[1,0]); axes[1,0].set_title('Borrower Incomes');
timePivotDTI[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[1,1]); axes[1,1].set_title('Debt to Income Ratios (%)');
timePivotDef_Rate[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[2,0]); axes[2,0].set_title('Default Rates (%)');
timePivotER_Rate[['CA', 'TX', 'NY', 'FL', 'IL']].plot(ax=axes[2,1]); axes[2,1].set_title('Early Repayment Rates (%)');
#plt.show()
plt.savefig('images/QrtlytrendCombo.png')

