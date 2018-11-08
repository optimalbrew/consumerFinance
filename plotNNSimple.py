"""
Simple plots of predictions from the fitted neural net.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/testNNresultSimple.csv')

df.columns
#Index(['probDefPred', 'probERPred', 'fracNumPmtsPred', 'probDef', 'probER','fracNumPmts', 'probOrig', 'probOrigPred'],
      #dtype='object')

df.shape

#probability of loan being repaid as per original term
df['probOrig'] = 1-df['probDef'] - df['probER']
#as predicted by model
df['probOrigPred'] = 1-df['probDefPred'] - df['probERPred']

prob_labels = ['Default', 'Repaid Early', 'Orig. Term']
ind = np.arange(len(prob_labels))  # the x locations for the groups: range from 0 to len()-1
width = 0.35  # the width of the bars

fig, ax = plt.subplots(1,1,figsize=(8,6)) 
ax.bar(x=ind - width/2, height = 100*df[['probDef','probER','probOrig']].mean(), width = width, 
				color= 'forestgreen', label='Actual'
                                )
ax.bar(x = ind + width/2, height = 100*df[['probDefPred','probERPred','probOrigPred']].mean(), width = width, 
                color= 'orange', label='Predicted',
                )
ax.set_ylabel('Probability (%)')
#ax1.set_xlabel(' ')
ax.set_title('Event predictions')
ax.set_xticks(ind)
ax.set_xticklabels(prob_labels)
ax.legend()

#plt.show()
plt.savefig('images/predsNNOverall.png')


#conditionals distribution plots (what the NN is not matching, this is only actual observations, with missing counterfactuals )
## conditional on default (actual)
dfDef = df.loc[lambda df: df.probDef==1.0,['fracNumPmts','fracNumPmtsPred']]
dfDef.columns = ['actual','NN prediction']

ax2 = dfDef.plot.kde(ind=pd.Series(np.arange(0,1,.01)))
#ax2.set_xlim = (bottom=0,top=100)
ax2.set_ylabel('Density')
ax2.set_xlabel('Fraction of loan term')
ax2.set_title('Expected number of payments (eventual default)')
plt.savefig('images/predsNNDensityDefCond.png')

#plt.show()

## Conditional on early repayment (actual)
dfER = df.loc[lambda df: df.probER==1.0,['fracNumPmts','fracNumPmtsPred']]
dfER.columns = ['actual','NN prediction']

#dfER.plot.scatter(x='actual',y='predicted')


ax3 = dfER.plot.kde(ind=pd.Series(np.arange(0,1,.01)))#plt.subplots(1,1,figsize=(8,6))
ax3.set_ylabel('Density')
ax3.set_xlabel('Fraction of loan term')
ax3.set_title('Expected number of payments (early repayment)')
#ax3.set_xlim = [[0,10]]
#plt.show()
plt.savefig('images/predsNNDensityERCond.png')
