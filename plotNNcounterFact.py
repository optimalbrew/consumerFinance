"""
Plotting the predictions from the neural network
This version uses counterfactual/synthetic data.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/resultNNCounterFact.csv')

df.columns
# Index(['probDefPred', 'timeDefPred', 'probERPred', 'timeERPred', 'probDef',
#        'timeDef', 'probER', 'timeER'],
#       dtype='object')
df.shape
# (1900, 8)
 

#df[:][0:4]
#   probDefPred  timeDefPred  probERPred  timeERPred  probDef  timeDef  probER    timeER  probOrig  probOrigPred
# 0     0.548441     0.935703    0.649535    0.735008      0.0      1.0     1.0  0.091398       0.0     -0.197976
# 1    -0.083071     0.998786    0.869842    0.487148      0.0      1.0     1.0  0.430108       0.0      0.213230
# 2     0.152641     0.822486    0.613376    0.613784      0.0      1.0     1.0  0.400000       0.0      0.233983
# 3     0.258570     0.801463    0.458434    0.813443      0.0      1.0     1.0  0.758065       0.0      0.282997

df[['probDef','probDefPred','probER','probERPred','timeER','timeERPred','timeDef','timeDefPred']].describe()



#probability of loan being repaid as per original term
df['probOrig'] = 1-df['probDef'] - df['probER']
#as predicted by model
df['probOrigPred'] = 1-df['probDefPred'] - df['probERPred']

prob_labels = ['Default', 'Repaid Early', 'Orig. Term']
ind = np.arange(len(prob_labels))  # the x locations for the groups: range from 0 to len()-1
width = 0.35  # the width of the bars

fig, ax1 = plt.subplots(1,1,figsize=(8,6)) 
ax1.bar(x=ind - width/2, height = 100*df[['probDef','probER','probOrig']].mean(), width = width, 
				color= 'forestgreen', label='Actual'
                                )
ax1.bar(x = ind + width/2, height = 100*df[['probDefPred','probERPred','probOrigPred']].mean(), width = width, 
                color= 'orange', label='Predicted',
                )
ax1.set_ylabel('Probability (%)')
#ax1.set_xlabel(' ')
ax1.set_title('Event predictions')
ax1.set_xticks(ind)
ax1.set_xticklabels(prob_labels)
ax1.legend()

#plt.show()
plt.savefig('images/predsNNOverallCounterFact.png')


# Density plots (what the NN is trying to match (partly), includes constructed counterfactuals)
# df[['timeDef','timeDefPred']].plot.kde(ind=pd.Series(np.arange(0,1,.01)))
# plt.show()
# plt.savefig('images/predsNNDensityDefCounterFact.png')
# 
# df[['timeER','timeERPred']].plot.kde(ind=pd.Series(np.arange(0,1,.01)))
# plt.show()
# plt.savefig('images/predsNNDensityERCounterFact.png')



#conditionals distribution plots (what the NN is not matching, this is only actual observations, with missing counterfactuals )
## conditional on default (actual)
dfDef = df.loc[lambda df: df.probDef==1.0,['timeDef','timeDefPred']]
dfDef.columns = ['actual','predicted']

#dfDef.plot.scatter(x='actual',y='predicted')

#fig, 
ax2 = dfDef.plot.kde(ind=pd.Series(np.arange(0,1,.01)))
ax2.set_xlim = (0,100)
ax2.set_ylabel('Density')
ax2.set_xlabel('Fraction of loan term')
ax2.set_title('Timeline cond. on actual default')
plt.savefig('images/predsNNDensityDefCondCounterFact.png')

#plt.show()

## Conditional on early repayment (actual)
dfER = df.loc[lambda df: df.probER==1.0,['timeER','timeERPred']]
dfER.columns = ['actual','predicted']

#dfER.plot.scatter(x='actual',y='predicted')


ax3 = dfER.plot.kde(ind=pd.Series(np.arange(0,1,.01)))#plt.subplots(1,1,figsize=(8,6))
ax3.set_ylabel('Density')
ax3.set_xlabel('Fraction of loan term')
ax3.set_title('Timeline cond. on actual early repay,')
#ax3.set_xlim = [[0,10]]
#plt.show()
plt.savefig('images/predsNNDensityERCondCounterFact.png')


# Relationship between predicted prob of early repayment and predicted timing
