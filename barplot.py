"""
Getting the barplots done.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data/pdDefRate.csv")
#['grade', 'def36mean', 'def36sd', 'def60mean', 'def60sd', 
	#'defTime36mean', 'defTime36sd', 'defTime60mean', 'defTime60sd', 
		#'repTime36mean', 'repTime36sd', 'repTime60mean', 'repTime60sd'])

grade =data['grade']

#convert to percent
defRate36mean = 100*data['def36mean']
defRate60mean = 100*data['def60mean']
defRate36sd = 100*data['def36sd']
defRate60sd = 100*data['def60sd']


#setup the plot paras
ind = np.arange(len(grade))  # the x locations for the groups: range from 0 to len()-1
width = 0.35  # the width of the bars

fig, [[ax1, ax3], [ax2, ax4]] = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(8, 6))

#fig1, ax1 = plt.subplots()
ax1.bar(x=ind - width/2, height = defRate36mean, width = width, 
				#yerr = defRate36sd,
                color= 'silver', label='36 months',
                # color= 'yellowgreen', label='36 months',
                #error_kw=dict(ecolor='lightgray', lw=1, capsize=2, capthick=1)
                )
ax1.bar(x = ind + width/2, height = defRate60mean, width = width, 
				#yerr= defRate60sd,
                color= 'gray', label='60 months',
                # color= 'gold', label='60 months',
                #error_kw=dict(ecolor='lightgray', lw=1, capsize=2, capthick=1)
                )

ax1.set_ylabel('Loan default rate (%)')
#ax1.set_xlabel('Loan (credit) grade')
ax1.set_title('Loan default rate (%)')
ax1.set_xticks(ind)
ax1.set_xticklabels(grade)
ax1.legend()

#plt.show()
#fig1.savefig('images/defaultRate.png', bbox_inches='tight')


#Same information in terms of repayment rates (no change in variance)
#fig2, ax2 = plt.subplots()
ax2.bar(x=ind - width/2, height = 100-defRate36mean, width = width, 
				#yerr = defRate36sd,
                color= 'silver', label='36 months',
                # color= 'yellowgreen', label='36 months',
                #error_kw=dict(ecolor='gray', lw=1, capsize=2, capthick=1)
                )
ax2.bar(x = ind + width/2, height = 100 - defRate60mean, width = width, 
				#yerr= defRate60sd,
                color= 'gray', label='60 months',
                # color= 'gold', label='60 months',
                #error_kw=dict(ecolor='gray', lw=1, capsize=2, capthick=1)
                )
ax2.set_ylim(bottom=30)
ax2.set_ylabel('Loan repayment rate (%)')
#ax2.set_xlabel('Loan (credit) grade')
ax2.set_title('Loan repayment rate (%)')
ax2.set_xticks(ind)
ax2.set_xticklabels(grade)
ax2.legend()

#plt.show()
#fig2.savefig('images/repayRate.png', bbox_inches='tight')




# when does default occur?
## convert default time to months
Def_Month36mean = 36*data['defTime36mean']
Def_Month60mean = 60*data['defTime60mean']
Def_Month36sd = 36*data['defTime36sd']
Def_Month60sd = 60*data['defTime60sd']




#fig3, ax3 = plt.subplots()
ax3.bar(x=ind - width/2, height = Def_Month36mean, width = width,
				#yerr = Def_Month36sd,
                color= 'silver', label='36 months',
                # color= 'yellowgreen', label='36 months',
                #error_kw=dict(ecolor='lightgray', lw=1, capsize=2, capthick=1)
                )
ax3.bar(x = ind + width/2,  height = Def_Month60mean, width=width, 
				#yerr= Def_Month60sd,
                color= 'gray', label='60 months',
                # color= 'gold', label='60 months',
                #error_kw=dict(ecolor='lightgray', lw=1, capsize=2, capthick=1)
                )
ax3.set_ylim(top=45)
ax3.set_ylabel('Last payment received (months)')
#ax3.set_xlabel('Loan (credit) grade')
ax3.set_title('Mean Time of default (months)')
ax3.set_xticks(ind)
ax3.set_xticklabels(grade)
ax3.legend()

#plt.show()
#fig3.savefig('images/defTime.png', bbox_inches='tight')


# when does repayment occur?
## convert repayment time to months
Rep_Month36mean = 36*data['repTime36mean']
Rep_Month60mean = 60*data['repTime60mean']
Rep_Month36sd = 36*data['repTime36sd']
Rep_Month60sd = 60*data['repTime60sd']

#fig4, ax4 = plt.subplots()
ax4.bar(x=ind - width/2, height = Rep_Month36mean, width = width, 
				#yerr = Rep_Month36sd,
                color= 'silver', label='36 months',
                # color= 'yellowgreen', label='36 months',
                #error_kw=dict(ecolor='lightgray', lw=1, capsize=2, capthick=1)
                )
ax4.bar(x = ind + width/2,  height = Rep_Month60mean, width=width, 
				#yerr= Rep_Month60sd,
                color= 'gray', label='60 months',
                # color= 'gold', label='60 months',
                #error_kw=dict(ecolor='lightgray', lw=1, capsize=2, capthick=1)
                )
ax4.set_ylim(top=45)
ax4.set_ylabel('When loan repaid (months)')
#ax4.set_xlabel('Loan (credit) grade')
ax4.set_title('Mean Time of Repayment (months)')
ax4.set_xticks(ind)
ax4.set_xticklabels(grade)
ax4.legend()

#plt.show()
#fig4.savefig('images/repayTime.png', bbox_inches='tight')
fig.savefig('images/subplotsNoError.png', bbob_inches='tight')

