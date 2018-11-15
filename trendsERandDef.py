""" 
Time Trends in 5 variables: 
Early repayment rate, default rate, choice of loan-terms (ratio), dti, income.
aggregated at annual level (takes about 1 hour on t2large).
"""
from __future__ import print_function

from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F

import numpy as np
import pandas as pd

#if __name__ == "__main__":

spark = SparkSession \
	.builder \
	.appName("timeTrends") \
	.config("spark.some.config.option", "some-value") \
	.getOrCreate()

df = spark.read.load("data/smallLS3bSample.csv",format="csv", sep=",", inferSchema="true", header="true")
#df = spark.read.load("data/LSmerged.csv",format="csv", sep=",", inferSchema="true", header="true")
			
df = df.drop(
	"id","member_id","funded_amnt","funded_amnt_inv","emp_title"\
	"verification_status","pymnt_plan","url","purpose","title",\
	"delinq_2yrs","earliest_cr_line","inq_last_6mths","mths_since_last_delinq","mths_since_last_record",\
	"open_acc","pub_rec","revol_bal","total_acc","initial_list_status","out_prncp","out_prncp_inv",\
	"total_pymnt_inv","total_rec_prncp","total_rec_int","total_rec_late_fee","recoveries",\
	"collection_recovery_fee","next_pymnt_d","last_credit_pull_d",\
	"collections_12_mths_ex_med","mths_since_last_major_derog","policy_code",\
	"annual_inc_joint","dti_joint","verification_status_joint","acc_now_delinq","tot_coll_amt",\
	"tot_cur_bal","open_acc_6m","open_act_il","open_il_12m","open_il_24m","mths_since_rcnt_il",
	"total_bal_il","il_util","open_rv_12m","open_rv_24m","max_bal_bc","all_util","total_rev_hi_lim",\
	"inq_fi","total_cu_tl","inq_last_12m","acc_open_past_24mths","avg_cur_bal","bc_open_to_buy",\
	"bc_util","chargeoff_within_12_mths","delinq_amnt","mo_sin_old_il_acct","mo_sin_old_rev_tl_op",\
	"mo_sin_rcnt_rev_tl_op","mo_sin_rcnt_tl","mort_acc","mths_since_recent_bc","mths_since_recent_bc_dlq",\
	"mths_since_recent_inq","mths_since_recent_revol_delinq","num_accts_ever_120_pd",\
	"num_actv_bc_tl","num_actv_rev_tl","num_bc_sats","num_bc_tl","num_il_tl","num_op_rev_tl",\
	"num_rev_accts","num_rev_tl_bal_gt_0","num_sats","num_tl_120dpd_2m","num_tl_30dpd",\
	"num_tl_90g_dpd_24m","num_tl_op_past_12m","pct_tl_nvr_dlq","percent_bc_gt_75","pub_rec_bankruptcies",\
	"tax_liens","tot_hi_cred_lim","total_bal_ex_mort","total_bc_limit","total_il_high_credit_limit",\
	"revol_bal_joint","sec_app_earliest_cr_line","sec_app_inq_last_6mths","sec_app_mort_acc","sec_app_open_acc",\
	"sec_app_revol_util","sec_app_open_act_il","sec_app_num_rev_accts","sec_app_chargeoff_within_12_mths",\
	"sec_app_collections_12_mths_ex_med","sec_app_mths_since_last_major_derog","hardship_flag","hardship_type",\
	"hardship_reason","hardship_status","deferral_term","hardship_amount","hardship_start_date",\
	"hardship_end_date","payment_plan_start_date","hardship_length","hardship_dpd","hardship_loan_status",\
	"orig_projected_additional_accrued_interest","hardship_payoff_balance_amount","hardship_last_payment_amount",\
	"disbursement_method","debt_settlement_flag","debt_settlement_flag_date","settlement_status","settlement_date",\
	"settlement_amount","settlement_percentage","settlement_term")
#kept: "loan_amnt", "issue_d","term","int_rate","grade","sub_grade","home_ownership","annual_inc","application_type", "loan_status",
	# "desc","addr_state","emp_length", "dti", "revol_util","total_pymnt", "last_pymnt_d","installment","last_pymnt_amnt"
df=df.drop('last_pymnt_amnt', 'application_type','total_pymnt', 'emp_title','verification_status',"zip_code", 'desc',
	'installment','sub_grade', 'emp_length', 'home_ownership',  'desc', 'revol_util')
# df=df.drop('loan_amnt', 'int_rate', )

df.count()  
#1873279   #nearly 2 million records  


#initial cleanup
##get rid of "does not meet ..." just  keep paid or charged off using Java regex
## use 1 for default and 0 for repaid, 2 for current
df = df.withColumn('loan_status', F.regexp_replace('loan_status', '.*Paid$' , '0').alias('loan_status'))
df = df.withColumn('loan_status', F.regexp_replace('loan_status', '.*Off$' , '1').alias('loan_status'))
df = df.withColumn('loan_status', F.regexp_replace('loan_status', '(.*Current$)|( In Grace Period)' , '2').alias('loan_status'))
df = df.withColumn('loan_status', F.regexp_replace('loan_status', '(Late \(16-30 days\))|(Late \(31-120 days\))' , '2').alias('loan_status'))


#string clean up, remove %, strip months away from term
df = df.withColumn('int_rate', F.regexp_replace('int_rate', '%' , '').alias('int_rate'))
#df = df.withColumn('revol_util', F.regexp_replace('revol_util', '%' , '').alias('revol_util'))
df = df.withColumn('term', F.regexp_replace('term', '( months)|( )' , '').alias('term'))	


#convert strings to float
df = df.withColumn('loan_status', df['loan_status'].cast(DoubleType()).alias('loan_status'))
#df = df.withColumn('revol_util',df['revol_util'].cast(DoubleType()).alias('revol_util'))
df = df.withColumn('dti',df['dti'].cast(DoubleType()).alias('dti'))
df = df.withColumn('int_rate',df['int_rate'].cast(DoubleType()).alias('int_rate'))
df = df.withColumn('term2', df['term'].cast(DoubleType()).alias('term2'))

#and handle date strings which are Mon-YYYY format
months={'Jan':'01-01','Feb':'02-15','Mar':'03-15','Apr':'04-15','May':'05-15','Jun':'06-15',\
	'Jul':'07-15','Aug':'08-15','Sep':'09-15','Oct':'10-15','Nov':'11-15','Dec':'12-31'}

for i in months:
	df = df.withColumn('issue_d', F.regexp_replace('issue_d', i , months[i]).alias('issue_d'))
	df = df.withColumn('last_pymnt_d', F.regexp_replace('last_pymnt_d', i , months[i]).alias('last_pymnt_d'))


df = df.withColumn('issue_d', (F.to_date('issue_d', 'MM-dd-yyyy')).alias('issue_d'))
df = df.withColumn('last_pymnt_d', (F.to_date('last_pymnt_d', 'MM-dd-yyyy')).alias('last_pymnt_d'))
df = df.withColumn('mnth_start2last', F.months_between(df.last_pymnt_d, df.issue_d).alias('mnth_start2last'))
df = df.withColumn('fracNumPmts', F.col('mnth_start2last')/F.col('term2'))
	
#Indicators for default, early repayment	


df2 = df #keep a backup
#then drop rows with leftover na's
df = df.na.drop(how='any')

#for large sample size, use a subsample
df = df.sample(fraction=0.1)

#if df.count()>0:
#	del(df2) #if df is okay (did not lose all data!)


#Pick a time period
## start and end dates

start_date = '2012-01-01'
end_date = '2013-12-31' #use 1+, so to include Dec, use 2014-01-01 what is actually needed

#In any time period, 

#(ER) Early repayments in that period: last payment must be within start and end period.
# (D) Same with early default: 
# (C) For those that are current, the last payment must be after the end, but the issue date must be before the end (not start)
# (S) what's left? Loans that were repaid as per schedule.

#ER rate = ER/(ER+D + C + S)
#Def rate = D/(ER+D + C + S)

# ER = df.filter(df.loan_status==0).filter(df.fracNumPmts<1).filter((df.last_pymnt_d >= start_date)&(df.last_pymnt_d < end_date)).count()
# Def = df.filter(df.loan_status==1).filter((df.last_pymnt_d >= start_date)&(df.last_pymnt_d < end_date)).count()
# Cur = df.filter((df.last_pymnt_d >= end_date)&(df.issue_d < end_date)).count() 
# Sch = df.filter(df.loan_status==0).filter(df.fracNumPmts>=1).filter((df.last_pymnt_d >= start_date)&(df.last_pymnt_d < end_date)).count()
# 
# ER_rate = ER/(ER+Def+Cur+Sch)
# Def_rate = Def/(ER+Def+Cur+Sch)

#NOTE: this finishes a month before end date, so end date should be picked carefully 1+
date_range= np.arange(start_date, end_date, dtype='datetime64[M]')
#date_range= np.arange('2010-01-01', '2018-04-01', dtype='datetime64[M]')
date_range=str(date_range).replace('[','').replace(']','').split(' ')
for i in range(len(date_range)):
	date_range[i] = date_range[i].replace("'","").replace("\n","")
	date_range[i] += '-01'
	date_range[i].replace('-12-01','-12-31')
## either test or convert strings to Spark datetime to be sure its interpreted correctly

#
yearIdx = np.arange(0,len(date_range), 12)

timeLine = date_range[0:len(yearIdx)]
for i in range(len(yearIdx)):
	timeLine[i] = date_range[yearIdx[i]]
	
#timeLine = ['2012-01-01', '2012-04-01', '2012-07-01']#, '2012-10-01','2013-01-01','2013-04-01','2013-07-01','2013-10-01','2014-01-01']

num_periods=len(timeLine)-1


############
#Using groupBy instead should be faster
#obtain schema from merge: merge defined below
#merge.schema
schema=StructType(List(
	StructField(defcount,LongType,false),
	StructField(state,StringType,true),
	StructField(ERcount,LongType,false),
	StructField(Schcount,LongType,false),
	StructField(Curcount,LongType,false),
	StructField(DTI,DoubleType,true),
	StructField(income,DoubleType,true),
	StructField(shortcount,LongType,false),
	StructField(longcount,LongType,false),
	StructField(Period,StringType,false),
	StructField(ER_rate,DoubleType,true),
	StructField(def_rate,DoubleType,true),
	StructField(termRatio,DoubleType,true)))

#create empty dataframe to start
mainDF = spark.createDataFrame([], schema)

for i in range(0, len(timeLine)-1):
	start_date = timeLine[i]
	end_date = timeLine[i+1]
	#print(start_date, end_date)
	#print('National for time {}'.format(start_date))
	print('Filtering results by time period')
	dft2 = df.filter((df.last_pymnt_d >= start_date)&(df.last_pymnt_d < end_date))
	dft2.columns
	#count defaults in this period
	print('Counting defaults') 
	Def = dft2.filter(df.loan_status==1).groupBy(df.addr_state).count()
	Def = Def.withColumnRenamed('count', 'defcount')
	
	#filter by repaid in this period
	print('Filtering repaid loans')
	dft2a = dft2.filter(df.loan_status==0)
	#count early repayment and those paid per schedule separately
	print('count early and scheduled repayments')
	ER = dft2a.filter(df.fracNumPmts<1).groupBy(df.addr_state).count()
	ER = ER.withColumnRenamed('count', 'ERcount')
		
	Sch = dft2a.filter(df.fracNumPmts>=1).groupBy(df.addr_state).count()
	Sch = Sch.withColumnRenamed('count', 'Schcount')
	#current, not repaid, not defaulted
	print('Count all other outstanding loans')
	Cur = df.filter((df.last_pymnt_d >= end_date)&(df.issue_d < end_date)).groupBy(df.addr_state).count()
	Cur = Cur.withColumnRenamed('count', 'Curcount')
	#rates
	#ER_rate[i] = ER/(ER+Def+Cur+Sch)
	#print(ER_rate[i,j])
	#Def_rate[i] = Def/(ER+Def+Cur+Sch)
	#print(Def_rate[i,j])
	
	#term choices
	#filter the state results for only loans issued in this period
	print('Filter loans issued in this period')
	dft3 = df.filter((df.issue_d >= start_date)&(df.issue_d < end_date))
	print('Count issued numbers separately by term')
	short = dft3.filter(df.term=='36').groupBy(df.addr_state).count()
	short = short.withColumnRenamed('count', 'shortcount')
	
	long = dft3.filter(df.term=='60').groupBy(df.addr_state).count()
	long = long.withColumnRenamed('count', 'longcount')
	#term_ratio[i] = long/short
	print('Get mean income and DTI for new loans in this period')
	temp=dft3.groupBy(df.addr_state).agg(F.mean('dti'), F.mean('annual_inc'))
	temp=temp.withColumnRenamed('avg(dti)', 'DTI')
	temp=temp.withColumnRenamed('avg(annual_inc)', 'income')
	#dti_ratio[i] = temp.take(1)[0][0]
	#income[i] = temp.take(1)[0][1]
	
	print('merging frames')
	merge = Def.join(ER, Def.addr_state == ER.addr_state, 'inner').drop(ER.addr_state)
	merge = merge.join(Sch,merge.addr_state == Sch.addr_state, 'inner').drop(Sch.addr_state)
	merge = merge.join(Cur,merge.addr_state == Cur.addr_state, 'inner').drop(Cur.addr_state)
	merge = merge.join(temp,merge.addr_state == temp.addr_state, 'inner').drop(temp.addr_state)
	merge = merge.join(short,merge.addr_state == short.addr_state, 'inner').drop(short.addr_state)
	merge = merge.join(long,merge.addr_state == long.addr_state, 'inner').drop(long.addr_state)
	merge = merge.withColumn('Period', F.lit(start_date))
	merge = merge.withColumn('ER_rate', F.col('ERcount')/(F.col('ERcount')+F.col('defcount')+F.col('Schcount')+F.col('Curcount')))
	merge = merge.withColumn('Def_rate', F.col('defcount')/(F.col('ERcount')+F.col('defcount')+F.col('Schcount')+F.col('Curcount')))
	merge = merge.withColumn('termRatio', F.col('longcount')/F.col('shortcount'))
	merge = merge.withColumnRenamed('addr_state', 'state')
	
	#keep track of progress? This will slow things down, but can kill process and still have something
	print('converting to pandas and to_csv'..)
	merge.toPandas().to_csv('data/merge'+i+'.csv', index=False)
	
	mainDF = mainDF.union(merge)
	print('next iteration..')


pdData = mainDF.toPandas()
#check column names
pdData.columns
#if required then change: pdData.columns.
pdData.to_csv('data/resultsState.csv', index = False) #upload to S3



spark.stop()


############# SLOWER METHODS using loops over states (and grades)

#grouping by grades
gradeList = ['A','B','C']#,'D', 'E','F','G']
num_grades = len(gradeList)

ER_rate =  np.zeros(num_periods*num_grades).reshape(num_periods,num_grades)
Def_rate =  np.zeros(num_periods*num_grades).reshape(num_periods,num_grades)
term_ratio = np.zeros(num_periods*num_grades).reshape(num_periods,num_grades)

for (i,j) in [(i,j) for i in range(num_periods) for j in range(num_grades)]:
	
	dft1 = df.filter(df.grade==gradeList[j])
	start_date = timeLine[i]
	end_date = timeLine[i+1]
	print('For time {} and grade {}'.format(start_date, gradeList[j]))
	
	dft2 = dft1.filter((df.last_pymnt_d >= start_date)&(df.last_pymnt_d < end_date))
	
	Def = dft2.filter(df.loan_status==1).count()
	ER = dft2.filter(df.loan_status==0).filter(df.fracNumPmts<1).count()
	Sch = dft2.filter(df.loan_status==0).filter(df.fracNumPmts>=1).count()
	Cur = dft1.filter((df.last_pymnt_d >= end_date)&(df.issue_d < end_date)).count() 
	#rates
	ER_rate[i,j] = ER/(ER+Def+Cur+Sch)
	print(ER_rate[i,j])
	Def_rate[i,j] = Def/(ER+Def+Cur+Sch)
	print(Def_rate[i,j])
	#term choices
	dft3 = dft1.filter((df.issue_d >= start_date)&(df.issue_d < end_date))
	short = dft3.filter(df.term=='36').count()
	long = dft3.filter(df.term=='60').count()
	term_ratio[i,j] = long/short

#add time as index or date column and also column headers for grades and save to csv	
er_grade = pd.DataFrame(ER_rate,index = pd.to_datetime(timeLine[:num_periods]),columns = gradeList)
er_grade.to_csv('data/er_grade.csv', index=True)

def_grade = pd.DataFrame(Def_rate,index = pd.to_datetime(timeLine[:num_periods]),columns = gradeList)
def_grade.to_csv('data/def_grade.csv', index=True)

term_ratio_grade = pd.DataFrame(term_ratio,index = pd.to_datetime(timeLine[:num_periods]),columns = gradeList)
term_ratio_grade.to_csv('data/term_ratio_grade.csv', index=True)

 
##Grouping by state (panel data)
# states=list(df.groupBy('addr_state').count().sort('count', ascending=False).select('addr_state').collect())
# a = str(states)
# a = a.replace("[","").replace("]","").replace('Row(addr_state=',"").replace(")","").replace("'","").replace("'","")
# states = a.split(', ') #comma followed by space
# del(a)
states = ['CA']#, 'TX', 'NY', 'FL', 'IL', 'NJ', 'GA', 'OH', 'PA', 'NC']
 		  #'VA', 'AZ', 'MI', 'MD', 'MA', 'CO', 'WA', 'IN', 'TN', 'CT',
#  		  ' MN', 'MO', 'NV', 'WI', 'SC', 'OR', 'AL', 'LA', 'OK', 'KY',
#  		  ' AR', 'KS', 'UT', 'MS', 'WV', 'NH', 'NM', 'RI', 'NE', 'HI',
#  		  ' ME', 'ID', 'DE', 'MT', 'VT', 'ND', 'AK', 'DC', 'SD', 'WY']

num_states = len(states)

ER_rate =  np.zeros(num_periods*num_states).reshape(num_periods,num_states)
Def_rate =  np.zeros(num_periods*num_states).reshape(num_periods,num_states)
term_ratio = np.zeros(num_periods*num_states).reshape(num_periods,num_states)
dti_ratio = np.zeros(num_periods*num_states).reshape(num_periods,num_states)
income = np.zeros(num_periods*num_states).reshape(num_periods,num_states)

#for (i,j) in [(i,j) for i in range(num_periods) for j in range(num_states)]:
#better to separate the loops to avoid filtering multiple times
for j in range(num_states):
	print('Filtering results by state = {}'.format(states[j]))
	dft1 = df.filter(df.addr_state==states[j])
	for i in range(num_periods):
		start_date = timeLine[i]
		end_date = timeLine[i+1]
		print('From {} to {} for  {}'.format(start_date,end_date, states[j]))
		#filter all observations by state
	
		#filter further by time period of interest
		print('Filtering results by time period')
		dft2 = dft1.filter((df.last_pymnt_d >= start_date)&(df.last_pymnt_d < end_date))
	
		#count defaults in this period
		print('Counting defaults') 
		Def = dft2.filter(df.loan_status==1).count()
	
		#filter by repaid in this period
		print('Filtering repaid loans')
		dft2a = dft2.filter(df.loan_status==0)
		#count early repayment and those paid per schedule separately
		print('count early and scheduled repayments')
		ER = dft2a.filter(df.fracNumPmts<1).count()
		Sch = dft2a.filter(df.fracNumPmts>=1).count()
		#current, not repaid, not defaulted
		print('Count all other outstanding loans')
		Cur = dft1.filter((df.last_pymnt_d >= end_date)&(df.issue_d < end_date)).count()
		#rates
		ER_rate[i,j] = ER/(ER+Def+Cur+Sch)
		#print(ER_rate[i,j])
		Def_rate[i,j] = Def/(ER+Def+Cur+Sch)
		#print(Def_rate[i,j])
	
		#term choices
		#filter the state results for only loans issued in this period
		print('Filter loans issued in this period')
		dft3 = dft1.filter((df.issue_d >= start_date)&(df.issue_d < end_date))
		print('Count issued numbers separately by term')
		short = dft3.filter(df.term=='36').count()
		long = dft3.filter(df.term=='60').count()
		term_ratio[i,j] = long/short
		print('Get mean income and DTI for new loans in this period')
		temp=dft3.agg(F.mean('dti'), F.mean('annual_inc'))
		dti_ratio[i,j] = temp.take(1)[0][0]
		income[i,j] = temp.take(1)[0][1]
		print('next iteration..')
	
#add time as index or date column and also column headers for states 	
er_state = pd.DataFrame(ER_rate,index = pd.to_datetime(timeLine[:num_periods]),columns = states)
def_state = pd.DataFrame(Def_rate,index = pd.to_datetime(timeLine[:num_periods]),columns = states)
term_ratio_state = pd.DataFrame(term_ratio,index = pd.to_datetime(timeLine[:num_periods]),columns = states)
dti_state = pd.DataFrame(dti_ratio,index = pd.to_datetime(timeLine[:num_periods]),columns = states)
inc_state = pd.DataFrame(income,index = pd.to_datetime(timeLine[:num_periods]),columns = states)

#same thing, but now national
ER_rate =  np.zeros(num_periods)
Def_rate =  np.zeros(num_periods)
term_ratio = np.zeros(num_periods)
dti_ratio = np.zeros(num_periods)
income = np.zeros(num_periods)

for i in range(num_periods):
	start_date = timeLine[i]
	end_date = timeLine[i+1]
	print('National for time {}'.format(start_date))
	print('Filtering results by time period')
	dft2 = df.filter((df.last_pymnt_d >= start_date)&(df.last_pymnt_d < end_date))
	
	#count defaults in this period
	print('Counting defaults') 
	Def = dft2.filter(df.loan_status==1).count()
	
	#filter by repaid in this period
	print('Filtering repaid loans')
	dft2a = dft2.filter(df.loan_status==0)
	#count early repayment and those paid per schedule separately
	print('count early and scheduled repayments')
	ER = dft2a.filter(df.fracNumPmts<1).count()
	Sch = dft2a.filter(df.fracNumPmts>=1).count()
	#current, not repaid, not defaulted
	print('Count all other outstanding loans')
	Cur = df.filter((df.last_pymnt_d >= end_date)&(df.issue_d < end_date)).count()
	#rates
	ER_rate[i] = ER/(ER+Def+Cur+Sch)
	#print(ER_rate[i,j])
	Def_rate[i] = Def/(ER+Def+Cur+Sch)
	#print(Def_rate[i,j])
	
	#term choices
	#filter the state results for only loans issued in this period
	print('Filter loans issued in this period')
	dft3 = df.filter((df.issue_d >= start_date)&(df.issue_d < end_date))
	print('Count issued numbers separately by term')
	short = dft3.filter(df.term=='36').count()
	long = dft3.filter(df.term=='60').count()
	term_ratio[i] = long/short
	print('Get mean income and DTI for new loans in this period')
	temp=dft3.agg(F.mean('dti'), F.mean('annual_inc'))
	dti_ratio[i] = temp.take(1)[0][0]
	income[i] = temp.take(1)[0][1]
	print('next iteration..')

		
#add national levels to state pandas DFs	
er_state['US'] = ER_rate
def_state['US'] = Def_rate
term_ratio_state['US'] = term_ratio
dti_state['US'] = dti_ratio
inc_state['US'] = income

#save to CSV files
er_state.to_csv('data/er_state.csv', index=True)
def_state.to_csv('data/def_state.csv', index=True)
term_ratio_state.to_csv('data/term_ratio_state.csv', index=True)
dti_state.to_csv('data/dti_state.csv', index=True)
inc_state.to_csv('data/inc_state.csv', index=True)





spark.stop()
