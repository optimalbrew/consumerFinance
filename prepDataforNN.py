""" 
Prepare dataset for multivariate analysis using neural nets
	e.g. modelLendNN.py with target variables
	* y1: probability of defaulting on a loan
	* y2: timing of default (conditional on above)
	* y3: probability of repaying a loan early
	* y4: timing of early repayment (conditional on above)
"""
from __future__ import print_function

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StringIndexer
from pyspark.ml.feature import Word2Vec
from pyspark.ml.clustering import KMeans	

from pyspark.sql.types import *
import pyspark.sql.functions as F


import pandas as pd
import matplotlib.pyplot as plt

#if __name__ == "__main__":

spark = SparkSession \
	.builder \
	.appName("Linear Regression") \
	.config("spark.some.config.option", "some-value") \
	.getOrCreate()

df = spark.read.load("data/smallLS3bSample.csv",format="csv", sep=",", inferSchema="true", header="true")
#df = spark.read.load("data/LoanStats3b.csv",format="csv", sep=",", inferSchema="true", header="true")
			

df = df.drop(
	"id","member_id","funded_amnt","funded_amnt_inv","emp_title","sub_grade"\
	"verification_status","pymnt_plan","url","purpose","title","application_type",\
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
	"settlement_amount","settlement_percentage",'total_pymnt', 'last_pymnt_amnt',"settlement_term")

#kept: "loan_amnt", "issue_d","term","int_rate","grade","home_ownership","annual_inc","application_type", "loan_status",
	# "desc","emp_length", "dti", "revol_util", "last_pymnt_d","installment",'total_pymnt', 'last_pymnt_amnt',


#df.count()


#initial cleanup
print("initial cleanup\n")

##get rid of "does not meet ..." just  keep paid or charged off using Java regex
## use 1 for default and 0 for repaid
df = df.withColumn('loan_status', F.regexp_replace('loan_status', '.*Paid$' , '0').alias('loan_status'))
df = df.withColumn('loan_status', F.regexp_replace('loan_status', '.*Off$' , '1').alias('loan_status'))

#string clean up, remove %, strip months away from term
df = df.withColumn('int_rate', F.regexp_replace('int_rate', '%' , '').alias('int_rate'))
df = df.withColumn('revol_util', F.regexp_replace('revol_util', '%' , '').alias('revol_util'))
df = df.withColumn('term', F.regexp_replace('term', '( months)|(' ')' , '').alias('term'))	


#convert strings to float
df = df.withColumn('loan_status', df['loan_status'].cast(DoubleType()).alias('loan_status'))
df = df.withColumn('revol_util',df['revol_util'].cast(DoubleType()).alias('revol_util'))
df = df.withColumn('dti',df['dti'].cast(DoubleType()).alias('dti'))
df = df.withColumn('int_rate',df['int_rate'].cast(DoubleType()).alias('int_rate'))
df = df.withColumn('term2', df['term'].cast(DoubleType()).alias('term2'))

#define ratio of installment to income ratio
df = df.withColumn('inst2inc', F.col('installment')/F.col("annual_inc"))

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


#Optional
## start and end dates based on last payment date:  only when using large samples
#start_date = '2011-01-01'
#end_date = '2013-12-31'

#Early repayments between certain dates
#df = df.filter((df.last_pymnt_d >= start_date)&(df.last_pymnt_d < end_date))
#df.count()

print('\n\n\nWord2vec...\n')
		
#for tokenizer change null to none
df = df.na.fill({'desc': 'none' }) #perhaps doesn't matter?

#replace
df = df.withColumn('desc', F.regexp_replace('desc', '(Borrower added on [0-9][0-9]/[0-9][0-9]/[0-9][0-9] >)|<br>|<br/>' , '').alias('desc'))
#take a look to verify
#df.select('desc').show(3,truncate=False)

#split the doc strings, or use tokenizer
regexTokenizer = RegexTokenizer(inputCol="desc", outputCol="words", pattern="\\W")	
df = regexTokenizer.transform(df)

#3D vector space
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="words", outputCol="result")
#Fit to find word embeddings
modelW2V = word2Vec.fit(df)

#Use the embeddings to transform, with the vector for "words" in the column "result" 
df = modelW2V.transform(df)
#rows without any comments NULL -> marked 'none' will share the same vector
#df.select('desc','result').show(10,truncate=True) # set to false for large vector space

#cluster the result
kmeans = KMeans(k=3, seed=1, featuresCol="result", predictionCol="pred_KM")
modelKM = kmeans.fit(df)
df = modelKM.transform(df)

#For regression, need to treat the predicted class as categorical variable, not integer
df = df.withColumn('pred_KM', df['pred_KM'].cast(StringType()).alias('pred_KM'))

print('\n\n\nString indexation for categorical variables.. \n')

#numerical vars for neural network
float_x_vars = ["loan_amnt", "int_rate","annual_inc", "dti", "revol_util", "installment", "inst2inc" ]
##StringEncoding of categorical variables
cat_x_vars = ["term","grade","home_ownership", "pred_KM","emp_length"]

#df2 = df #backup in case of trouble

for cat_var in cat_x_vars:
	df = StringIndexer(inputCol= cat_var, outputCol= cat_var +'Idx').fit(df).transform(df).drop(cat_var)
	df = df.withColumnRenamed(cat_var +'Idx',cat_var)

#df.select(cat_x_vars).show(5) #check

##Create y or target variables for neural networks
#probability/indicator for default
df = df.withColumn('probDef',F.when(df['loan_status']==1,1.0).otherwise(0.0)) #default is 1, repaid is 0
#indicator for early replayment
df = df.withColumn('probER',F.when((df['loan_status']==0)&(df['fracNumPmts']<1),1.0).otherwise(0.0))
#indicator for on-schedule repayment can be inferred as probDef=probER=0,0, with 

#visually:
#plot of timing of either default or eventual (not early repayment)
#df.filter((df['loan_status']==1)|(df.fracNumPmts >=1)).select(df.fracNumPmts).toPandas().plot.hist()
#plt.show()  #This is  bi-modal, mostly low over 0,1 and then a spike at 1.

#plot of timing of either repayment (whenever)
#df.filter(df['loan_status']==0).select(df.fracNumPmts).toPandas().plot.hist()
#plt.show() #This is more like a uniform over 0,1 + a spike at 1.

#Explore the issue some more
avgDefTime = df.filter( (df['loan_status']==1)&(df.fracNumPmts <1)).agg(F.mean(df.fracNumPmts)).take(1)[0][0]#.show()
#0.4437902647746959  , when default does happen

DefCount = df.filter( (df['loan_status']==1)&(df.fracNumPmts <1)).count()
#1550
#df.filter( (df['loan_status']==1)&(df.fracNumPmts >=1)).count()
#7   
avgERTime = df.filter( (df['loan_status']==0)&(df.fracNumPmts <1)).agg(F.mean(df.fracNumPmts)).take(1)[0][0]
#0.5443095394875822
ERcount = df.filter( (df['loan_status']==0)&(df.fracNumPmts <1)).count()
#5145                                                                            
onTimeCount = df.filter( (df['loan_status']==0)&(df.fracNumPmts >=1)).count() 
#2581 

#Summary for counterfactuals. 
#* When ER is observed, it happens at 0.54 mark (mean) #freq (5145). 
#* When default is observed, the average mark is 0.44 and freq (1550)
#* Not ER or not default implies repaid, possibly late: mean is 1.005, freq (2581)

#Thus:
## Not ER can be abstracted by time .44 w.p. 1550/(1550+2581) ~ 0.375 or 1 otherwise 
	# AssignedTimeNotDef = avgERTime w.p. (ERcount/(ERcount+onTimeCount)) or 1 o/w
## Not default can be abstracted by time 0.54 w.p. 5145/(5145+2581) ~ .666  or 1 o/w
	# AssignedTimeNotER = avgDefTime w.p. (Defcount/(Defcount+onTimeCount)) or 1 o/w

thresholdNotER = DefCount/(DefCount+onTimeCount) #avgDefTime is less than threshold. 

thresholdNotDef =  ERcount/(ERcount+onTimeCount) #avgERTime is less than threshold

#draw random values 
df = df.withColumn('rand', F.rand(seed=7))

df = df.withColumn('randTimeNotDef', F.when(df.rand > thresholdNotDef, 1).otherwise(avgERTime))
df = df.withColumn('randTimeNotER', F.when(df.rand > thresholdNotER, 1).otherwise(avgDefTime))



#Average values for "Counter factuals." Use take(1)[0][0] to extract value from dataframe
#average time when Def!=1 (repaid whenever)
# avgTimeNotDef = df.filter(df['loan_status']==0).agg(F.mean(df.fracNumPmts)).take(1)[0][0]#.show()
# #average time taken when when ER!=1 (either defaulted or repaid on time or eventually)
# avgTimeNotER = df.filter((df['loan_status']==1)|(df.fracNumPmts >=1)).agg(F.mean(df.fracNumPmts)).take(1)[0][0]#.show
# 
# sdevTimeNotDef = df.filter(df['loan_status']==0).agg(F.stddev(df.fracNumPmts)).take(1)[0][0]#.show()
# #average time taken when when ER!=1 (either defaulted or repaid on time or eventually)
# sdevTimeNotER = df.filter((df['loan_status']==1)|(df.fracNumPmts >=1)).agg(F.stddev(df.fracNumPmts)).take(1)[0][0]#.show

#random times for counterfactuals
# low_randTimeNotDef = avgTimeNotDef - sdevTimeNotDef #lower bound
# low_randTimeNotER = avgTimeNotER - sdevTimeNotER #lower bound
# scale_randTimeNotDef = 1-low_randTimeNotDef
# scale_randTimeNotER = 1-low_randTimeNotER
#uniform rand from mean-1sd to 1 (upper bound is 1)


#period when default occurs (0 in case of repayment)
df = df.withColumn('timeDef', F.col('probDef')*F.col('fracNumPmts') + F.col('randTimeNotDef')*(1-F.col('probDef'))) #1 if not default
#period when early repayment occurs (0 when default or)
df = df.withColumn('timeER', F.col('probER')*F.col('fracNumPmts') + F.col('randTimeNotER')*(1-F.col('probER'))) #1 if not repaid early


y_vars = ['probDef','timeDef', 'probER', 'timeER', 'fracNumPmts']

#df.select(['loan_status','fracNumPmts'] + y_vars).show(5)

#df2 = df #keep a backup
#df2.count()

#then drop rows with leftover na's
#df = df.na.drop(how='any')
#df.count() #if loss is big, investigate and fill.na as needed

#otherwise, remove df2
#del(df2)

#export to csv (via coalesce)

print('\n\n\nGetting ready to write data to csv\n')

#df = df.select(float_x_vars + cat_x_vars + y_vars)
#df = df.na.drop(how='any')
#df.coalesce(1).write.csv('data/pdDataNN.csv') 

#Using pandas
pdData = df.select(float_x_vars + cat_x_vars + y_vars)
pdData = pdData.na.drop(how='any')

#del(df)

#pdData.count()
#If there is a large loss, then investigate why
pdData = pdData.toPandas()
pdData.to_csv('data/pdDataNN.csv', index=False)

del(pdData)

spark.stop()
