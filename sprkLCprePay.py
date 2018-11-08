""" 
Default and Prepayment behavior:
Summarization by credit grade.

"""
from __future__ import print_function

from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F

import pandas as pd

#if __name__ == "__main__":

spark = SparkSession \
	.builder \
	.appName("Linear Regression") \
	.config("spark.some.config.option", "some-value") \
	.getOrCreate()

df = spark.read.load("data/LoanStats3b.csv",format="csv", sep=",", inferSchema="true", header="true")
			

df = df.drop(
	"id","member_id","funded_amnt","funded_amnt_inv","emp_title",\
	"verification_status","pymnt_plan","url","purpose","title","zip_code","addr_state",\
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
	# "desc","emp_length", "dti", "revol_util","total_pymnt", "last_pymnt_d","installment","last_pymnt_amnt"


df.count()
#42538 #

#initial cleanup

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

#and handle date strings which are Mon-YYYY format
months={'Jan':'01-01','Feb':'02-15','Mar':'03-15','Apr':'04-15','May':'05-15','Jun':'06-15',\
	'Jul':'07-15','Aug':'08-15','Sep':'09-15','Oct':'10-15','Nov':'11-15','Dec':'12-31'}

for i in months:
	df = df.withColumn('issue_d', F.regexp_replace('issue_d', i , months[i]).alias('issue_d'))
	df = df.withColumn('last_pymnt_d', F.regexp_replace('last_pymnt_d', i , months[i]).alias('last_pymnt_d'))


df = df.withColumn('issue_d', (F.to_date('issue_d', 'MM-dd-yyyy')).alias('issue_d'))
df = df.withColumn('last_pymnt_d', F.to_date('last_pymnt_d', 'MM-dd-yyyy'))
df = df.withColumn('mnth_start2last', F.months_between(df.last_pymnt_d, df.issue_d).alias('mnth_start2last'))
df = df.withColumn('fracNumPmts', F.col('mnth_start2last')/F.col('term2'))
		
#for tokenizer change null to none
df = df.na.fill({'desc': 'none' }) #perhaps doesn't matter?


#df = df.drop('sub_grade', 'total_pymnt', 'last_pymnt_amnt', 'application_type')

df2 = df #keep a backup
#then drop rows with leftover na's

df = df.drop("loan_amnt", "int_rate","sub_grade","home_ownership","annual_inc","application_type",
	 "desc","emp_length", "dti", "revol_util","total_pymnt","installment","last_pymnt_amnt", "issue_d",
	 	)

df = df.na.drop(how='any')

if df.count()>0:
	del(df2) #if df is okay (did not lose all data!)

df = df.drop('issue_d','last_pymnt_d','mnth_start2last')

#Loan default rates (mean and stddev) by loan grade
#mean
def36 = df.filter(df.term2==36).groupBy(df.grade).agg(F.mean(df.loan_status).alias('def36mean'),
														F.stddev(df.loan_status).alias('def36sd'))

def60 = df.filter(df.term2==60).groupBy(df.grade).agg(F.mean(df.loan_status).alias('def60mean'),
														F.stddev(df.loan_status).alias('def60sd'))
#stats on early repayment, compared to all loans (not just those that were repaid)
df = df.withColumn('ER',F.when((df.fracNumPmts<1)&(df.loan_status==0),1).otherwise(0)) #create an indicator for repayment

ER36 = df.filter(df.term2==36).groupBy(df.grade).agg(F.mean(df.ER).alias('ER36mean'), F.stddev(df.ER).alias('ER36sd'))
#ER36.sort('grade').show()
# +-----+-------------------+-------------------+                                 
# |grade|           ER36mean|             ER36sd|
# +-----+-------------------+-------------------+
# |    A| 0.5719117699974812|0.49481057718338295|
# |    B| 0.5505616008691233| 0.4974413135831339|
# |    C|  0.528578874218207|0.49918980691847653|
# |    D|0.49854093073260636| 0.5000106704025198|
# |    E| 0.4868519909842224|0.49988969963910895|
# |    F| 0.4368794326241135|0.49635193988434306|
# |    G| 0.4864864864864865| 0.5067117097095317|
# +-----+-------------------+-------------------+
ER60 = df.filter(df.term2==60).groupBy(df.grade).agg(F.mean(df.ER).alias('ER60mean'), F.stddev(df.ER).alias('ER60sd'))
#ER60.sort('grade').show()
# +-----+------------------+-------------------+                                  
# |grade|          ER60mean|             ER60sd|
# +-----+------------------+-------------------+
# |    A|0.7861356932153393|0.41033486976071437|
# |    B|0.7346020053202373| 0.4415900828240326|
# |    C|0.6888206204106099|0.46299334256719205|
# |    D| 0.629277566539924| 0.4830310492302113|
# |    E|0.5855280659749933| 0.4926634091801306|
# |    F|0.5581901239939091|  0.496656388107999|
# |    G|0.5482233502538071| 0.4979218916924311|
# +-----+------------------+-------------------+


defTime36 =  df.filter(df.term2==36).filter(df.loan_status==1).groupBy(df.grade).agg(F.mean(df.fracNumPmts).alias('defTime36mean'),
																	F.stddev(df.fracNumPmts).alias('defTime36sd'))
defTime60 =  df.filter(df.term2==60).filter(df.loan_status==1).groupBy(df.grade).agg(F.mean(df.fracNumPmts).alias('defTime60mean'),
																	F.stddev(df.fracNumPmts).alias('defTime60sd'))

#ER time early repayment only
ERTime36 = df.filter(df.term2==36).filter(df.ER==1).groupBy(df.grade).agg(F.mean(df.fracNumPmts).alias('ERTime36mean'),
															F.stddev(df.fracNumPmts).alias('ERTime36sd'))
ERTime60 = df.filter(df.term2==60).filter(df.ER==1).groupBy(df.grade).agg(F.mean(df.fracNumPmts).alias('ERTime60mean'),
															F.stddev(df.fracNumPmts).alias('ERTime60sd'))


#Join all of them up by grade
grade_table = def36.join(def60, def36.grade==def60.grade,'inner').drop(def36.grade)
grade_table = grade_table.join(ER36, grade_table.grade==ER36.grade,'inner').drop(ER36.grade)
grade_table = grade_table.join(ER60, grade_table.grade==ER60.grade,'inner').drop(ER60.grade)
grade_table = grade_table.join(defTime36, grade_table.grade==defTime36.grade,'inner').drop(defTime36.grade)
grade_table = grade_table.join(defTime60, grade_table.grade==defTime60.grade,'inner').drop(defTime60.grade)
grade_table = grade_table.join(ERTime36, grade_table.grade==ERTime36.grade,'inner').drop(ERTime36.grade)
grade_table = grade_table.join(ERTime60, grade_table.grade==ERTime60.grade,'inner').drop(ERTime60.grade)


#optional
grade_table = grade_table.sort(grade_table.grade)

#expensive, switching columns around, optional
# grade_table = grade_table.select('grade', 'def36mean', 'def36sd', 'def60mean', 'def60sd', 'defTime36mean', 
# 						'defTime36sd', 'defTime60mean', 'defTime60sd', 
# 						'repTime36mean', 'repTime36sd', 'repTime60mean', 'repTime60sd',
# 						'ER36mean', 'ER36sd','ER60mean','ER60sd',
# 						 'ERTime36mean', 'ERTime36sd', 'ERTime60mean', 'ERTime60sd')

#grade_table.show()

#export to csv (via pandas)
pdTable = grade_table.toPandas()
pdTable.to_csv('data/pdDefRate.csv', index=False)


spark.stop()
