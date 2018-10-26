""" 
Early Repayment regressions

"""
from __future__ import print_function

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import Word2Vec
from pyspark.ml.clustering import KMeans	

from pyspark.ml.regression import LinearRegression
#from pyspark.ml.classification import LogisticRegression
#from pyspark.ml.regression import RandomForestRegressor
#from pyspark.ml.regression import GBTRegressor
#from pyspark.sql import Row
from pyspark.sql.types import *
#from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import RFormula
import pyspark.sql.functions as F
#from pyspark.ml.evaluation import RegressionEvaluator

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

#initial cleanup

##get rid of "does not meet ..." just  keep paid or charged off using Java regex
## use 1 for default and 0 for repaid
df = df.withColumn('loan_status', F.regexp_replace('loan_status', '.*Paid$' , '0').alias('loan_status'))
df = df.withColumn('loan_status', F.regexp_replace('loan_status', '.*Off$' , '1').alias('loan_status'))

#string clean up, remove %, strip months away from term
df = df.withColumn('int_rate', F.regexp_replace('int_rate', '%' , '').alias('int_rate'))
df = df.withColumn('revol_util', F.regexp_replace('revol_util', '%' , '').alias('revol_util'))
df = df.withColumn('term', F.regexp_replace('term', '( months)|( )' , '').alias('term'))	


#reduce employment length categories
df = df.withColumn('emp_length', F.regexp_replace('emp_length', '(n/a)|(< 1 year)' , '< 1 yr').alias('emp_length'))
df = df.na.fill({'emp_length': '< 1 yr'})
df = df.withColumn('emp_length', F.regexp_replace('emp_length', '(1 year)|(2 years)|(3 years)|(4 years)' , '1-4 years').alias('emp_length'))	
df = df.withColumn('emp_length', F.regexp_replace('emp_length', '(5)|(6)|(7)|(8)|(9)' , '5-9').alias('emp_length'))

# df.groupBy('emp_length').count().show()
# +----------+-----+                                                              
# |emp_length|count|
# +----------+-----+
# |    < 1 yr|21113|
# | 5-9 years|53477|
# | 1-4 years|52394|
# | 10+ years|61199|
# +----------+-----+

#simplify home ownership categories	
df = df.withColumn('home_ownership', F.regexp_replace('home_ownership', '(OWN)|(NONE)' , 'OWN').alias('home_ownership'))
df = df.na.fill({'home_ownership': 'OWN' }) 

df.groupBy(df.home_ownership).count().sort('count').show()
# +--------------+-----+                                                          
# |home_ownership|count|
# +--------------+-----+
# |           OWN|15079|
# |          RENT|74060|
# |      MORTGAGE|93900|
# +--------------+-----+

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
df = df.withColumn('last_pymnt_d', (F.to_date('last_pymnt_d', 'MM-dd-yyyy')).alias('last_pymnt_d'))
df = df.withColumn('mnth_start2last', F.months_between(df.last_pymnt_d, df.issue_d).alias('mnth_start2last'))
df = df.withColumn('fracNumPmts', F.col('mnth_start2last')/F.col('term2'))
		
#for tokenizer change null to none
df = df.na.fill({'desc': 'none' }) #perhaps doesn't matter?


df2 = df #keep a backup
#then drop rows with leftover na's
df = df.na.drop(how='any')

if df.count()>0:
	del(df2) #if df is okay (did not lose all data!)

#A summary of approximate stats. Lower relativeError for more precision, 0 is exact
#df.filter(df.term2==60).filter(df.loan_status==0).approxQuantile(col=['mnth_start2last'], probabilities=[.25,.5,.75], relativeError=.15)
#will not work wit groupBy + agg(). If required for groups, need to loop over, and filter by group

df.groupBy(df.term).agg(F.avg(df.fracNumPmts)).show()
# +----+------------------+                                                       
# |term|  avg(fracNumPmts)|
# +----+------------------+
# |  60|0.5334839555374444| #repaid very early
# |  36|0.7324283072750936|
# +----+------------------+


#start with text processing (most likely it has no significant impact)
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

df.select('pred_KM','desc').show(20,truncate=False)

df.groupBy(df.pred_KM).count().sort('count').show()
# +-------+------+                                                                
# |pred_KM| count|
# +-------+------+
# |      0| 22030|brief, generic credit card
# |      2| 56377|Longer desc, more details (intermediate freq)
# |      1|104632|No description
# +-------+------+

#For regression, need to treat the predicted class as categorical variable, not integer
df = df.withColumn('pred_KM', df['pred_KM'].cast(StringType()).alias('pred_KM'))
#now the subs
df = df.withColumn('pred_KM', F.regexp_replace('pred_KM', '0' , 'Generic desc.').alias('pred_KM'))
df = df.withColumn('pred_KM', F.regexp_replace('pred_KM', '1' , 'No desc.').alias('pred_KM'))
df = df.withColumn('pred_KM', F.regexp_replace('pred_KM', '2' , 'Detailed desc.').alias('pred_KM'))


df.groupBy(df.grade).count().sort('count').show()
# +-----+-----+                                                                   
# |grade|count|
# +-----+-----+
# |    G|  578|
# |    A|  614|
# |    F| 2814|
# |    B| 4079|
# |    E| 4928|
# |    D| 5129|
# |    C|10085|
# +-----+-----+


### Break-even period

#Number of installments that must be paid for principal to be recovered.
payback60 = df.filter(df.term2==60).filter(df.loan_status==0).groupBy(df.grade).agg(F.mean(df.loan_amnt)/F.mean(df.installment)).sort(df.grade)
payback60.show()
# +-----+-----------------------------------+                                     
# |grade|(avg(loan_amnt) / avg(installment))|
# +-----+-----------------------------------+
# |    A|                  48.32954603903535|
# |    B|                  44.91147577806454|
# |    C|                  41.28973856725123|
# |    D|                  38.74003796920464|
# |    E|                 36.555859756320366|
# |    F|                  35.04084785958727|
# |    G|                  33.93630007382986|
# +-----+-----------------------------------+


payback36 = df.filter(df.term2==36).filter(df.loan_status==0).groupBy(df.grade).agg(F.mean(df.loan_amnt)/F.mean(df.installment)).sort(df.grade)
payback36.show()
# +-----+-----------------------------------+                                     
# |grade|(avg(loan_amnt) / avg(installment))|
# +-----+-----------------------------------+
# |    A|                  32.05468295348546|
# |    B|                  30.16429696488312|
# |    C|                 28.722511349435106|
# |    D|                 27.459438285764392|
# |    E|                 26.444498266501213|
# |    F|                 25.591998840251794|
# |    G|                 25.139767478701053|
# +-----+-----------------------------------+



#all 60 months loans that were repaid (not just early, but all)
df = df.filter(df.loan_status==0).filter(df.term2==60)

## regression formula
#predict the number of installments that will be paid (0-1) with anything less than 1
#implying early repayment of loan

#which cols?
#cols:['loan_amnt', 'int_rate', 'installment', 'grade', 'emp_length', 'home_ownership', 'annual_inc', 'issue_d', 'dti',
# 'revol_util', 'total_pymnt', 'last_pymnt_d', 'last_pymnt_amnt', 'mnth_start2last', 
#'fracNumPmts', 'pred_KM']


formula = RFormula(
	formula = "fracNumPmts ~ installment + annual_inc + dti + int_rate + revol_util  + home_ownership + grade + emp_length + pred_KM",
	featuresCol="features",
	labelCol="label")

#transformed data frame with vectors assembled
regFormulaFit = formula.fit(df).transform(df)

#training data frame
training = regFormulaFit.select(["label","features"])
lr = LinearRegression(labelCol = "label", featuresCol= "features", maxIter=10)#, regParam=0.3)
lrModel = lr.fit(training)
trainingSummary = lrModel.summary


df.select('fracNumPmts').describe().show()
# +-------+------------------+                                                    
# |summary|       fracNumPmts|
# +-------+------------------+
# |  count|             28227|
# |   mean|0.5334839555374444|
# | stddev|0.2962701727734131|
# |    min|               0.0|
# |    max|1.1075268816666666|
# +-------+------------------+

print("\nThe typical 60 month loan is repaid in 32 months\n") 

loanterm = 60

df.select(loanterm*F.mean(df.fracNumPmts)).show()
# +-----------------------+                                                       
# |(avg(fracNumPmts) * 60)|	
# +-----------------------+
# |      32.00903733224666|
# +-----------------------+


#Note: DTI and revol util do not behave in same direction.



print("\n\n\nRegression results\n")
print("\n$50 inc. in installment:  %s months change(t-stat) (%s) \n"\
 %(str(50*loanterm*lrModel.coefficients[0]), str(trainingSummary.tValues[0]))) 

print("\n$10000 inc. in annual income:%s months change (t-stat) (%s) \n"\
 %(str(10000*loanterm*lrModel.coefficients[1]), str(trainingSummary.tValues[1]))) 

print("\n10 percentage point inc, in dti: %s months change  (t-stat) (%s) \n"\
 %(str(10*loanterm*lrModel.coefficients[2]), str(trainingSummary.tValues[2]))) 


print("\n1 percentage-point inc. in interest rate: %s months change (t-stat) (%s) \n"\
 %(str(loanterm*lrModel.coefficients[3]), str(trainingSummary.tValues[3]))) 

print("\n10 percentage-point inc. in revol. util: %s months change (t-stat) (%s) \n"\
 %(str(10*loanterm*lrModel.coefficients[4]), str(trainingSummary.tValues[4]))) 


#display order for home ownership
df.groupBy(df.home_ownership).count().sort('count',ascending=False).show()
print("\n Home ownership: %s months change  (t-stat) (%s) \n"\
 %(str(loanterm*lrModel.coefficients[5:7]), str(trainingSummary.tValues[5:7])))

#display order for grade
df.groupBy(df.grade).count().sort('count',ascending=False).show()
print("\nGrade: %s months change from \nwith (t-stat) (%s) \n"\
 %(str(loanterm*lrModel.coefficients[7:13]), str(trainingSummary.tValues[7:13])))

#display order
df.groupBy(df.emp_length).count().sort('count',ascending=False).show()
print("\nEmployment length: %s months change \nwith (t-stat) (%s) \n"\
 %(str(loanterm*lrModel.coefficients[13:16]), str(trainingSummary.tValues[13:16])))

#display order for description
df.groupBy(df.pred_KM).count().sort('count',ascending=False).show()
print("\nDescription: %s months change (t-stat) (%s) \n"\
 %(str(loanterm*lrModel.coefficients[16:18]), str(trainingSummary.tValues[16:18])))


print("t-stats: %s" % str(trainingSummary.tValues))



spark.stop()
