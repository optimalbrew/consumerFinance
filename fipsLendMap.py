"""
   	  ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 2.3.1
      /_/
Join lendclub loan data with county fips info using ZIPxx as key.
3 digit ZIP+XX is mapped to multiple counties based on relative odds of the number of zipXX

#Example for Austin, Texas: there are 78 zip codes of form 787XX 
#    6 787XX zip codes in Williamson County,TX,48491
#   68 787XX codes for Travis County,TX,48453
#    4 787XX codes for Hays County,TX,48209
# Thus, given 787XX, we want to assign to Travis Co. with probability 68/78.
#Note: Cannot use population, etc. Hays doesn't just have 4 zip codes! It has 16.

"""
from __future__ import print_function

from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
import pandas as pd

if __name__ == "__main__":
	spark = SparkSession \
		.builder \
		.appName("Python Spark SQL basic example") \
		.config("spark.some.config.option", "some-value") \
		.getOrCreate()
	
	df_data = spark.read.load("./data/LS2018Q2.csv",format="csv", sep=",", inferSchema="true", header="true")
	#drop columns we do not need

	df_data = df_data.drop(\
		"id","member_id","funded_amnt","funded_amnt_inv", "term",\
		"installment","grade","sub_grade","emp_title","emp_length","home_ownership",\
		"verification_status","issue_d","loan_status","pymnt_plan","url","desc","purpose","title",\
		"delinq_2yrs","earliest_cr_line","inq_last_6mths","mths_since_last_delinq",\
		"mths_since_last_record","open_acc","pub_rec","revol_bal","revol_util","total_acc",\
		"initial_list_status","out_prncp","out_prncp_inv","total_pymnt","total_pymnt_inv",\
		"total_rec_prncp","total_rec_int","total_rec_late_fee","recoveries",\
		"collection_recovery_fee","last_pymnt_d","last_pymnt_amnt","next_pymnt_d",\
		"last_credit_pull_d","collections_12_mths_ex_med","mths_since_last_major_derog",\
		"policy_code","application_type","annual_inc_joint","dti_joint",\
		"verification_status_joint","acc_now_delinq","tot_coll_amt","tot_cur_bal",\
		"open_acc_6m","open_act_il","open_il_12m","open_il_24m","mths_since_rcnt_il",\
		"total_bal_il","il_util","open_rv_12m","open_rv_24m","max_bal_bc","all_util",\
		"total_rev_hi_lim","inq_fi","total_cu_tl","inq_last_12m","acc_open_past_24mths",\
		"avg_cur_bal","bc_open_to_buy","bc_util","chargeoff_within_12_mths","delinq_amnt",\
		"mo_sin_old_il_acct","mo_sin_old_rev_tl_op","mo_sin_rcnt_rev_tl_op",\
		"mo_sin_rcnt_tl","mort_acc","mths_since_recent_bc","mths_since_recent_bc_dlq",\
		"mths_since_recent_inq","mths_since_recent_revol_delinq","num_accts_ever_120_pd",\
		"num_actv_bc_tl","num_actv_rev_tl","num_bc_sats","num_bc_tl","num_il_tl",\
		"num_op_rev_tl","num_rev_accts","num_rev_tl_bal_gt_0","num_sats",\
		"num_tl_120dpd_2m","num_tl_30dpd","num_tl_90g_dpd_24m","num_tl_op_past_12m",\
		"pct_tl_nvr_dlq","percent_bc_gt_75","pub_rec_bankruptcies","tax_liens",\
		"tot_hi_cred_lim","total_bal_ex_mort","total_bc_limit",\
		"total_il_high_credit_limit","revol_bal_joint","sec_app_earliest_cr_line","sec_app_inq_last_6mths",\
		"sec_app_mort_acc","sec_app_open_acc","sec_app_revol_util","sec_app_open_act_il",\
		"sec_app_num_rev_accts","sec_app_chargeoff_within_12_mths","sec_app_collections_12_mths_ex_med",\
		"sec_app_mths_since_last_major_derog","hardship_flag","hardship_type","hardship_reason",\
		"hardship_status","deferral_term","hardship_amount","hardship_start_date","hardship_end_date",\
		"payment_plan_start_date","hardship_length","hardship_dpd","hardship_loan_status",\
		"orig_projected_additional_accrued_interest","hardship_payoff_balance_amount","hardship_last_payment_amount",\
		"disbursement_method","debt_settlement_flag","debt_settlement_flag_date","settlement_status","settlement_date",\
		"settlement_amount","settlement_percentage","settlement_term"\
		)
	
	print('done reading')
	spark.stop()
	quit()
	
	#keeping: "loan_amnt", #"int_rate",\ #"annual_inc",#"zip_code","addr_state","dti",\
	
	df_fips = spark.read.load("./data/crosswalk/outZipLowerXX.csv",format="csv", sep=",", inferSchema="true", header="true")
	#totalXX,fipCount,ZIPxx,CtyNAME,State,CtyFips
	print('Dataframes ready')
	
	print('Data File..')
	df_data.show(3)
	print('Fips File..')
	df_fips.show(3)
	
	#df_data.columns
	
	#df_fips.columns
		
	#insert a blank column for values to be mapped
	df_fips = df_fips.withColumn('Val2map', F.lit(0))
		
	#aggregate the sum and store in new dataframe
	df_sum_zipxx = df_data.groupBy(df_data.zip_code).sum('loan_amnt')
	
	print('Data summed and grouped..')
	df_sum_zipxx.show()
	
	
	#do a inner join
	merged = df_sum_zipxx.join(df_fips, df_sum_zipxx.zip_code == df_fips.ZIPxx, 'inner')
	
	merged.columns
	#['zip_code', 'sum(loan_amnt)', 'totalXX', 'fipCount', 'ZIPxx', 'CtyNAME', 'State', 'CtyFips', 'Val2map']
	
	merged.count() #returns number of rows
	#6389
	
	#not yet.
	merged2 = merged.select(merged.zip_code, merged.CtyFips, merged.State, ((F.col('fipCount')/F.col('totalXX'))*F.col('sum(loan_amnt)')).alias('Val2map'))
	
	print('Data merged with Fips file')
	merged.show(10)
	# +--------+--------------+-------+--------+-----+----------------+-----+-------+-------+
	# |zip_code|sum(loan_amnt)|totalXX|fipCount|ZIPxx|         CtyNAME|State|CtyFips|Val2map|
	# +--------+--------------+-------+--------+-----+----------------+-----+-------+-------+
	# |   471xx|       1433400|     87|      19|471xx|    Clark County|   IN|  18019|      0|
	# |   471xx|       1433400|     87|       8|471xx| Crawford County|   IN|  18025|      0|
	# |   471xx|       1433400|     87|      12|471xx|    Floyd County|   IN|  18043|      0|
	# |   471xx|       1433400|     87|      18|471xx| Harrison County|   IN|  18061|      0|
	# |   471xx|       1433400|     87|       1|471xx|  Jackson County|   IN|  18071|      0|
	# |   471xx|       1433400|     87|       3|471xx|Jefferson County|   IN|  18077|      0|
	# |   471xx|       1433400|     87|       1|471xx| Lawrence County|   IN|  18093|      0|
	# |   471xx|       1433400|     87|       5|471xx|   Orange County|   IN|  18117|      0|
	# |   471xx|       1433400|     87|       2|471xx|    Perry County|   IN|  18123|      0|
	# |   471xx|       1433400|     87|       7|471xx|    Scott County|   IN|  18143|      0|
	# +--------+--------------+-------+--------+-----+----------------+-----+-------+-------+	
	
	print('Values summed and allocated to FIPS')
	merged2.show(10)
	# +--------+-------+------------------+                                           
	# |zip_code|CtyFips|           Val2map|
	# +--------+-------+------------------+
	# |   471xx|  18019|313041.37931034487|
	# |   471xx|  18025|131806.89655172414|
	# |   471xx|  18043| 197710.3448275862|
	# |   471xx|  18061| 296565.5172413793|
	# |   471xx|  18071|16475.862068965518|
	# |   471xx|  18077| 49427.58620689655|
	# |   471xx|  18093|16475.862068965518|
	# |   471xx|  18117| 82379.31034482758|
	# |   471xx|  18123|32951.724137931036|
	# |   471xx|  18143|115331.03448275862|
	# +--------+-------+------------------+
		
	print('Final allocations')
	sumFips = merged2.groupBy(merged2.CtyFips).sum('Val2map')	
	sumFips.show(10)
	# +-------+------------------+                                                    
	# |CtyFips|      sum(Val2map)|
	# +-------+------------------+
	# |  37111|236543.15476190476|
	# |  22097|413952.14587100665|
	# |  26087| 697642.5229800469|
	# |  13285| 398064.1304347826|
	# |  31035|48105.846914008325|
	# |  18051|236454.48679471787|
	# |  23015| 243588.5649054142|
	# |  40011| 134258.6678832117|
	# |  40107| 103987.2785377191|
	# |  13289|121994.42829457365|
	# +-------+------------------+		
	
	##Now repeat for means 
	df_avg_zipxx = df_data.groupBy(df_data.zip_code).avg('dti')
	print('Data averaged and grouped..')
	df_avg_zipxx.show()
	
	print('Data merged with Fips file')
	merged = df_avg_zipxx.join(df_fips, df_avg_zipxx.zip_code == df_fips.ZIPxx, 'inner')
	merged2 = merged.select(merged.zip_code, merged.CtyFips, merged.State, ((F.col('fipCount')/F.col('totalXX'))*F.col('avg(dti)')).alias('Val2map'))
	
	avgFips = merged2.groupBy(merged2.CtyFips).avg('Val2map')
	merged.show(10)
	print('Values averaged and allocated to FIPS')
	merged2.show(10)
	print('Final allocations')
	avgFips.show(10)
	
	sumPD = sumFips.toPandas()
	avgPD = avgFips.toPandas()
	
	sumPD.to_csv("mapping/sumPD.csv", index=False)
	avgPD.to_csv("mapping/avgPD.csv", index=False)
	
	
	
	spark.stop()
