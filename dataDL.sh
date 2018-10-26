#!/bin/bash

#lendclub "Invitation to analyze" at https://www.lendingclub.com/info/download-data.action
#download data and push to S3 bucket

#download multiple files: verbose version using Bash arrays
#curl can do this directly in a single connection/handshake


#base URl for files
base="https://resources.lendingclub.com/"

#declare array of file names 
declare -a fnameArr=("LoanStats3a.csv.zip" "LoanStats3b.csv.zip" "LoanStats3c.csv.zip" 
					"LoanStats3d.csv.zip" "LoanStats_2016Q1.csv.zip" 
					"LoanStats_2016Q2.csv.zip" "LoanStats_2016Q3.csv.zip" 
					"LoanStats_2016Q4.csv.zip" "LoanStats_2017Q1.csv.zip" 
					"LoanStats_2017Q2.csv.zip" "LoanStats_2017Q3.csv.zip"
					"LoanStats_2017Q4.csv.zip" "LoanStats_2018Q1.csv.zip")
#file sizes
#9,384kb|35,725kb|37,978kb|67,112kb|23,493kb|17,079kb|17,283kb|18,097kb|16,833kb|18,430kb|21,437kb|20,616kb|18,626kb|22,335kb|

#data on rejected applications
declare -a rejname("RejectStatsA.csv.zip" "RejectStatsB.csv.zip" "RejectStatsD.csv.zip"
					"RejectStats_2016Q1.csv.zip" "RejectStats_2016Q2.csv.zip"
					"RejectStats_2016Q3.csv.zip" "RejectStats_2016Q4.csv.zip" 
					"RejectStats_2017Q1.csv.zip" "RejectStats_2017Q2.csv.zip"
					"RejectStats_2017Q3.csv.zip" "RejectStats_2017Q4.csv.zip" 
					"RejectStats_2018Q1.csv.zip" "RejectStats_2018Q2.csv.zip")
#file sizes
#9,742kb|30,190kb|27,674kb|10,992kb|9,662kb|12,432kb|14,818kb|14,590kb|17,684kb|21,467kb|21,930kb|17,526kb|22,070kb|

#there is also the Excel dictionary
curl -O "https://resources.lendingclub.com/LCDataDictionary.xlsx" 

#download the data files sequentially (multiple connections, handshakes, yikes..).
for i in "${fnameArr[@]}"
do
	#in Bash safer to use quotes "$i" when var may have spaces or shell expandable chars"		
	echo "downloading file " "$i"
	curl -O "$base""$i" #download
done

echo "Downloaded all approved loans data, moving to rejected applications data.."

#download the rejected applications data
for i in "${rejname[@]}"
do		
	echo "downloading file " "$i"
	curl -O "$base""$i" #download
done


#move files to S3
echo "Moving files to S3"

for f in "${fnameArr[@]}"
do
	aws s3 cp  ~/data/"$f" s3://lendclub/data/"$f"
done

