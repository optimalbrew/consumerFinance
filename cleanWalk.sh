#!/bin/bash

#cross walk from ZipXX to County FIPS
#created for lendClub example. need to go from zipXX e.g. 902XX to county fips
#this mapping is not unique. So need a way for either fractional or probabilistic assignment.
#Example for Austin, Texas: there are 78 zip codes of the form 787XX 
#    6 787XX zip codes in Williamson County,TX,48491
#   68 787XX codes for Travis County,TX,48453
#    4 787XX codes for Hays County,TX,48209
# Thus, given 787XX, we want to assign to Travis Co. with probability 68/78.
#Note: Cannot use population, etc. Hays doesn't just have 4 zip codes! It has 16.

#Download initial data from kaggle api from user danofer/zipcodes-county-fips-crosswalk

#Or use the HUD (.xlsx) file from https://www.huduser.gov/portal/datasets/usps.html

#sample the file 
cat ZIP-COUNTY-FIPS_2017-06.csv | awk 'BEGIN {srand()} !/^$/ { if (rand() <= .01) print $0}' > sampledOutput.txt
cat sampledOutput.txt
rm sampledOutput.txt

#seems like all lines end up with a H or C classification  
#try to isolate classifications
sed 's/.*[0-9]\{5\},//g' ZIP-COUNTY-FIPS_2017-06.csv > test.csv
sort test.csv > test2.csv
uniq -c test2.csv
# 351 C7
# 50307 H1
# 1035 H4
#  130 H5
# 1066 H6
rm test*.csv

sed 's/\(,H[0-9]\)//g' ZIP-COUNTY-FIPS_2017-06.csv> crosswalk0a.csv
sed 's/\(,C[0-9]\)//g' crosswalk0a.csv >crosswalk0b.csv
sed 's/\([0-9]\{5\},\)/\1,\1/g' crosswalk0b.csv > crosswalk1.csv # \{n\} is n reps of pattern
sed 's/\([0-9]\{2\},,\)/XX,/g' crosswalk1.csv > crosswalk2.csv
sed 's/\(ZIP\)/\1xx,\1/g' crosswalk2.csv > crosswalk3.csv
sed 's/\(,CLASSFP\)//g' crosswalk3.csv > cross_walk.csv
#remove intermediate files
rm crosswalk*.csv

#echo 'Base cross walk version.'
#head cross_walk.csv

#for joins, a version with just two columns may be useful, as that can be reduced further by removing duplicates.
#first sort by ZIP code  (uniq only compares adjacent lines)
#sort -k2 -t, -r -o OutWalk.csv cross_walk.csv #k column, t delimiter, r reverse, o outfile

#sed 's/\(ZIPxx,ZIP,COUNTYNAME,STATE,STCOUNTYFP\)/ZIPxx, FIPS/g' cwalk.csv > cwalk0.csv
sed 's/\([[:alpha:]]\)//g' cross_walk.csv > cwalk1.csv
sed 's/\([0-9]\{5\},\)//g' cwalk1.csv > cwalk2.csv
sed 's/\([0-9]\{3\}\),/\1XX,/g' cwalk2.csv > cwalk3.csv
sed 's/-//g' cwalk3.csv > cwalk3a.csv #remove dashes
sed 's/\.//g' cwalk3a.csv > cwalk3b.csv #remove periods 
sed 's/'\''//g' cwalk3b.csv > cwalk3c.csv #remove quotes
sed 's/\(,,,,\)/ZIPxx, FIPS/g' cwalk3c.csv > cwalk4.csv
sed 's/,\( *\) ,,/,/g' cwalk4.csv > cwalk5a.csv #remove any number of spaces between , and ,,
sed 's/,,,/,/g' cwalk5a.csv > cwalk5b.csv #remove ,,,

#sort by fips
sort -k2 -t,  -o cwalk6.csv cwalk5b.csv #k column, t delimiter, r reverse, o outfile
#remove duplicates (adjacent lines)
uniq cwalk6.csv cwalk7.csv

sort -k1 -t, -r -o zipXX2ctyFips.csv cwalk7.csv #onl

echo 
echo 'Zip XX and County FIPS codes.'
echo ' '
head zipXX2ctyFips.csv
echo 'Clearly, the relationship is not one to one. So check the number of times a line is repeated'

rm cwalk*.csv

#get counts of repetitions
sed 's/[0-9]\{5\},//g' cross_walk.csv | sort -k1 -t, -r | uniq -c > cwRep.csv
sed 's/1 ZIPxx,ZIP,COUNTYNAME,STATE,STCOUNTYFP/repcount,ZIPxx,CtyNAME,State,CtyFips/g' cwRep.csv > cwRepMod.csv
sed 's/ \([0-9]\{3\}XX\)/,\1/g' cwRepMod.csv > cwRep.csv
rm cwRepMod.csv

#examples
echo
echo 'PoughKeepsie, NY'
grep 126XX cwRep.csv #Poughkeepsie, NY
#4 126XX,Dutchess County,NY,36027

echo
echo "Seattle, WA"
grep 981XX cwRep.csv #Seattle, WA
#   1 981XX,Kitsap County,WA,53035
#  55 981XX,King County,WA,53033

echo
echo 'Austin, TX'
grep 787XX cwRep.csv #Austin, TX
#    6 787XX,Williamson County,TX,48491
#   68 787XX,Travis County,TX,48453
#    4 787XX,Hays County,TX,48209

echo 'It is not always this lopsided.'

echo
echo 'Cleveland, OH'
grep 440XX cwRep.csv #Cleveland, OH
#    3 440XX,Trumbull County,OH,39155
#    3 440XX,Summit County,OH,39153
#    2 440XX,Medina County,OH,39103 
#   17 440XX,Lorain County,OH,39093
#   13 440XX,Lake County,OH,39085
#   20 440XX,Geauga County,OH,39055
#    1 440XX,Erie County,OH,39043
#    7 440XX,Cuyahoga County,OH,39035
#   21 440XX,Ashtabula County,OH,39007

#the number of zip codes for that zipXX
grep 440XX cross_walk.csv|wc -l
#      87

#so we ought to divide by this as the assignment to any county. 
#Do not use county population! County may have various zip codes not all with the same ZipXX!
#Ex: Lorain County, OH Pop: 300K has 25 zip codes (only 17 of them are 440XX).
# Medina County, OH pop: 178k also has 25 zip codes (only 2 of them are 440XX). 

#list of unique zipXX's
sed 's/.*\([0-9]\{3\}XX\).*/\1/g' cwRep.csv > zipXXlist0.csv
uniq zipXXlist0.csv zipXXuniq.txt
wc -l zipXXuniq.txt
#     911 zipXXuniq.txt
rm zipXXlist0.csv


#add the total to the zipXX file so we can calculate the probability of assigning a 
#unique zipXX to several possible county FIPS

#read in zipXX from list of unique zipXX
while read -r line0; do
	a=`grep $line0 cross_walk.csv|wc -l` #total zip codes with that ZipXX
	grep $line0 cwRep.csv > b.txt #write to file (overwrite, do not append)	
	while read -r line1;do
		echo $a,$line1 >> outZipXX.csv #append to this file
	done < b.txt
echo 'done with' $line0 
done < zipXXuniq.txt

rm b

echo 'Do not forget to change the header! else we have 0,repcount,ZIPxx,CtyNAME,State,CtyFips'

#now we can get the probabilities via the (inverse) ratio of 1st and 2nd values.
grep 440XX outZipXX.csv 
# 87,3,440XX,Trumbull County,OH,39155 #assign 440XX to FIPS 39155 with prob 3/87
# 87,3,440XX,Summit County,OH,39153
# 87,2,440XX,Medina County,OH,39103
# 87,17,440XX,Lorain County,OH,39093
# 87,13,440XX,Lake County,OH,39085
# 87,20,440XX,Geauga County,OH,39055
# 87,1,440XX,Erie County,OH,39043
# 87,7,440XX,Cuyahoga County,OH,39035
# 87,21,440XX,Ashtabula County,OH,39007

