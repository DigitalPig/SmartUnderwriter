#!/usr/bin/env python3

import os
import os.path
import numpy as np
import pandas as pd

start_path = os.getcwd()
source_path = os.path.join(start_path,'processed','merged')
dest_path = os.path.join(start_path, 'processed','total')

##years = list(range(2010, 2015))
years = range(2013,2015)


summaryfile_type = {'Unnamed':int,
                    'Unnamed2':int,
                    'LOAN_ID':object,
                    'ORIG_CHN':object,
                    'Seller.Name':object,
                    'ORIG_RT':float,
                    'ORIG_AMT':float,
                    'ORIG_TRM':float,
                    'ORIG_DTE':object,
                    'FRST_DTE':object,
                    'OLTV':float,
                    'OCLTV':float,
                    'NUM_BO':float,
                    'DTI':float,
                    'CSCORE_B':float,
                    'FTHB_FLG':object,
                    'PURPOSE':object,
                    'PROP_TYP':object,
                    'NUM_UNIT':int,
                    'OCC_STAT':object,
                    'STATE':object,
                    'ZIP_3':object,
                    'MI_PCT':float,
                    'Product_Type':object,
                    'CSCORE_C':float,
                    'Monthly.Rpt.Prd':object,
                    'Servicer.Name':object,
                    'LAST_RT':float,
                    'LAST_UPB':float,
                    'Loan.Age':float,
                    'Months.To.Legal.Mat':float,
                    'Adj.Month.To.Mat':float,
                    'Maturity.Date':object,
                    'MSA':int,
                    'Delq_Status':object,
                    'MOD_FLAG':object,
                    'Zero.Bal.Code':float,
                    'ZB_DTE':object,
                    'LPI_DTE':object,
                    'FCC_DTE':object,
                    'DISP_DT':object
}




 
counter = 0
for year in years:
    print('Now processing {year}'.format(year=year))
    filename = 'merged_summary_' + str(year) + '.csv'
    quar_file = os.path.join(source_path, filename)
    try:
        source = pd.read_csv(quar_file, dtype = summaryfile_type)
    except (OSError, IOError) as err:
        print(err)
        continue
    source.drop(source.columns[[0,1]],axis=1,inplace=True)
    if counter == 0:
        destfile = source.copy()
        counter = 1
    else:
        destfile = pd.concat([source,destfile])
dfilename = 'total_2014.csv'
destfile.to_csv(os.path.join(dest_path,dfilename))
print('Writing is completed!')




