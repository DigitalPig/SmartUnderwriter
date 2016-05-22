#!/usr/bin/env python3

import os
import os.path
import numpy as np
import pandas as pd

start_path = os.getcwd()
source_path = os.path.join(start_path,'processed')
dest_path = os.path.join(start_path, 'processed','merged')

##years = list(range(2010, 2015))
years = range(2000,2015)
quarters = range(1,5)

summaryfile_type = {'Unnamed':int,
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

for year in years:
    is_error = False
    for quarter in quarters:
        print('Now processing {year}-Q{quarter}'.format(year=year,quarter=quarter))
        filename = 'summary_' + str(year) + 'Q' + str(quarter) + '.csv'
        quar_file = os.path.join(source_path, filename)
        try:
            source = pd.read_csv(quar_file, dtype = summaryfile_type)
        except (OSError, IOError) as err:
            print(err)
            continue
        if quarter == 1:
            destfile = source.copy()
        else:
            destfile = pd.concat([source,destfile])
    dfilename = 'merged_summary_'+str(year)+'.csv'
    destfile.to_csv(os.path.join(dest_path,dfilename))
    print('Writing is completed!')




