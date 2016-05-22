# coding: utf-8

#!/usr/bin/env python3

import pandas as pd
import os
import os.path
import numpy as np


#%% This is for Spyder IDE

## Define global label

Acquisition_label = {"LOAN_ID": np.dtype(str),
                     "ORIG_CHN": object,
                     "Seller.Name": object,
                     "ORIG_RT": np.dtype(float),
                     "ORIG_AMT": np.dtype(np.uint32),
                     "ORIG_TRM": np.dtype(np.uint32),
                     "ORIG_DTE": object,
                     "FRST_DTE": object,
                     "OLTV": np.dtype(float),
                     "OCLTV": np.dtype(float),
                     "NUM_BO": np.dtype(float),
                     "DTI":np.dtype(float),
                     "CSCORE_B": np.dtype(float),
                     "FTHB_FLG": object,
                     "PURPOSE": np.dtype(object),
                     "PROP_TYP":np.dtype(object),
                     "NUM_UNIT": np.dtype(int),
                     "OCC_STAT":np.dtype(object),
                     "STATE": np.dtype(object),
                     "ZIP_3": object,
                     "MI_PCT": np.dtype(float),
                     "Product.Type": np.dtype(object),
                     "CSCORE_C": np.dtype(float)
}

Acquisition_names = ["LOAN_ID",
                     "ORIG_CHN",
                     "Seller.Name",
                     "ORIG_RT",
                     "ORIG_AMT",
                     "ORIG_TRM",
                     "ORIG_DTE",
                     "FRST_DTE",
                     "OLTV",
                     "OCLTV",
                     "NUM_BO",
                     "DTI",
                     "CSCORE_B",
                     "FTHB_FLG",
                     "PURPOSE",
                     "PROP_TYP",
                     "NUM_UNIT",
                     "OCC_STAT",
                     "STATE",
                     "ZIP_3",
                     "MI_PCT",
                     "Product.Type",
                     "CSCORE_C"]


Performance_label ={"LOAN_ID": object,
                    "Monthly.Rpt.Prd": str,
                    "Servicer.Name": str,
                    "LAST_RT": float,
                    "LAST_UPB": float,
                    "Loan.Age": str, ##change
                    "Months.To.Legal.Mat": float,
                    "Adj.Month.To.Mat": float,
                    "Maturity.Date": str,
                    "MSA":str,
                    "Delq.Status": str,
                    "MOD_FLAG": str,
                    "Zero.Bal.Code": np.dtype(float),
                    "ZB_DTE":np.dtype(str),
                    "LPI_DTE":np.dtype(str),
                    "FCC_DTE":np.dtype(str),
                    "DISP_DT":np.dtype(str),
#                    "FCC_COST":np.dtype(str),
#                    "PP_COST":np.dtype(str),
#                    "AR_COST":np.dtype(str),
#                    "IE_COST":np.dtype(str),
#                    "TAX_COST":np.dtype(str),
#                    "NS_PROCS":np.dtype(str),
#                    "CE_PROCS":np.dtype(str),
#                    "RMW_PROCS":np.dtype(str),
#                    "O_PROCS":np.dtype(object),
#                    "NON_INT_UPB":np.dtype(str),
#                    "PRIN_FORG_UPB":np.dtype(str)
}

Performance_names =["LOAN_ID",
                    "Monthly.Rpt.Prd",
                    "Servicer.Name",
                    "LAST_RT",
                    "LAST_UPB",
                    "Loan.Age",
                    "Months.To.Legal.Mat",
                    "Adj.Month.To.Mat",
                    "Maturity.Date",
                    "MSA",
                    "Delq.Status",
                    "MOD_FLAG",
                    "Zero.Bal.Code",
                    "ZB_DTE",
                    "LPI_DTE",
                    "FCC_DTE",
                    "DISP_DT",
#                    "FCC_COST",
#                    "PP_COST",
#                    "AR_COST",
                    # "IE_COST",
                    # "TAX_COST",
                    # "NS_PROCS",
                    # "CE_PROCS",
                    # "RMW_PROCS",
                    # "O_PROCS",
                    # "NON_INT_UPB",
                    # "PRIN_FORG_UPB"
]



def processData(Pfile, Afile, year, quarter, folder = 'processed'):
    '''
    This function is to process the Fannie raw data and condense it to the
    summary dataset.
    Usage:
    processData("Performance_2014Q2.txt", "Acquisition_2014Q2.txt", '2014', '2')
    return True if everything is successful.
    Otherwise return False, possible with any exceptions.
    
    '''
    global Performance_label, Acquisition_label, Performance_names, Acquisition_names
    print('Start to process data from {year} {quarter}...\n'.format(year=year,
                                                               quarter = quarter))

    perform = pd.read_csv(Pfile, header = None, sep = '|', names =
                          Performance_names, na_values = "NaN",
                          index_col = False, dtype = Performance_label,
                          usecols=range(17))
    print('Performance Reading Finished!')
    acquisition = pd.read_csv(Afile, header = None, sep='|', names =
                              Acquisition_names,na_values = "NaN",
                              index_col = False, dtype=Acquisition_label,
                              error_bad_lines = False)
    print('All Reading Finished!')
    #%% In[21]:
    ## Massage the data
    ## 1) Convert LOAN_ID field into characater field.
    perform['LOAN_ID'] = perform['LOAN_ID'].astype('<U20')
    acquisition['LOAN_ID'] = acquisition['LOAN_ID'].astype('<U20')
    print("LOAN_ID conversion complete!")
    ## Change the NaN Zero Balance Code into 0 for the convenience of later processing.
    #perform['Monthly.Rpt.Prd'] = pd.to_datetime(perform['Monthly.Rpt.Prd'],
    #                                            format ='%m/%d/%Y')
    #acquisition['ORIG_DTE'] = pd.to_datetime(acquisition['ORIG_DTE'], format='%m/%Y')
    #print("Montly Report Date conversion complete!")
    perform.loc[perform['Zero.Bal.Code'].isnull(), 'Zero.Bal.Code'] = 0
    print('Processing the performance data...\n')
    perform_byid = perform.groupby(["LOAN_ID"], sort = False).last()
    print('Groupby Performance data is done')
    
    ## Merge the processed data with acquisition data together.
    print('Merging performance and acquisition data...\n')
    res = acquisition.merge(perform_byid.reset_index(), on = 'LOAN_ID', how = 'outer')
    print('Writing summary file summary_{0}Q{1}.csv...\n'.format(year,quarter))
    SFile_prefix = 'summary_'
    filename = SFile_prefix + str(year) + 'Q' + str(quarter) + '.csv'
    cwd = os.getcwd()
    fullpath = os.path.join(cwd, folder, filename)
    res.to_csv(fullpath)
    return True


if (__name__ == "__main__"):
    years = range(2003,2004)
    quarters = range(2,5)
    cwd = os.getcwd()
    dest = os.path.join(cwd,'raw')
    Afile_prefix = 'Acquisition_'
    Pfile_prefix = 'Performance_'
    

    for year in years:
        for quarter in quarters:
            Afile = os.path.join(dest,
                                 Afile_prefix+str(year)+'Q'+str(quarter)+'.txt')
            Pfile = os.path.join(dest,
                                 Pfile_prefix+str(year)+'Q'+str(quarter)+'.txt')
            processData(Pfile, Afile, year, quarter)

    print("All finished! Enjoy!")
