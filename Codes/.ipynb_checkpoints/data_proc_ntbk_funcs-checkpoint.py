# system imports 
import os 
import gc
import glob
from functools import reduce
from tqdm.notebook import tqdm

# data processing
import numpy as np
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# data viz
import matplotlib.pyplot as plt
from sklearn import tree


## FUNCS FOR PROCSSING INTER DATA
#--------------------------------------------------------
# def func to clean cache
def garbc():
    gc.collect();
    gc.collect();

#function to convert scientific notation to simplified notation with one decimal place
def fmt(x):
# Large number format
    for i in ['', 'K', 'M', 'B', 'T']:
        if abs(x) < 1000 : return str(abs(round(x, 1))) + i
        x /= 1000

# def func to get file size in mb
def get_mb(path, fname):
    return round(os.path.getsize(os.path.join(path, fname)) / 1e6, 2)


# def func to get max available file
def get_latest(f_path, f_frmt):
    return glob.glob(os.path.join(f_path, f_frmt))[-1]


# func to get the required files for processing 
def get_lat_files(scope, source_path, file_frmt, end_mth, ret_dt=False):
    '''
    scope = int (len in months)
    source_path = source path
    file_ftm = datetime format of filename convention
    end_mth = year and month of run date in datetime 
    ret_dt (bool) = set true to get array of datetime values of output files as a second function return output

    Returns: list of files, list of mths of files
    '''

    # subtract one from scope to include edge 
    scope -= 1

    # loop through the last 24 months of files available
    for mth in reversed(pd.date_range(end_mth-relativedelta(months=24), end_mth, freq='MS')):
        # check if filename exists 
        if os.path.exists(os.path.join(source_path, mth.strftime(file_frmt))):
            upper_lim = mth
            break
            
    # get the correct 12 month range of available files 
    mths = pd.date_range(upper_lim-relativedelta(months=scope), upper_lim, freq='MS')
    files =  [mth.strftime(file_frmt) for mth in mths]

    # return 
    if ret_dt:
        return files, mths
    else:
        return files

# def func to get range of files given len of months required (takes latest available data as max)
def get_files_max(mths_len, f_path, f_frmt):
    # get latest maximum date available
    #max_dt = datetime.strptime(max(os.listdir(f_path)), f_frmt)
    clean_frmt = f_frmt.replace('%Y','*').replace('%y','*').replace('%M','*').replace('%m','*').replace('%d','*')
    max_dt = datetime.strptime(max(glob.glob(os.path.join(f_path, clean_frmt))).split('\\')[-1], f_frmt)
    # get range of months and get the files corresponding to those months
    mth_range = pd.date_range(max_dt-relativedelta(months=mths_len), max_dt, freq='MS')

    files = [mth.strftime(f_frmt) for mth in mth_range]

    # print the fsize in mb of each file 
    cache=[{'Filename':fname, 'Size (MB):':f"{get_mb(f_path, fname)}mb"} for fname in files]
    display(pd.DataFrame(cache))

    # return files
    return [os.path.join(f_path, file) for file in files]

# def func to get range of files given len of months required (requires max date to be set)
def get_files(max_dt, mths_len, f_path, f_frmt):
    '''
    max_dt = datetime for upper edge year and month
    '''
    # get range of months and get the files corresponding to those months
    mth_range = pd.date_range(max_dt-relativedelta(months=mths_len), max_dt, freq='MS')
    print(mth_range)
    files = [mth.strftime(f_frmt) for mth in mth_range if os.path.exists(os.path.join(f_path, mth.strftime(f_frmt)))]

    # print the fsize in mb of each file 
    cache=[{'Filename':fname, 'Size (MB):':f"{get_mb(f_path, fname)}mb"} for fname in files]
    display(pd.DataFrame(cache))

    # return files
    return [os.path.join(f_path, file) for file in files]


## Data Preproc
#------------------------------------------------------------------------------------------------------------------------------------------
# def func to concat and process ng fin data
def proc_ng(sc_path, files, tran_types, verbose=True, proxy='Sales'):
    ids = set(pd.read_parquet(max(glob.glob(os.path.join(sc_path, '*scorecard*_b.parquet'))), columns=['CST_ID','scorecard_id','cl_id_grp'])\
                .query("(~scorecard_id.str.contains('ATTRITED') & cl_id_grp.isin(['b.Retail']))", engine='python').CST_ID)

    # declare cols
    if proxy=='Invoice':
        cols = ['SRC_CUST_ID', 'BENE_ARID', 'BENE_CUST_ID', 'TRAN_TYPE', 'DATE_TIME']
    else:
        cols = ['SRC_CUST_ID', 'SRC_ARID', 'BENE_CUST_ID', 'TRAN_TYPE', 'DATE_TIME']


    if proxy != 'Invoice':
        # loop through each ng fin file
        for i, file in enumerate(files):
            # read file, filter cols and tran types to cover
            temp_df = pd.read_parquet(file, columns = cols)\
                        .query("SRC_CUST_ID.isin(@ids) & BENE_CUST_ID.isin(@ids) & TRAN_TYPE.isin(@tran_types)", engine='python')\
                            .drop_duplicates()
            if verbose:
                print(f"Successfully Read and Processed: {file}")

            # process month column (IF ANY)
            if 'DATE_TIME' in cols:
                temp_df['MONTH'] = temp_df.DATE_TIME.max().month
                temp_df.drop(columns=['DATE_TIME'], inplace=True)

            # concat 
            if i == 0:
                out_df = temp_df.copy()
            else:
                out_df = pd.concat([out_df, temp_df], axis=0, ignore_index=True)
            #display(out_df)
    else:
        for i, file in enumerate(files):
            # read file, filter cols and tran types to cover
            temp_df = pd.read_parquet(file, columns = cols)\
                        .query("SRC_CUST_ID.isin(@ids) & BENE_CUST_ID.isin(@ids) & TRAN_TYPE.isin(@tran_types)", engine='python')\
                            .drop_duplicates()
            if verbose:
                print(f"Successfully Read and Processed: {file}")

            # process month column (IF ANY)
            if 'DATE_TIME' in cols:
                temp_df['MONTH'] = temp_df.DATE_TIME.max().month
                temp_df.drop(columns=['DATE_TIME'], inplace=True)

            # concat 
            if i == 0:
                out_df = temp_df.copy()
            else:
                out_df = pd.concat([out_df, temp_df], axis=0, ignore_index=True)

    # return concatenatred data
    print('Completed Concatenation for ng fin!')
    return out_df.reset_index(drop=True)



# def func to process outstapay data (for payroll proxy along with ng fin)
def proc_outstapay(sc_path, files, verbose=True):
    # get retail ids
    ids = set(pd.read_parquet(max(glob.glob(os.path.join(sc_path, '*scorecard*_b.parquet'))), columns=['CST_ID','scorecard_id','cl_id_grp'])\
                .query("(~scorecard_id.str.contains('ATTRITED') & cl_id_grp.isin(['b.Retail']))", engine='python').CST_ID)

    # loop through each file 
    for i, file in enumerate(files):
        # read file, filter out Proxy txns and IRIS txns
        temp_df = pd.read_parquet(file, columns=['TRANSMITTING_CST_ID', 'TRANSMITTING_AR_ID', 'RECEIVING_AL_ID', 'TXN_TYPE', 'CHANNEL', 'TXN_AMOUNT', 'DATE', 'TIME'])\
                        .rename(columns={'TRANSMITTING_CST_ID':'SRC_CUST_ID', 'TRANSMITTING_AR_ID':'SRC_ARID', 'RECEIVING_AL_ID':'BENE_CUST_ID', 'TXN_TYPE':'TRAN_TYPE'})\
                            .query("SRC_CUST_ID.isin(@ids) & TRAN_TYPE.isin(['IPY','QRP']) & CHANNEL != 'IRIS'", engine='python')\
                                .drop(columns=['CHANNEL'])\
                                    .drop_duplicates(subset=['SRC_ARID','TIME'])

        # process month column (IF ANY)
        if 'DATE' in temp_df.columns:
            temp_df['MONTH'] = temp_df.DATE.max().month
            temp_df.drop(columns=['DATE'], inplace=True)

        if verbose:
             print(f"Successfully Read and Processed: {file}")

        # concat 
        if i == 0:
            out_df = temp_df.copy()
        else:
            out_df = pd.concat([out_df, temp_df], axis=0, ignore_index=True)

    print('Completed Concatenation for outstapay!')
    return out_df.reset_index(drop=True).drop(columns=['TIME', 'TXN_AMOUNT'])


# def func to process instapay incoming data (along with ng fin for payroll proxy)
def proc_instapay(sc_path, files, verbose=True):
    # get retail ids
    ids = set(pd.read_parquet(max(glob.glob(os.path.join(sc_path, '*scorecard*_b.parquet'))), columns=['CST_ID','scorecard_id','cl_id_grp'])\
                .query("(~scorecard_id.str.contains('ATTRITED') & cl_id_grp.isin(['b.Retail']))", engine='python').CST_ID)

    # loop through each file 
    for i, file in enumerate(files):
        # read file, filter out Proxy txns and IRIS txns 
        temp_df = pd.read_parquet(file, columns=['RECEIVING_CST_ID', 'RECEIVING_AR_ID','TRANSMITTING_AL_ID', 'TXN_TYPE','TXN_AMOUNT', 'DATE', 'TIME'])\
                        .rename(columns={'TRANSMITTING_AL_ID':'SRC_CUST_ID', 'RECEIVING_CST_ID':'BENE_CUST_ID', 'RECEIVING_AR_ID': 'BENE_ARID', 'TXN_TYPE':'TRAN_TYPE'})\
                            .query("BENE_CUST_ID.isin(@ids) & TRAN_TYPE.isin(['IPY','QRP'])", engine='python')\
                                .drop(columns=['TRAN_TYPE'])\
                                    .drop_duplicates(subset=['BENE_ARID','TIME'])

        # process month column (IF ANY)
        if 'DATE' in temp_df.columns:
            temp_df['MONTH'] = temp_df.DATE.max().month
            temp_df.drop(columns=['DATE'], inplace=True)

        if verbose:
             print(f"Successfully Read and Processed: {file}")

        # concat 
        if i == 0:
            out_df = temp_df.copy()
        else:
            out_df = pd.concat([out_df, temp_df], axis=0, ignore_index=True)

    print('Completed Concatenation for instapay!')
    return out_df.reset_index(drop=True).drop(columns=['TIME'])
        
        
# def func to process and concat ng non fin data 
def proc_ng_nonfin(sc_path, files, verbose=True):
    # download latest ppm file and get retail ids
    ids = set(pd.read_parquet(max(glob.glob(os.path.join(sc_path, '*scorecard*_b.parquet'))), columns=['CST_ID', 'scorecard_id','cl_id_grp'])\
                .query("(~scorecard_id.str.contains('ATTRITED') & cl_id_grp.isin(['b.Retail']))", engine='python').CST_ID)
    
    # loop through each ng non fin file 
    for i, file in enumerate(files):
        # read file, filter cols select required tran type
        temp_df = pd.read_parquet(file, columns=['SRC_CUST_ID','ACTION_MADE','DATE_TIME'])\
                    .query("SRC_CUST_ID.isin(@ids) & ACTION_MADE=='Inquiry'", engine='python').dropna(subset=['SRC_CUST_ID'])

        if verbose:
            print(f"Successfully Read and Processed: {file}")

        # process month col if any
        if 'DATE_TIME' in temp_df.columns:
            temp_df['MONTH'] = temp_df.DATE_TIME.max().month
            temp_df.drop(columns=['DATE_TIME'], inplace=True)
        
        # concat
        if i == 0:
            out_df = temp_df.copy()
        else:
            out_df = pd.concat([out_df, temp_df], axis=0, ignore_index=True)

    # return concatenatred data
    print('Completed Concatenation for ng non fin!')
    return out_df.reset_index(drop=True)

# def func to process ari cleaned buds data
def proc_buds(path, files, verbose=True):
    # declare columnst to use for buds 
    columns = ['BRANCH','TRAN_NAME','AR_ID','CST_ID','MARKET_SEGMENT','TRAN_AMOUNT','BUSINESS_DATE']

    # take max date to filter overflows or underflows
    frmt = 'cleaned_buds_ej_%Y_%m.parquet'
    enddate = datetime.strptime(files[-1].split('\\')[-1], frmt)
    print(f"Edge: {enddate}")

    # concat all data
    BUDS_df = pd.concat(
                [
                (
                    pd.read_parquet(file_path,columns=columns)
        
                    #only get the DEPOSITS and NON CORP CLIENTS 
                    .query("(CST_ID != 0) & TRAN_NAME.str.contains('DEPOSIT') ", engine ='python')
                    
                    
                    # sort by date and add month col
                    .sort_values(by = 'BUSINESS_DATE',ascending=True)
                    .assign(MONTH = lambda x: x.BUSINESS_DATE.astype('datetime64[M]'))
            
        
                )
                for file_path in tqdm(files)
                ]
                        )
    # return file 
    return BUDS_df.loc[BUDS_df.BUSINESS_DATE <= enddate]



## FUNCS FOR PROCESSING PROXY FEATURES
#-----------------------------------------------------------------------------------------------------

# features for branch (deposits and invoice) proxy
def get_buds_features(df):
    '''df = output of proc_buds'''

    # get overall branch deposit presence across period (avg. number of unique days of branch visits per month)
    overall_presence = (
                        df      
                        .groupby(['CST_ID','AR_ID','BRANCH','MONTH'])
                        .BUSINESS_DATE.nunique()
    
                        #include all the other months before you do an average  
                        .unstack(fill_value =0).stack()
                        
                        .reset_index()
                        .groupby(['CST_ID','AR_ID','BRANCH'])
                        .agg(
                            BRANCH_DEPOSIT_PRESENCE = (0,'mean')
                            )
                        )
    # get avg number of unique branches visited monthly
    monthly_uniq_br = (
                        df      
                        .groupby(['CST_ID','AR_ID','MONTH'])
                        .BRANCH.nunique()
        
                        #include all the other months before you do an average  
                        .unstack(fill_value =0).stack()
                        
                        .reset_index()
                        .groupby(['CST_ID','AR_ID'])
                        .agg(
                            MONTHLY_UNIQUE_BRANCHES = (0,'mean')
                            )
                        )
    
    # merge and add column for unique total branches
    working_df = pd.merge(overall_presence, monthly_uniq_br, on=['CST_ID','AR_ID'])\
                    .merge(df[['CST_ID','AR_ID','BRANCH']], on=['CST_ID','AR_ID'])\
                        .pipe(lambda x: x.assign(NO_OF_UNIQUE_BRANCHES=x.groupby(['CST_ID','AR_ID']).BRANCH.transform('nunique')))
    

    # getting proxy flags
    final = working_df.assign(BR_DEPOSIT_JTBD_MANUAL = lambda x: ((x.BRANCH_DEPOSIT_PRESENCE >= 2) & (x.MONTHLY_UNIQUE_BRANCHES >=1) ).astype(int),
                              RECEIVE_JTBD_MANUAL = lambda x:(((x.BRANCH_DEPOSIT_PRESENCE >=1) & (x.NO_OF_UNIQUE_BRANCHES >=5))).astype(int)
                                )
    
    # bring to cst id levl (choose most active ARID based on branch presence)
    return final.sort_values(by=['BRANCH_DEPOSIT_PRESENCE'], ascending=False).groupby('CST_ID').head(1)

# payroll proxy func
def get_pyrl_features(pyrl_df, scope='CST_ID'):
    '''
    Returns the following features:
    MEAN_TRAN_TO_UNIQ_BENE - Mean transaction count to unique BENE_CUST_IDs per month
    TOT_UNIQ_BENE_OMNIPRESENT - Total count of unique BENE_CUST_IDS present in at least one txn per month across entire date range
    SELF_EMPLOYED_USES_BPI - base target flags (loose since this is pre 'uniproxy' tightening)
    '''
    
    # get mean monthly transfer count to each unique BENE ID
    a = np.floor(pyrl_df.groupby(['SRC_ARID','BENE_CUST_ID','MONTH']).agg(MTH_COUNT = ('MONTH','count'))\
                    .reset_index().groupby(['SRC_ARID']).agg(MEAN_TRAN_TO_UNIQ_BENE = ('MTH_COUNT','mean'))).reset_index()


    # get number of unique ids present across all months
    b = np.ceil(pyrl_df.groupby(['SRC_ARID','BENE_CUST_ID']).agg(MONTHS_PRESENT=('MONTH','nunique'))\
                            .query('MONTHS_PRESENT==6').reset_index()\
                                .groupby(['SRC_ARID']).agg(TOT_UNIQ_BENE_OMNIPRESENT=('SRC_ARID','count'))\
                                    .reset_index())

    # merge both to get full payroll proxy features (HAS TO BE OUTER SINCE NOT ALL WILL HAVE OMNI RECEIVERS)
    pay = a.merge(b, on='SRC_ARID', how='outer')\
                .rename(columns={'SRC_ARID':'AR_ID'})

    # fill nans for tot uniq bene omni so they wont get dropped
    pay['TOT_UNIQ_BENE_OMNIPRESENT'] = pay['TOT_UNIQ_BENE_OMNIPRESENT'].fillna(0)

    # merge back target col
    #pay = pay.merge(pyrl_df.rename(columns={'SRC_ARID':'AR_ID'})[['AR_ID', 'SELF_EMPLOYED_USES_BPI']], on='AR_ID', how='left')\
                #.drop_duplicates(subset=['AR_ID'])
    pay = pay.merge(pyrl_df.rename(columns={'SRC_ARID':'AR_ID'})[['AR_ID']], on='AR_ID', how='left')\
                .drop_duplicates(subset=['AR_ID'])

    # drop those with nans
    print(f'nan rows to drop: {pay.isna().sum()}')
    pay = pay.dropna(how='any').set_index('AR_ID')

    # return df
    return pay


# func to process invoice features
def get_invoice_features(df):
    # get mean transaction count per month for BENE
    a = df.groupby(['BENE_ARID','MONTH']).agg(TOT_MTH_INVOICE = ('BENE_ARID','count'))\
            .reset_index().groupby(['BENE_ARID']).agg(MEAN_MTH_INVOICE_COUNT = ('TOT_MTH_INVOICE', 'mean')).reset_index()

    # get mean number of unique senders per month
    b = df.groupby(['BENE_ARID','MONTH']).agg(TOT_UNIQ_SENDERS = ('SRC_CUST_ID', 'nunique'))\
            .reset_index().groupby(['BENE_ARID']).agg(MEAN_MTH_UNIQ_SENDERS = ('TOT_UNIQ_SENDERS', 'mean')).reset_index()
    
    # merge both sales df
    sales_df = a.merge(b, on='BENE_ARID', how='inner')

    # merge back the target col 
    #sales_df = sales_df.merge(df[['BENE_ARID','SELF_EMPLOYED_USES_BPI']], on='BENE_ARID', how='left')\
                        #.drop_duplicates(subset=['BENE_ARID'])
    sales_df = sales_df.merge(df[['BENE_ARID']], on='BENE_ARID', how='left')\
                            .drop_duplicates(subset=['BENE_ARID'])
    
    # drop those with nans
    print(f'nan rows to drop: {sales_df.isna().sum()}')

    # return features for invoice proxy
    return sales_df.dropna(how='any').rename(columns={'BENE_ARID':'AR_ID'}).set_index('AR_ID')

# func to process balance inq features
def get_bal_inq_features(df):
    '''
    Returns the following features:
    MEAN_MTH_INQ - mean number of balance inquiries made on NG per month
    '''
    # get mean monthly total balance inquiries 
    bal_inq = df.groupby(['SRC_CUST_ID','MONTH']).agg(TOT_MTH_INQ = ('ACTION_MADE','count')).reset_index().rename(columns={'SRC_CUST_ID':'CST_ID'})\
                            .groupby(['CST_ID']).agg(MEAN_MTH_INQ = ('TOT_MTH_INQ','mean')).reset_index()

    # merge back target col
    bal_inq = bal_inq.merge(df.rename(columns={'SRC_CUST_ID':'CST_ID'})[['CST_ID','SELF_EMPLOYED_USES_BPI']], on='CST_ID', how='left')\
                        .drop_duplicates(subset=['CST_ID']).dropna(subset=['SELF_EMPLOYED_USES_BPI'])

    # set rows with nan to 0
    bal_inq['MEAN_MTH_INQ'].fillna(0)
    bal_inq.dropna(inplace=True)

    # return
    return bal_inq.set_index('CST_ID')