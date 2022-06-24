import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import seaborn as sns
import json
import numpy as np
import pickle
import os
import shutil
from PIL import Image
import argparse
import scipy.misc
import imageio
from scipy import signal
from scipy.signal import find_peaks
import math
from numpy.fft import rfft, fft
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from transforms3d.axangles import axangle2mat
import matplotlib.pyplot as plt
import seaborn as sns
import time
# pd.set_option('display.unicode.ambiguous_as_wide', True)
# pd.set_option('display.unicode.east_asian_width', True)

part = [
    "Immediately before Parkinson medication", 
    "Just after Parkinson medication (at your best)"
    ]

#spg_path = '../result_backward/feature_seg'
spg_path = '/mnt/DataCenter/maruokai/datasets/pd/result/feature_seg'

def find_nearest(array, value):
    return (np.abs(array - value)).argmin()

def matching(dfb, dfa, before_short):
    tb = np.array(dfb.iloc[:,0])
    ta = np.array(dfa.iloc[:,0])
    
    # binary search
    if before_short:
        idx = np.searchsorted(ta,tb) + 1
        idx = np.clip(idx, 0, len(ta)-1)
        dfa = dfa.iloc[idx,:].reset_index(drop=True)
    else:
        idx = np.searchsorted(tb,ta)
        idx = np.clip(idx, 0, len(tb)-1)
        dfb = dfb.iloc[idx,:].reset_index(drop=True)
        
    matched = pd.concat([dfa, dfb], axis=1, join='inner')
    
    return matched


def matching_spg(df, pathset):
    matched_df = pd.DataFrame(columns=['healthCode', 'record_Before', 'record_After', 'Label'])
    hc, label =  df.loc[0,['healthCode']].array[0], df.loc[0,['Label']].array[0]
    #print(df.head())
    print(label)
    count = 0
    
    for i in range(len(df)):
        name_before = []
        name_after = []
        
        for num in range(0,10000):
            cand_before = df.loc[i,['recordId_Before']].array[0] + "_" + str(num) + ".png"
            if cand_before in pathset:
                name_before.append(cand_before)
            else:
                break
                
        for num in range(0,10000):
            cand_after = df.loc[i,['recordId_After']].array[0] + "_" + str(num) + ".png"
            if cand_after in pathset:
                name_after.append(cand_after)
            else:
                break
        
        
        for nb in name_before:
            for na in name_after:
                matched_df.loc[count]=[hc, nb, na, label]
                count += 1
        
    return matched_df

def data_matching(top_user, merged_df):
    all_matched_df = None

    for user in top_user:
        print("processing on {}".format(user))
        user_df = merged_df[merged_df['healthCode'] == user]
        user_before = user_df[user_df['medTimepoint'] == part[0]][['createdOn', 'recordId']]
        user_before = user_before.rename(columns={"createdOn": "Time_Before", "recordId": "recordId_Before"})
        user_before = user_before.sort_values(by=['Time_Before'])
        user_before = user_before.reset_index(drop=True)
        
        user_after = user_df[user_df['medTimepoint'] == part[1]][['createdOn', 'recordId']]
        user_after = user_after.rename(columns={"createdOn": "Time_After", "recordId": "recordId_After"})
        user_after = user_after.sort_values(by=['Time_After'])
        user_after = user_after.reset_index(drop = True)
        
        # if one user not contain any before or after record then pass
        if len(user_after)<1 or len(user_before)<1:
            continue
        
        if len(user_before) > len(user_after):
            user_info = user_df[user_df['medTimepoint'] == part[1]][['healthCode', 'SUM']]
            matched_df = matching(user_before, user_after, False)
        else:
            user_info = user_df[user_df['medTimepoint'] == part[0]][['healthCode', 'SUM']]
            matched_df = matching(user_before, user_after, True)
        
        user_info = user_info.reset_index(drop = True)
        user_info = user_info.rename(columns={"SUM": "Label"})
        matched_df = pd.concat([user_info, matched_df], axis=1, join='inner')
        
        # remove the matched data that longer than 1 day
        matched_df = matched_df[abs(matched_df["Time_After"] - matched_df["Time_Before"])<1e8]
        matched_df = matched_df.reset_index(drop=True)
        
        # no match record then pass
        if len(matched_df)<1:
            continue
        
        pathset = set(os.listdir(spg_path))
        matched_df = matching_spg(matched_df, pathset)
        
        if all_matched_df is None:
            all_matched_df = matched_df
        else:
            all_matched_df = pd.concat([all_matched_df, matched_df])

    return all_matched_df


if __name__ == '__main__':

    data_dir = '/mnt/DataCenter/maruokai/datasets/pd' # path to the csv file
    topk = 91

    # read gait, survey and voice data
    pd_gait_path = os.path.join(data_dir, 'pd_gait.csv')
    pd_survey_path = os.path.join(data_dir, 'pd_survey.csv')
    pd_voice_path = os.path.join(data_dir, 'pd_voice.csv')

    if not os.path.exists(pd_gait_path):
        print("pd gait file is not exist")
    if not os.path.exists(pd_survey_path):
        print("pd survey file is not exist")
    if not os.path.exists(pd_voice_path):
        print("pd voice file is not exist")

    pd_gait_df = pd.read_csv(pd_gait_path, index_col=0)
    pd_survey_df = pd.read_csv(pd_survey_path, index_col=0)
    # pd_gait_df.head()


    check = {
        "Immediately before Parkinson medication", 
        "Just after Parkinson medication (at your best)"
    }

    normalized_data = pd_gait_df[pd_gait_df['medTimepoint'].isin(check)]
    normalized_data.nunique() # healthCode -> 470

    # healthCode list
    selected_df = normalized_data[['healthCode', 'createdOn', 'recordId', 'deviceMotion_walking_outbound.json.items', 'medTimepoint']]
    #selected_df = normalized_data[['healthCode', 'createdOn', 'recordId', 'deviceMotion_walking_return.json.items', 'medTimepoint']]
    selected_df = selected_df.dropna(axis = 0, how = 'any')

    usr_count = selected_df['healthCode'].value_counts()
    top_user = set(usr_count[:topk].keys())

    top_df = selected_df[selected_df['healthCode'].isin(top_user)]

    new_pd_survey_df = pd_survey_df.copy()
    new_pd_survey_df['SUM'] = np.where(new_pd_survey_df.SUM >= 10, 1, 0)
    new_pd_survey_df = new_pd_survey_df.groupby(by=['healthCode'])['SUM'].mean()
    new_pd_survey_df = new_pd_survey_df.to_frame()
    new_pd_survey_df.reset_index(inplace=True)
    new_pd_survey_df = new_pd_survey_df[new_pd_survey_df['healthCode'].isin(top_user)]
    
    #only take the users who have drug resistance or not.
    new_pd_survey_df = new_pd_survey_df[(new_pd_survey_df['SUM']==0) | (new_pd_survey_df['SUM']==1)]
    new_pd_survey_df.dropna(axis = 0, how = 'any')

    print(new_pd_survey_df['SUM'].value_counts())

    merged_df = top_df.merge(new_pd_survey_df[['healthCode','SUM']], on='healthCode', how='left')

    top_user = list(new_pd_survey_df['healthCode'])
    all_matched_df = data_matching(top_user, merged_df)
    all_matched_df.to_csv('data.csv',index=False)
