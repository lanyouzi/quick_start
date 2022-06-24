from curses import A_ALTCHARSET
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import seaborn as sns
import json
import numpy as np
import pickle
from scipy.signal import find_peaks
import pandas as pd
import json
import numpy as np
import os
import shutil
import pylab as plt
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
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap, SpectralEmbedding
import time
# pd.set_option('display.unicode.ambiguous_as_wide', True)
# pd.set_option('display.unicode.east_asian_width', True)

def calculate_fi(fis):
    N = len(fis)
    T = 100.0
    yf = rfft(fis)
    abs_yf = np.abs(yf)
    ps = np.square(abs_yf)
    xf = np.linspace(0, T/2, len(ps))
    loco_l = 0
    loco = 0
    for i in range(len(xf)):
        if xf[i] < 3:
            if i + 1 < len(xf):
                loco_l = min(3, xf[i + 1]) - max(0.5, xf[i])
            else:
                loco_l = 3 - max(0.5, xf[i])
            if loco_l > 0:
                loco += loco_l * ps[i]
    freeze_l = 0
    freeze = 0
    for i in range(len(xf)):
        if xf[i] < 8:
            if i + 1 < len(xf):
                freeze_l = min(8, xf[i + 1]) - max(3, xf[i])
            else:
                freeze_l = 8 - max(3, xf[i])
            if freeze_l > 0:
                freeze += freeze_l * ps[i]
    wc_r2 = 1/(T / 2)**2
    norm_yf = abs_yf/np.linalg.norm(abs_yf)
    smooth = 0
    dw = T/2/len(xf)
    for i in range(len(xf)):
        smooth -= math.sqrt(wc_r2 + norm_yf[i]) * dw
    return freeze/loco, smooth


def calculate_phase(N):
    T = N/100.0
    return T*0.6, T*0.4
    
def calculate_vcc(x, y, z, gx, gy, gz):
    acc = np.array([x, y, z])
    grav = np.array([gx, gy, gz])
    vertical_acc = np.dot(acc, grav) / np.dot(grav, grav) * grav
    return -vertical_acc[0]

def segmentation(data, seg, drop_last=True):
    '''
    a generator returns *seg* items each time
    :param data: data
    :param seg: length of each segment or list of length
    :param drop_last: whether drop the last few items
    :return: segmented data
    '''
    count = 0
    if isinstance(seg, list):
        seg = iter(seg)
        seg_length = seg.__next__()
    else:
        seg_length = seg
    while count + seg_length <= data.shape[0]:
        next_index = count + seg_length
        yield data[count:next_index]
        count = next_index
        if not isinstance(seg, int):
            try:
                seg_length = seg.__next__()
            except StopIteration as stop:
                return
    if not drop_last and count < data.shape[0]:
        yield data[count:]

def loader(data, batch_size):
    start = 0
    while start + batch_size <= data.shape[1]:
        yield data[:,start: start + batch_size]
        start += batch_size

def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    cs_y = CubicSpline(xx[:,1], yy[:,1])
    cs_z = CubicSpline(xx[:,2], yy[:,2])
    return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    return tt_cum

def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])
    X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
    X_new[:,2] = np.interp(x_range, tt_new[:,2], X[:,2])
    return X_new

def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X , axangle2mat(axis,angle))

def query_csv(csv_file, data_path, feature_save_path, hc_path, feature_aug_path, hc_aug_path, is_back=False, rate = 1):
    #os.mkdir(feature_save_path)
    #os.mkdir(hc_path)
    #os.mkdir(feature_aug_path)
    #os.mkdir(hc_aug_path)
    df = pd.read_csv(csv_file)
    print(feature_save_path)
    taps = signal.firwin (219, [0.01, 0.06], pass_zero=False)
    count = 0 #count of data
    count_all = 0 #count includes augmentation
    # loop through each row
    time.sleep(1)

    if is_back:
        itm_name = "deviceMotion_walking_return.json.items"
    else:
        itm_name = "deviceMotion_walking_outbound.json.items"

        
    for filename, record_id, med in tqdm(zip(df[itm_name], df["recordId"], df["medTimepoint"]), total = df.shape[0]):
        if not check[med]:
            continue
        
        if filename == '0':
            continue
        
        filename = os.path.join(data_path, filename)
#         print(filename + " " + str(scount));
        with open(filename) as json_data:
            data = json.load(json_data)
            x_accel = []  # initialize empty list for storing x-acceleration values
            y_accel = []
            z_accel = []
            x_rotat = []  # initialize empty list for storing x-acceleration values
            y_rotat = []
            z_rotat = []
            fi_vertical = []

            if not data or len(data) <= 200:
                continueo.
                x = item.get("userAcceleration").get("x")
                y = item.get("userAcceleration").get("y")
                z = item.get("userAcceleration").get("z")
                gx = item.get("gravity").get("x")
                gy = item.get("gravity").get("y")
                gz = item.get("gravity").get("z")

                fi = calculate_vcc(x, y, z, gx, gy, gz)
                fi_vertical.append(fi)
                x_accel.append(x)
                y_accel.append(y)
                z_accel.append(z)
                # Extract
                x = item.get("rotationRate").get("x")
                y = item.get("rotationRate").get("y")
                z = item.get("rotationRate").get("z")
                x_rotat.append(x)
                y_rotat.append(y)
                z_rotat.append(z)
            try:
                x_accel = np.array(x_accel)
                y_accel = np.array(y_accel)
                z_accel = np.array(z_accel)
                fi_vertical = np.array(fi_vertical)
                Acc = np.sqrt(x_accel**2 + y_accel**2 + z_accel**2)
                normalize = lambda x: x/x.max()
                Acc = normalize(Acc)
                x_accel = x_accel.tolist ()
                y_accel = y_accel.tolist ()
                z_accel = z_accel.tolist ()
                Acc_ff = signal.filtfilt (taps, 1.0, Acc)
                Acc_ff = normalize (Acc_ff)
                peaks, _ = find_peaks (Acc_ff, height=0, distance=80)
                accel_array = np.array([x_accel, y_accel, z_accel])
                rotat_array = np.array([x_rotat, y_rotat, z_rotat])
                result = np.vstack((accel_array, rotat_array))
            
                num = 0
                l = segmentation((result.T)[peaks[0]:], list(np.diff(peaks)))
                fi_l = segmentation(fi_vertical[peaks[0]:], list(np.diff(peaks)))
                index = 0
                for (batch, fis) in zip(l, fi_l): 
                    nbatch = batch.flatten()
                    fig = plt.figure(figsize=(8, 6), dpi=80)
                    ax1 = plt.axes(frameon=False)
                    ax1.set_frame_on(False)
                    ax1.get_xaxis().tick_bottom()
                    ax1.axes.get_yaxis().set_visible(False)
                    ax1.axes.get_xaxis().set_visible(False)
                    plt.axis('off')
                    pxx, freq, t, cax = plt.specgram(nbatch, NFFT=16, vmin=-120, vmax=30, Fs=100, noverlap=8)
                    # plt.colorbar(cax)
                    # plt.colorbar('off')
                    fig = plt.gcf()
                    fig_path = os.path.join(feature_save_path, record_id + "_"+ str(num) + '.png')
                    fig.savefig(fig_path, bbox_inches='tight', pad_inches=0)
                    plt.close(o)
                    fi_score, smooth = calculate_fi(fis)
                    stance, swing = calculate_phase(len(fis))
                    hc_feature = np.array([fi_score, stance, swing, smooth])
                    np.save(os.path.join(hc_path, record_id + "_"+ str(num) + '.npy'), hc_feature)
                    count += 1
                    count_all += 1
                    type_count = 0
                    aug_num = 0
                    while count_all/count<rate:
                        temp_batch = batch
                        acc_batch = batch[:,0:3]
                        gyro_batch = batch[:,3:6]
                        if(type_count < 5):
                            temp_batch[:,0:3] = DA_TimeWarp(acc_batch)
                            temp_batch[:,3:6] = DA_TimeWarp(gyro_batch)
                        else:
                            temp_batch[:,0:3] = DA_Rotation(DA_TimeWarp(acc_batch))
                            temp_batch[:,3:6] = DA_Rotation(DA_TimeWarp(gyro_batch))
                        count_all+=1
                        type_count+=1
                        nbatch = temp_batch.flatten()
                        fig = plt.figure(figsize=(8, 6), dpi=80)
                        ax1 = plt.axes(frameon=False)
                        ax1.set_frame_on(False)
                        ax1.get_xaxis().tick_bottom()
                        ax1.axes.get_yaxis().set_visible(False)
                        ax1.axes.get_xaxis().set_visible(False)
                        plt.axis('off')
                        pxx, freq, t, cax = plt.specgram(nbatch, NFFT=16, vmin=-120, vmax=30, Fs=100, noverlap=8)
                        # plt.colorbar(cax)
                        # plt.colorbar('off')
                        fig = plt.gcf()
                        fig_path = os.path.join(feature_aug_path, record_id + "_"+ str(num) + "_" + str(aug_num) + '.png')
                        fig.savefig(fig_path, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        fi_score, smooth = calculate_fi(fis)
                        stance, swing = calculate_phase(len(fis))
                        hc_feature = np.array([fi_score, stance, swing, smooth])
                        np.save(os.path.join(hc_aug_path, record_id + "_"+ str(num) + "_" + str(aug_num) + '.npy'), hc_feature)
                        aug_num += 1
                    num += 1
                            
            except:
                continue


if __name__=='__main__':  
    data_dir = '/mnt/DataCenter/maruokai/datasets/pd'
    is_back = False

    if is_back:
        result_dir = os.path.join(data_dir, 'result_backward')
    else:
        result_dir = os.path.join(data_dir, 'result')
        
    pd_gait_path = os.path.join(data_dir, 'pd_gait.csv')
    pd_survey_path = os.path.join(data_dir, 'pd_survey.csv')
    pd_voice_path = os.path.join(data_dir, 'pd_voice.csv')

    if not os.path.exists(pd_gait_path):
        print("gait path not exist")
        


    save_path = os.path.join(result_dir, 'feature_seg')
    hc_save_path = os.path.join(result_dir, 'hc_feature')
    aug_path = os.path.join(result_dir, 'feature_aug')
    aug_hc_save_path = os.path.join(result_dir, 'aug_hc_feature')

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(hc_save_path):
        os.mkdir(hc_save_path)
    if not os.path.exists(aug_path):
        os.mkdir(aug_path)
    if not os.path.exists(aug_hc_save_path):
        os.mkdir(aug_hc_save_path)
        
    check = {
        "Immediately before Parkinson medication" : True,
        "I don't take Parkinson medications" : False,
        "Another time" : False,
        "Just after Parkinson medication (at your best)" : True,
    }

    query_csv(pd_gait_path, data_dir, save_path, hc_save_path, aug_path, aug_hc_save_path, is_back = is_back)