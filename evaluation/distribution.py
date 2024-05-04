import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

def calculate_distribution(data: pd.DataFrame) -> pd.DataFrame:
    return data.describe().transpose()

def filter_nan(distances: list) -> list:
    dists = []
    begin, end = 0, 0
    for i, distance in enumerate(distances):
        if not math.isnan(distance):
            dists.append(distance)
            if begin == 0:
                begin = i
            end = i
        else:
            dists.append(0)
    return dists, begin, end

def convolve(data: list) -> list:
    return np.convolve(data, np.ones(10), 'valid') / 10

def calculate_curve(data: pd.DataFrame, range_t: list) -> pd.DataFrame:
    dice = data['Dice loss']
    jaccard = data['Jaccard index']
    euc_dist = data['Euclidean dist']
    jaccard = jaccard[range_t[0]:range_t[1]]
    dice = dice[range_t[0]:range_t[1]]
    euc, begin, end = filter_nan(euc_dist[range_t[0]:range_t[1]])
    return convolve(dice), convolve(jaccard), convolve(euc), begin, end

def save_plot(y1: list, y2: list, preprocess: str, name: str, i: int):
    fig = plt.figure(i)
    fig.clear()
    x = np.arange(0, len(y1), 1)
    plt.plot(x, y1, label = preprocess + ' with preconv layer')
    plt.plot(x, y2, label= preprocess + ' with simple UNet')
    plt.axvline(x = 20, color = 'b', label = 'Root canal start')
    plt.axvline(x = len(y1)-20, color = 'r', label = 'Root canal end')
    plt.xlim(0, len(y1))
    if name == 'Dice' or name == 'Jaccard':
        plt.ylim(0, 1.05)
    else:
        plt.ylim(0, 2.5)
    plt.xlabel('Slice')
    plt.ylabel('Value')
    plt.title(name)
    plt.legend()
    plt.savefig('./finetune/' + preprocess + name + str(i) + '.png')

def create_plot(data1: pd.DataFrame, data2: pd.DataFrame, preprocess: str):
    l = len(data1['Dice loss']+1)
    avg_dices1, avg_dices2 = [], []
    avg_jaccards1, avg_jaccards2 = [], []
    eucs1, eucs2 = [], []
    longest = 0
    th = 20
    for i in range(4):
        if i == 2:
            continue
        avg_dice1, avg_jaccard1, euc1, begin1, end1 = calculate_curve(data1, [i*l//4, (i+1)*l//4])
        avg_dices1.append(avg_dice1[begin1-th:end1-th])
        avg_jaccards1.append(avg_jaccard1[begin1-th:end1-th])
        eucs1.append(euc1[begin1-th:end1-th])
        if end1 - begin1 > longest:
            longest = end1 - begin1 + 2 * th
            
        avg_dice2, avg_jaccard2, euc2, begin2, end2 = calculate_curve(data2, [i*l//4, (i+1)*l//4])
        avg_dices2.append(avg_dice2[begin1-th:end1-th])
        avg_jaccards2.append(avg_jaccard2[begin1-th:end1-th])
        eucs2.append(euc2[begin1-th:end1-th])
        if end2 - begin2 > longest:
            longest = end2 - begin2 + 2 * th
        
    y_ = np.linspace(0, longest-1, longest)
    interpol_dices1, interpol_dices2 = [], []
    interpol_jaccards1, interpol_jaccards2 = [], []
    interpol_eucs1, interpol_eucs2 = [], []
    
    for i in range(3):
        y1 = np.linspace(0, len(avg_dices1[i])-1, len(avg_dices1[i]))
        y2 = np.linspace(0, len(avg_dices2[i])-1, len(avg_dices2[i]))
        interpol_dices1.append(np.interp(y_, y1, avg_dices1[i]))
        interpol_dices2.append(np.interp(y_, y2, avg_dices2[i]))
        
        interpol_jaccards1.append(np.interp(y_, y1, avg_jaccards1[i]))
        interpol_jaccards2.append(np.interp(y_, y2, avg_jaccards2[i]))
        
        interpol_eucs1.append(np.interp(y_, y1, eucs1[i]))
        interpol_eucs2.append(np.interp(y_, y2, eucs2[i]))
    
    avg_interpol_dices1 = np.average(interpol_dices1, axis = 0)
    avg_interpol_dices2 = np.average(interpol_dices2, axis = 0)
    avg_interpol_jaccards1 = np.average(interpol_jaccards1, axis = 0)
    avg_interpol_jaccards2 = np.average(interpol_jaccards2, axis = 0)
    avg_interpol_eucs1 = np.average(interpol_eucs1, axis = 0)
    avg_interpol_eucs2 = np.average(interpol_eucs2, axis = 0)
     
    save_plot(avg_interpol_dices1, avg_interpol_dices2, preprocess, 'Dice', 0)
    save_plot(avg_interpol_jaccards1, avg_interpol_jaccards2, preprocess, 'Jaccard', 0)
    save_plot(avg_interpol_eucs1, avg_interpol_eucs2, preprocess, 'Distance', 0)
        
        

def create_boxplot(data1: np.ndarray, data2: np.ndarray, name: str, i: int):
    fig = plt.figure(1)
    fig.clear()
    plt.boxplot([data1, data2])
    plt.xticks([1, 2], ['Preconv', 'Simple'])
    plt.title(name)
    plt.savefig('./finetune/' + name + str(i) + '.png')
    

if __name__ == '__main__':
    data1 = pd.read_csv('./finetune/results_cropped_preconv/results.csv')
    data2 = pd.read_csv('./finetune/results_cropped_simple_ch/results.csv')
    data3 = pd.read_csv('./finetune/results_all_preconv/results.csv')
    data4 = pd.read_csv('./finetune/results_all_simple_ch/results.csv')
    create_plot(data1, data2, 'Cropped')
    create_plot(data3, data4, 'Resized')
    
    d1 = pd.read_csv('./finetune/results_cropped_preconv/results_roots.csv')
    d2 = pd.read_csv('./finetune/results_cropped_simple_ch/results_roots.csv')
    d3 = pd.read_csv('./finetune/results_all_preconv/results_roots.csv')
    d4 = pd.read_csv('./finetune/results_all_simple_ch/results_roots.csv')
    
    create_boxplot(d1['Euclidean dist'], d2['Euclidean dist'], 'Distance', 0)
    create_boxplot(d1['Dice loss'], d2['Dice loss'], 'Dice', 0)
    create_boxplot(d1['Jaccard index'], d2['Jaccard index'], 'Jaccard', 0)
    
    create_boxplot(d3['Euclidean dist'], d4['Euclidean dist'], 'Distance', 1)
    create_boxplot(d3['Dice loss'], d4['Dice loss'], 'Dice', 1)
    create_boxplot(d3['Jaccard index'], d4['Jaccard index'], 'Jaccard', 1)
    
    calculate_distribution(data1).to_csv('./finetune/results_cropped_preconv_distribution.csv')
    calculate_distribution(data2).to_csv('./finetune/results_cropped_simple_ch_distribution.csv')
    calculate_distribution(data3).to_csv('./finetune/results_all_preconv_distribution.csv')
    calculate_distribution(data4).to_csv('./finetune/results_all_simple_ch_distribution.csv')