#!/usr/bin/python3
'''
Plot keras model training data
'''

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import argparse
import seaborn as sns
import pandas as pd
import glob

def get_loss_dict(loss_file):
    ll = np.load(loss_file)
    l = ll['loss_history']
    loss = l.any()

    return loss

def plot_accuracy(loss, save_name=None, exp_name=None):
    plt.figure()
    plt.plot(loss['acc'])
    plt.plot(loss['val_acc'])
    plt.title('Model Accuracy')
    if exp_name:
        plt.suptitle(exp_name)
    plt.ylabel('Classification Accuracy')
    plt.xlabel('Training Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()

def plot_loss(loss, save_name=None, exp_name=None):
    plt.figure()
    plt.plot(loss['loss'])
    plt.plot(loss['val_loss'])
    plt.title('Model Loss')
    if exp_name:
        plt.suptitle(exp_name)
    plt.ylabel('Loss')
    plt.xlabel('Training Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()

def plot_from_csv(csv_path, save_dir=None, exp_name=None):
    '''
    Plot training accuracy and loss from a keras CSV log
    '''
    df = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    if df.shape[1] > 3:
        d = {'acc' : df[:,1], 'val_acc' : df[:,3], 'loss' : df[:,2], 'val_loss' : df[:,4]}
        plot_accuracy(d, save_name = os.path.join(save_dir, exp_name + '_accuracy.png'), exp_name = exp_name)
        plot_loss(d, save_name = os.path.join(save_dir, exp_name + '_loss.png'), exp_name = exp_name)
    else:
        d = {'loss' : df[:,1], 'val_loss' : df[:,2]}
        plot_loss(d, save_name = os.path.join(save_dir, exp_name + '_loss.png'), exp_name = exp_name)

def plot_peak_val_accs(csv_path, regex, save_dir, exp_name=None, col_names=None):
    '''
    Plots peak validation accuracies from models matching a provided regex

    Parameters
    ----------
    csv_path : string.
        Path to CSV logs of training performance.
    regex : string.
        Pattern for matching relevant CSV files.
    save_dir : string.
        path to save figures.
    exp_name : string.
        string to preappend to saved figures.
    col_names : list of strings.
        list of names to label columns (alphabetical order)
    '''

    f = glob.glob(os.path.join(csv_path, regex))
    f.sort()

    df = pd.DataFrame(np.zeros((len(f), 2)))
    for i in range(len(f)):
        log = pd.read_csv(f[i])
        df.iloc[i,:] = [f[i], log.val_acc.max()]

    df.columns = ['representation', 'val_acc']
    if col_names:
        df['representation'] = col_names

    g = sns.barplot(x='representation', y='val_acc', data=df)
    g.set_xlabel('Representation')
    g.set_ylabel('Peak. Val. Acc.')

    plt.savefig(os.path.join(save_dir, exp_name + '_peak_val_accs.png'))

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', help = 'path to csv')
    parser.add_argument('save_dir', help = 'dir for saved figs')
    parser.add_argument('exp_name', help = 'name for the experiment')
    args = parser.parse_args()
    csv_path, save_dir, exp_name = args.csv_path, args.save_dir, args.exp_name

    plot_from_csv(csv_path, save_dir, exp_name)
    return

if __name__ == '__main__':
    main()
