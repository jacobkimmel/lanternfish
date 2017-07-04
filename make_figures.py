'''
Make Figures for Lanternfish Manuscript
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

def load_data_sets(csv_dir, pattern, nonpattern=None):
    '''
    Load CSVs into a list of pd DataFrames for plotting
    '''

    f = glob.glob( os.path.join(csv_dir, pattern) )

    # remove unwanted files
    ridx = []
    if nonpattern:
        for i in range(len(f)):
            for p in nonpattern:
                if p in f[i]:
                    ridx.append(i)
    f = [i for j, i in enumerate(f) if j not in ridx]

    f.sort()

    names = []
    data = []
    for i in range(len(f)):
        n = f[i].split('/')
        df = pd.read_csv(f[i])

        names.append(n[-1])
        data.append(df)

    return data, names

def plot_group_training(df, save_dir, save_name, title='',
                        width=4, height=3, legend_loc='lower right',
                        xlim=None, ylim=None, ylabel='Accuracy'):
    '''
    Plots all lines for training data in a df on a single plot

    Parameters
    ----------
    df : pandas DataFrame.
        columns named epoch, ...
        where all other columns will be plotted and labeled with the column
        title as the label

    Returns
    -------
    Saved plot at save_dir + save_name
    '''
    groups = list(df.columns)
    groups.remove('epoch')


    plt.figure(figsize=(width, height))
    for i in range(len(groups)):
        plt.plot(df[groups[i]])

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend(groups, loc=legend_loc)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, save_name))

    return

def plot_grouped_bars(df, save_dir, save_name, width=4, height=3, x='Size', y='Peak Acc.', hue='Type', title=''):

    plt.figure(figsize=(width,height))
    ax = sns.barplot(x=x, y=y, hue=hue, data=df)
    ax.legend_.remove() # legend overlaps weirdly
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, save_name))

    return


#--------
# Fig 3
#--------

# Load non-augmented data sets
nda_dfs, nda_names = load_data_sets('notebook/','20170515*multiclass*.csv',['dynaug','adadelta'])
# Load augmented data sets
yda_dfs, yda_names = load_data_sets('notebook/','20170515*multiclass*dynaug*.csv',['adadelta'])


# Plot non-augmented binary and gaussian large beside each other
bin25idx = [i for i, s in enumerate(nda_names) if 'bin_disk25' in s]
gauss20idx = [i for i, s in enumerate(nda_names) if 'gauss_sig20' in s]
nda_bin25 = nda_dfs[bin25idx[0]]
nda_gauss20 = nda_dfs[gauss20idx[0]]

acc_df = pd.concat([nda_bin25['epoch'], nda_bin25['acc'], nda_bin25['val_acc'],
                    nda_gauss20['acc'], nda_gauss20['val_acc']], 1)
acc_df.columns = ['epoch', 'Bin25 Training', 'Bin25 Validation', 'Gauss20 Training', 'Gauss20 Validation']

plot_group_training(acc_df, 'notebook/figs/', 'lanternfish_fig3A_noaug.png', 'Sim. Classification')

# Plot augmented binary and gaussian large beside each other
bin25idx = [i for i, s in enumerate(yda_names) if 'bin_disk25' in s]
gauss20idx = [i for i, s in enumerate(yda_names) if 'gauss_sig20' in s]
yda_bin25 = yda_dfs[bin25idx[0]]
yda_gauss20 = yda_dfs[gauss20idx[0]]

acc_df = pd.concat([yda_bin25['epoch'], yda_bin25['acc'], yda_bin25['val_acc'],
                    yda_gauss20['acc'], yda_gauss20['val_acc']], 1)
acc_df.columns = ['epoch', 'Bin25 Training', 'Bin25 Validation', 'Gauss20 Training', 'Gauss20 Validation']

plot_group_training(acc_df, 'notebook/figs/', 'lanternfish_fig3B_dynaug.png', 'Sim. Classification')

# Plot comparisons of representation size

acc_df = [ yda_dfs[0]['epoch'] ]
for i in range(3):
    acc_df.append(yda_dfs[i]['acc'])
    acc_df.append(yda_dfs[i]['val_acc'])
acc_df = pd.concat(acc_df, 1)
acc_df.columns = ['epoch', 'Bin01 Training', 'Bin01 Validation', 'Bin05 Training', 'Bin05 Validation', 'Bin25 Training', 'Bin25 Validation']

plot_group_training(acc_df, 'notebook/figs/', 'lanternfish_fig3C_szComparison.png', 'Sim. Classification')

# Load large data sets

large_dfs, large_names = load_data_sets('notebook/','20170604*largeN*.csv')
bin25idx = [i for i, s in enumerate(large_names) if 'bin_disk25' in s]
gauss20idx = [i for i, s in enumerate(large_names) if 'gauss_sig20' in s]

large_bin25 = large_dfs[bin25idx[0]]
large_gauss20 = large_dfs[gauss20idx[0]]

acc_df = pd.concat([large_bin25['epoch'], large_bin25['acc'], large_bin25['val_acc'],
                    large_gauss20['acc'], large_gauss20['val_acc']], 1)
acc_df.columns = ['epoch', 'Bin25 Training', 'Bin25 Validation', 'Gauss20 Training', 'Gauss20 Validation']
plot_group_training(acc_df, 'notebook/figs/', 'lanternfish_fig3D_largeN.png', 'Sim. Classification')

# Plot peak validation accuracies of dyn aug subsets

peak_val = pd.DataFrame(np.zeros((len(large_dfs), 3)))
peak_val.columns = ['Peak Acc.', 'Type', 'Size']
for n in range(len(large_dfs)):
    peak_val.iloc[n,0] = np.max(large_dfs[n]['val_acc'])

    if 'bin' in yda_names[n]:
        t = 'Binary'
    else:
        t = 'Gaussian'
    peak_val.iloc[n,1] = t

    if any(sz in large_names[n] for sz in ['disk01', 'sig03']):
        size = 'Small'
    elif any(sz in large_names[n] for sz in ['disk05', 'sig10']):
        size = 'Medium'
    else:
        size = 'Large'
    peak_val.iloc[n,2] = size

plot_grouped_bars(peak_val, 'notebook/figs', 'lanternfish_fig3E.png')

#--------
# Fig 4
#--------

# Load MuSC v MEF classification data
no_tranfer_df, no_transfer_names = load_data_sets('notebook/', '20170405*no_transfer*.csv')
transfer_df, transfer_names = load_data_sets('notebook/', '20170405*.csv', nonpattern=['no_transfer'])

acc_df = pd.concat([no_tranfer_df[0]['epoch'], no_tranfer_df[0]['acc'], no_tranfer_df[0]['val_acc'],
                    transfer_df[0]['acc'], transfer_df[0]['val_acc']],1)
acc_df.columns = ['epoch', 'Training (de novo)', 'Validation (de novo)',
                    'Training (transfer)', 'Validation (transfer)']

plot_group_training(acc_df, 'notebook/figs/', 'lanternfish_fig4A.png', 'MuSC v. MEF classification')

# Load MuSC v Myoblast classification data
muscmyo_df, muscmyo_names = load_data_sets('notebook/', '20170502*.csv')
acc_df = pd.concat([muscmyo_df[0]['epoch'], muscmyo_df[0]['acc'], muscmyo_df[0]['val_acc']],1)
acc_df.columns = ['epoch', 'Training', 'Validation']

plot_group_training(acc_df, 'notebook/figs/', 'lanternfish_fig4B.png', 'MuSC v. Myoblast Classification', legend_loc='lower center')

# Load WT v Neoplastic MEF data

wt_mr_df, wt_mr_names = load_data_sets('notebook/', '20170406*.csv')
acc_df = pd.concat([wt_mr_df[0]['epoch'], wt_mr_df[0]['acc'], wt_mr_df[0]['val_acc'],
                    wt_mr_df[1]['acc'], wt_mr_df[1]['val_acc']],1)
acc_df.columns = ['epoch', 'Training (de novo)', 'Validation (de novo)', 'Training (transfer)', 'Validation (transfer)']

plot_group_training(acc_df, 'notebook/figs/', 'lanternfish_fig4C.png', 'MEF WT v. Neoplastic', ylim=(0.2,0.9))


# Load MuSC v. Myo mimetic classifier training data

mim_df, mim_names = load_data_sets('notebook/', '20170524*.csv')
acc_df = pd.concat([mim_df[0]['epoch'], mim_df[0]['acc'], mim_df[0]['val_acc']], 1)
acc_df.columns = ['epoch', 'Training', 'Validation']

plot_group_training(acc_df, 'notebook/figs/', 'lanternfish_fig4E.png', 'Myogenic Mimetic Simulation Classification', ylim=(0.8,1))

# Load MuSC v. Myoblast + mimetic pretraining
myo_mim_df, myo_mim_names = load_data_sets('notebook/', '20170601*.csv')
acc_df = pd.concat([myo_mim_df[0]['epoch'], myo_mim_df[0]['acc'], myo_mim_df[0]['val_acc']], 1)
acc_df.columns = ['epoch', 'Training', 'Validation']

plot_group_training(acc_df, 'notebook/figs/', 'lanternfish_fig4F.png', 'MuSC v. Myoblast (Mimetic Pretraining)', ylim=(0.8, 1))

#--------
# Fig 5
#--------

# Load autoencoder training data
ae_df, ae_names = load_data_sets('notebook/', '20170315*ae*.csv')

loss_df = ae_df[0]
loss_df.columns = ['epoch', 'Training', 'Validation']
plot_group_training(loss_df, 'notebook/figs/', 'lanternfish_fig5A.png', 'Multiclass Autoencoder Training', ylabel='Loss', legend_loc='upper right')

# Load autoencoder as pretraining for multiclass sim classification
trans_df, trans_names = load_data_sets('notebook/', '20170320_multiclass_transfer*.csv')
# Load training without autoencoder transfer
nt_df, nt_names = load_data_sets('notebook/', '20170320_multiclass_quarter*.csv')

loss_df = pd.concat([trans_df[0]['epoch'], trans_df[0]['acc'], trans_df[0]['val_acc'], nt_df[0]['acc'], nt_df[0]['val_acc']], 1)
loss_df.columns = ['epoch', 'Transfer Training', 'Transfer Validation', 'De novo Training', 'De novo Validation']

plot_group_training(loss_df, 'notebook/figs/', 'lanternfish_fig5E.png', 'Classification with Transfer Learning')
