'''
Extract autoencoder features for unsupervised clustering
'''

from keras.models import load_model
import keras.backend as K
import numpy as np
import csv
from motcube_preprocessing import *
import os
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

def ae_feature_extractor(val_dir, model, out_layer=18, batch_size = 12, target_size=(156,156,100), out_size=(9,9,6), flat=True, conv_feat=False):
    '''
    Predicts classes from a validation set using a Keras model

    Parameters
    ----------
    val_dir : string.
        path to directory of validation samples, stored in class specific
        dir's per Keras standard practice.
    model : Keras model object.
        model used to generate predictions.
    out_layer : integer.
        model layer to collect output.
    batch_size : integer.
        size of batches for prediction.
    target_size : tuple of integers.
        tuple specifying the input sample size. (x, y, z).
    out_size : tuple of integers.
        tuple specifying the output sample size. (x, y, z).
        if features are flat, single integer value.
    flat : boolean.
        return features as a flattened N x M feature matrix.
    conv_feat : boolean.
        convolutional features, reduce by taking mean/stdev
        pixelwise across all filter weights.

    Returns
    -------
    yfeat : ndarray.
        N x M ndarray of features extracted from autoencoder compressed layers.
    ytrue : ndarray.
        N x 1 ndarray of ground truth class labels.
    classes : dict.
        dict of class names and corresponding index labels.
    '''

    valgen = MotcubeDataGenerator()
    val_generator = valgen.flow_from_directory(val_dir, class_mode='categorical', color_mode='grayscale', target_size = target_size, batch_size = batch_size, shuffle=False)

    ytrue = val_generator.classes
    classes = val_generator.class_indices

    extract_features = K.function(
                [model.layers[0].input, K.learning_phase()],
                [model.layers[out_layer].output])

    # Compress feature sets by taking mean and stdev of each pixel
    # in x,y,z across all filter kernels
    # Reduces dimensionality from K*x*y*z to 2*x*y*z
    i = 0

    if conv_feat:
        yfeat = np.zeros([val_generator.nb_sample, 2, out_size[0], out_size[1], out_size[2]])
    else:
        yfeat = np.zeros([val_generator.nb_sample, out_size])

    while i < (val_generator.nb_sample/batch_size):
        ybatch = extract_features([next(val_generator)[0], 0])[0]

        if conv_feat:
            fmeans = np.mean(ybatch, axis = 1)
            fstdevs = np.std(ybatch, axis = 1)
            bfeat = np.stack([fmeans, fstdevs])
            bfeat = np.swapaxes(bfeat, 0, 1) # restores (samp, chan, x, y, z)

            yfeat[i*batch_size:(i+1)*batch_size,:,:,:,:] = bfeat
        else:
            yfeat[i*batch_size:(i+1)*batch_size, :] = ybatch

        i = i + 1

    if flat:
        yfeat = np.reshape( yfeat, ( yfeat.shape[0], np.prod(yfeat.shape[1:]) ) )

    return yfeat, ytrue, classes

def write_ae_feature(yfeat, save_dir, exp_name):
    '''
    Writes autoencoder features to a CSV
    '''
    np.savetxt(os.path.join(save_dir, exp_name + 'ae_features.csv'), yfeat, delimiter=',')
    return

def kmeans_cluster_plot(X, y, k=3, cmap=plt.cm.Paired, n_dim=2):
    kmeans = cluster.KMeans(n_clusters=k).fit(X)
    klab = kmeans.labels_
    fig = plt.figure()

    if n_dim < 3:
        plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap)
    else:
        ax = Axes3D(fig)
        ax.scatter3D(X[:,0], X[:,1], X[:,2], c=y, cmap=cmap)
    plt.title('k-means clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    if n_dim > 2:
        plt.zlabel('PC3')

    return fig

def hcluster_plot(X, y, linkage='ward'):
    '''
    Parameters
    ----------
    X : N x M feature matrix
    y : N x 1 labels_
    linkage : string. ['average', 'ward', 'complete'].
        heirarchical cluster linkage mechanism to implement.
    '''
    df = pd.DataFrame(X)
    df.y = y

    labcols = pd.Series(y).map(dict(zip(np.unique(df.y), sns.color_palette('pastel', len(np.unique(df.y))))))

    fig = plt.figure()
    p = sns.clustermap(df, row_colors=labcols, col_cluster=0,
                    method=linkage, yticklabels=False, xticklabels=False,
                    linewidths=0)

def tsne_plot(X, y, class_names=[], cmap=plt.cm.Paired, n_components=2, perplexity=30, n_iter=1000):
    model = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
    Xt = model.fit_transform(X)

    fig = plt.figure()
    plt.scatter(Xt[:,0], Xt[:,1], c=y, cmap=cmap)

    if class_names:
        colors = cmap.colors[:len(class_names)]
        labs = []
        for i in range(len(class_names)):
            labs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
        plt.legend(labs, class_names, loc=0)

    plt.title('tSNE')

    return fig


def main():
    parser = argparser.ArgumentParser()
    parser.add_argument('val_dir', help='path to validation data, organized by class per keras convention')
    parser.add_argument('model_path', help='path to trained model weights')
    parser.add_argument('--batch_size', default=12, help='batch size for processing')
    parser.add_argument('--target_size', default = 256, nargs = '+', help='target size x y z')

    return
