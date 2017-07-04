'''
Cell Mimesis
Generates simulated tracks based on real cell motility data
'''
import numpy as np
from scipy import stats
import pandas as pd
from functools import partial
from multiprocessing import Pool
from sklearn.decomposition import PCA
from sklearn import cluster

def load_real_data(inputX, inputY):
    X = np.loadtxt(inputX, delimiter = ',')
    Y = np.loadtxt(inputY, delimiter = ',')
    return X, Y

def load_class_labels(classification_matrix_path):
    # skip header
    M = np.loadtxt(classification_matrix_path, delimiter=',', skiprows=1)
    labels = M[:,-1]
    return labels

def displacement_matrix(X, Y, tau=1):
    '''
    Parameters
    ----------
    X, Y : ndarray, N x T of X, Y positions.
    tau : time lag for displacement calculation.

    Returns
    -------
    disp : ndarray, N x T-tau array of displacement magnitudes.
    '''
    T = X.shape[1]

    dx = X[:,tau:] - X[:,:-tau]
    dy = Y[:,tau:] - Y[:,:-tau]

    disp = np.sqrt(dx**2 + dy**2)
    return disp

def largest_pow2(num):
    '''
    Find the largest power of 2 that is less than 2X a given number num
    '''
    for i in range(0,10):
        if int(num / 2**i) > 1:
            continue
        else:
            return i-1

def rescaled_range(X, n):
        # takes a series X and subseries size n
        # finds the average rescaled range <R(n)/S(n)>
        # for all sub-series of size n
        N = len(X)
        if n > N:
            return None
        # Create subseries of size n
        num_subseries = int(N/n)
        Xs = np.zeros((num_subseries, n))
        for i in range(0, num_subseries):
            Xs[i,] = X[ int(i*n) : int(n+(i*n)) ]

        # Calculate mean rescaled range R/S
        # for subseries size n
        RS = []
        for subX in Xs:

            m = np.mean(subX)
            Y = subX - m
            Z = np.cumsum(Y)
            R = max(Z) - min(Z)
            S = np.std(subX)
            if S <= 0:
                print("S = ", S)
                continue
            RS.append( R/S )
        RSavg = np.mean(RS)

        return RSavg

def hurst_mandelbrot(X):
    '''
    Calculated Hurst coefficient using Mandlebrot's rescaled range
    method

    Parameters
    ----------
    X : ndarray, 1 x T of displacements.

    Returns
    -------
    H : float, [0, 1], Hurst coefficient.

    Notes
    -----

    for E[R(n)/S(n)] = Cn**H as n --> inf
    H : 0.5 - 1 ; long-term positive autocorrelation
    H : 0.5 ; fractal Brownian motion
    H : 0-0.5 ; long-term negative autocorrelation

    N.B. SEQUENCES MUST BE >= 18 units long, otherwise
    linear regression for log(R/S) vs log(n) will have
    < 3 points and cannot be performed
    '''
    RSl = []
    ns = []
    for i in range(0, largest_pow2(len(X))):
        ns.append(int(len(X)/2**i))
    for n in ns:
        RSl.append( rescaled_range(X, n) )

    m, b, r, pval, stderr = stats.linregress(np.log(ns), np.log(RSl))
    # H is == m
    return m

def turn_distribution(X, Y, tau = 1):
    '''
    Calculates the angle of turns between each set of two points
    in a time series of coordinates, X and Y

    Parameters
    ----------
    X, Y : ndarray, 1 x T of corresponding X and Y coordinates.
    tau : integer, time interval between points to consider for turn angles.

    Returns
    -------
    thetas : ndarray, 1 x T-(2*tau) of turn angles.

    Notes
    -----
    Considers angle between points x_i and x_i+tau, where tau < i < T-tau
    as:

        v0 = x_i - x_i-tau
        v1 = x_i+tau - x_i

        cos(theta) = v0 dot v1 / ||v0|| ||v1||
        theta = arccos(v0 dot v1 / ||v0|| ||v1||)
    '''
    T = len(X)
    thetas = np.zeros((T-(2*tau)))

    for t in range(tau, T-tau):
        v0 = np.array([ X[t], Y[t] ]) - np.array([ X[t-tau], Y[t-tau] ])
        v1 = np.array([ X[t+tau], Y[t+tau] ]) - np.array([ X[t], Y[t] ])

        if ( np.linalg.norm(v0) * np.linalg.norm(v1) ) == 0:
            thetas[t-tau] = 0
            continue
        else:
            costheta = np.dot(v0, v1) / ( np.linalg.norm(v0) * np.linalg.norm(v1) )
            # round costheta to avoid vals > 1 due to float round errors
            thetas[t-tau] = np.arccos( np.round(costheta, decimals=3) )

    return np.array(thetas)

def turn_matrix(X, Y, tau = 1):
    '''
    Generate a matrix of turn angles from sample X and Y coordinates

    Parameters
    ----------
    X : ndarray, N x T of X coordinates.
    Y : ndarray, N x T of Y coordinates.

    Returns
    -------
    turn_mat : ndarray, N x T-(2*tau) of turn angles.
    '''
    turn_mat = np.zeros([X.shape[0], X.shape[1]-(2*tau)])
    for n in range(X.shape[0]):
        turn_mat[n,:] = turn_distribution(X[n,:], Y[n,:], tau=tau)
    return turn_mat

def dist_statistics(dist):
    '''
    Parameters
    ----------
    dist : N x T ndarray of feature values.

    Returns
    -------
    s : ndarray.
        1 x 4 array of statistics of the distribution
        mean, stdev, mean kurtosis, and mean hurst coeff.
    cs : ndarray, N x 4.
        N x 5 array of statistics for each cell in the input
        distribution.
    '''
    N = dist.shape[0]
    mean_dist = np.mean(dist, axis = 1)
    median_dist = np.median(dist, axis = 1)
    std_dist = np.std(dist, axis = 1)
    kurt_dist = np.zeros([dist.shape[0], 1])
    hurst = np.zeros([dist.shape[0], 1])
    for n in range(dist.shape[0]):
        kurt_dist[n] = stats.kurtosis(dist[n,:])
        hurst[n] = hurst_mandelbrot(dist[n,:])
    meankurt_dist = np.mean(kurt_dist)
    meanhurst_dist = np.mean(hurst)

    cs = np.hstack([mean_dist.reshape(N,1), median_dist.reshape(N,1), std_dist.reshape(N,1), kurt_dist, hurst])

    s = np.array([mean_dist, std_dist, meankurt_dist, meanhurst_dist])

    return s, cs

def define_clusters(disp_cs, turn_cs, nb_pc=10, nb_clusters=5):
    '''
    Defines clusters based on displacement and turn features
    using PCA decomposition and k-means clustering.

    Parameters
    ----------
    disp_cs : ndarray, N x 5.
        statistics of cell displacements.
    turn_cs : ndarray, N x 5.
        statistics of cell turns.
    nb_pc : integer.
        number of principal components to use for clustering.
    nb_clusters : integer.
        number of clusters to fit.

    Returns
    -------
    labels : ndarray, N x 1.
        vector of integer class labels (1 indexed).
    '''

    feat = np.hstack([disp_cs, turn_cs])
    pca = PCA(n_components=nb_pc)
    feat_reduced = pca.fit_transform(feat)

    estimator = cluster.KMeans(n_clusters=nb_clusters)
    estimator.fit(feat_reduced)

    return (estimator.labels_ + 1)

def fit_johnson_per_label(X, labels):
    '''
    Fits a Johnson Distribution (bounded) to a univariate variable X
    for each class label in labels

    Parameters
    ----------
    X : ndarray, N x T, univariate numeric.
    labels : ndarray, integer class labels.

    Returns
    -------
    distributions : list of rv_continous objects.
        List of fitted Johnson distributions, indexed by label.
    '''

    distributions = []
    for l in np.unique(labels):
        X_l = X[labels == l]
        X_lr = X_l.reshape((X_l.shape[0]*X_l.shape[1]),1)
        a, b, loc, scale = stats.johnsonsb.fit(X_lr)
        d = stats.johnsonsb(a, b, loc=loc, scale=scale)
        distributions.append(d)
    return distributions


def walker(n, displacement_dist, turn_dist, T = 100, origin = (0,0), bound_x=(-np.Inf, np.Inf), bound_y=(-np.Inf, np.Inf)):
    '''
    Simulates a motion path exhibiting walking behavior by sampling
    from a provided displacement distribution and turn angle distribution

    Parameters
    ----------
    n : any number, used to make compatible with multiprocess pool.
        Unused arg.
    displacement_dist : scipy.stats.rv_continous object.
        Distribution of displacement sizes [px] for random sampling.
    turn_dist : scipy.stats.rv_continous object.
        Distribution of turn angles [radians] for random sampling.
    T : integer.
        length in time units of the path to be simulated.
    origin : tuple, origin (X, Y) coordinate.
    bound_x : tuple, optional.
        (min, max) allowed values for X coordinates.
        Useful to confine model to a specified box.
    bound_y : tuple, optional.
        (min, max) allowed values for Y coordinates.
        Useful to confine model to a specified box.

    Returns
    -------
    X : ndarray, 1 x T of X coordinates.
    Y : ndarray, 1 x T of Y coordinates.
    '''

    X = np.zeros(T)
    Y = np.zeros(T)
    X[0] = origin[0] # set initial position per origin parameter
    Y[0] = origin[1]

    t = 1
    while t < T:
        step = displacement_dist.rvs(size=1)
        if t == 1:
            direction = np.random.random(1) * 2 * np.pi
        turn = turn_dist.rvs(size=1)
        # Update direction based on the turn sampled
        # Modulo 2pi unwraps the circle
        direction = (direction + turn) % (2 * np.pi)
        dx = np.cos(direction)*step
        dy = np.sin(direction)*step

        # check bounding conditions
        x_min = bound_x[0] < (X[t-1] + dx)
        x_max = bound_x[1] > (X[t-1] + dx)
        y_min = bound_y[0] < (Y[t-1] + dy)
        y_max = bound_y[1] > (Y[t-1] + dy)
        bound_check = np.array([x_min, x_max, y_min, y_max])

        if np.all(bound_check):
            X[t] = X[t-1] + dx
            Y[t] = Y[t-1] + dy
            t += 1
        else:
            pass

    return X, Y

def simulate_from_distributions(disp_distributions,
                                turn_distributions,
                                obj = 5000,
                                T=100,
                                bound_x=(-np.Inf, np.Inf),
                                bound_y=(-np.Inf, np.Inf),
                                labels = None):
    '''
    Generates simulated paths from a set of displacement and turn angle
    distributions

    Parameters
    ----------
    disp_distributions : scipy.stats.rv_continous object.
        Distribution model of instantaneous XY displacement magnitudes.
    turn_distributions : scipy.stats.rv_continous object.
        Distribution model of instantaneous turn angles in [radians].
    obj : integer.
        Number of objects to simulate.
        if labels = None, number of objects per class.
        if proportional != None, number of total objects.
    T : integer.
        Length of simulated walks.
    bound_x : tuple, optional.
        (min, max) allowed values for X coordinates.
        Useful to confine model to a specified box.
    bound_y : tuple, optional.
        (min, max) allowed values for Y coordinates.
        Useful to confine model to a specified box.
    labels : ndarray, N x 1 of integer class labels.

    Returns
    -------
    setX : list of ndarrays. Indexed by label.
        Each element in the list is an N x T ndarray of X coordinates.
    setY : list of ndarrays. Indexed by label.
        Each element in the list is an N x T ndarray of X coordinates.
    '''

    setX = []
    setY = []

    if np.any(labels):
        # generate vector with proportional numbers of each simulated class
        nb_class = np.zeros(np.max(labels).astype('int32'))
        for i in range(np.max(labels).astype('int32')):
            nb_class[i] = np.sum( labels == (i+1) )
        props = nb_class / np.sum(nb_class)

        nb_sim = np.round(props * obj).astype('int32')

    else:
        # otherwise, generate the vector with the same number for each class
        nb_sim = np.ones(len(disp_distributions))
        nb_sum *= obj
        nb_sum = nb_sum.astype('int32')

    for l in range(len(disp_distributions)):
        X = np.zeros([nb_sim[l], T])
        Y = np.zeros([nb_sim[l], T])

        # Set up partial function with preloaded args
        part_walker = partial(walker, displacement_dist=disp_distributions[l],
                                turn_dist=turn_distributions[l], T=T,
                                bound_x = bound_x, bound_y = bound_y)

        # for n in range(nb_sim[l]):
        #    X[n,:], Y[n,:] = walker(disp_distributions[l], turn_distributions[l], T=T, bound_x = bound_x, bound_y = bound_y)

        # process with multiprocess.Pool
        # feed range values to dummy variable n
        p = Pool()
        paths = list(p.map(part_walker, range(nb_sim[l])))

        for i in range(len(paths)):
            X[i,:] = paths[i][0]
            Y[i,:] = paths[i][1]

        setX.append(X)
        setY.append(Y)

    return setX, setY

def sample_data_to_sims(X,
                        Y,
                        labels,
                        obj=5000,
                        T=100,
                        bound_x=(-np.Inf, np.Inf),
                        bound_y=(-np.Inf, np.Inf)):
    '''
    Fit Johnson distributions to sampled displacement and turn angle values
    for each class specified in labels

    Parameters
    ----------
    X, Y : ndarray, N x T of X, Y positions.
    labels : ndarray, N x 1.
        vector of integer class labels.
    obj : integer.
        Number of objects per class to simulate.
    T : integer.
        Length of simulated walks.
    bound_x : tuple, optional.
        (min, max) allowed values for X coordinates.
        Useful to confine model to a specified box.
    bound_y : tuple, optional.
        (min, max) allowed values for Y coordinates.
        Useful to confine model to a specified box.

    Returns
    -------
    setX : list of ndarrays.
        List of N x T ndarrays of X coordinates.
    setY : list of ndarrays.
        List of N x T ndarrays of Y coordinates.
    '''

    displacement_mat = displacement_matrix(X, Y, tau=1)
    turn_mat = turn_matrix(X, Y, tau=1)

    disp_distributions = fit_johnson_per_label(displacement_mat, labels)
    turn_distributions = fit_johnson_per_label(turn_mat, labels)

    setX, setY = simulate_from_distributions(disp_distributions,
                                            turn_distributions,
                                            obj = 100000,
                                            T = T,
                                            bound_x = bound_x,
                                            bound_y = bound_y,
                                            labels = labels)

    return setX, setY

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('input_X', help='path to input CSV of X coordinates')
    parser.add_argument('input_Y', help='path to input CSV of Y coordinates')
    parser.add_argument('output_dir', default = './', help='output directory for simulated paths')
    parser.add_argument('--obj', default=100000, help='number of objects to simulate')
    parser.add_argument('--T', default=100, help='length of paths to simulate')
    parser.add_argument('--label_matrix', default=False, help='path to CSV with class labels in far right column')


    args = parser.parse_args()
    input_X = args.input_X
    input_Y = args.input_Y
    label_matrix = args.label_matrix
    output_dir = args.output_dir
    obj = args.obj
    T = args.T

    print('Loading data...')
    X, Y = load_real_data(input_X, input_Y)
    if label_matrix:
        labels = load_class_labels(label_matrix)
    else:
        disp_mat = displacement_matrix(X, Y)
        turn_mat = turn_matrix(X, Y)
        disp_s, disp_cs = dist_statistics(disp_mat)
        turn_s, turn_cs = dist_statistics(turn_mat)
        labels = define_clusters(disp_cs, turn_cs)

    print('Simulating paths...')
    setX, setY = sample_data_to_sims(X, Y, labels=labels, obj=obj, T=T, bound_x=(-1024,1024), bound_y=(-1024,1024))

    print('Saving paths to ', output_dir, '...')
    for i in range(len(setX)):
        np.savetxt(os.path.join(output_dir, 'mimesisX_class' + str(i+1) + '.csv'), setX[i])
        np.savetxt(os.path.join(output_dir, 'mimesisY_class' + str(i+1) + '.csv'), setY[i])

    return

if __name__ == '__main__':
    main()
