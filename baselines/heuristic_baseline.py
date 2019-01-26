'''
Heuristic Classifier Baseline

Implements a heuristic feature extractor paired to a simple SVM as a baseline
classification model.
'''
import numpy as np
from sklearn.preprocessing import scale
from cell_mimesis import turn_distribution, turn_matrix
from scipy import stats, spatial

class HeuristicMotionClassifier(object):

    def __init__(self, extractor, classif, scale=False):
        '''
        Wrapper for a motion classifier utilizing a callable feature extractor
        `extractor` and classifier `classif`.

        Parameters
        ----------
        extractor : callable.
            feature extractor. takes `X` and return (N, M) feature matrix.
        classif : callable.
            classification model with `.fit` and `.predict` methods.
        scale : boolean.
            scale data before fitting or prediction.
        '''

        self.extractor = extractor
        self.classif = classif
        self.scale = scale

    def fit(self, X, y):
        '''
        Extract features and fit the model to `X` and `y`

        Parameters
        ----------
        X : ndarray.
            N x T x 2, array of motion coordinates where (n,t,0) is `x`,
            (n,t,1) is `y`.
        y : ndarray.
            N x 1 array of integer class labels.
        '''
        self.X_train = X
        self.y_train = y

        self.features_train = self.extractor(X)
        print(self.features_train[:5,:])
        if self.scale:
            self.features_train = scale(self.features_train)
        self.classif.fit(self.features_train, y)

        return

    def predict(self, X):
        '''
        Predict motion classes of objects in `X` using `self.classif`

        Parameters
        ----------
        X : ndarray.
            N x T x 2, array of motion coordinates where (n,t,0) is `x`,
            (n,t,1) is `y`.

        Returns
        -------
        y_pred : ndarray.
            N x 1 array of integer class labels.
        '''
        features_pred = self.extractor(X)
        if self.scale:
            features_pred = scale(features_pred)
        y_pred = self.classif.predict(features_pred)
        return y_pred

    def evaluate(self, X, y):
        '''
        Evaluate classification accuracy on samples in `X` of class `y`

        Parameters
        ----------
        X : ndarray.
            N x T x 2, array of motion coordinates where (n,t,0) is `x`,
            (n,t,1) is `y`.
        y : ndarray.
            N x 1 array of integer class labels.

        Returns
        -------
        evaluation : float.
            evaluation accuracy.
        '''
        features_eval = self.extractor(X)
        if self.scale:
            features_eval = scale(features_eval)
        pred = self.classif.predict(features_eval)
        evaluation = np.sum(pred == y)/y.shape[0]
        return evaluation

class MotionFeatureExtractor(object):

    def __init__(self):
        '''
        Extracts simple heuristic features of motion from a matrix of
        (N, T, 2) motion coordinates.
        '''

    def _displacements(self):
        '''Generates displacment array'''
        self.disp = np.abs(self.X[:,1:,:] - self.X[:,:-1,:])
        self.disp = np.sum(self.disp, 2)
        return

    def _total_distance(self):
        '''Calculates total distance'''
        self.total_distance = np.sum(self.disp, 1)
        return

    def _mean_displacement(self):
        '''Calculates mean displacment'''
        self.mean_disp = np.mean(self.disp, 1)
        return

    def _var_displacement(self):
        self.var_disp = np.std(self.disp, 1)

    def _net_distance(self):
        '''Calculates net distance'''
        self.net_distance = np.sqrt(np.sum((self.X[:,-1,:] - self.X[:,0,:])**2, 1))
        return

    def _progressivity(self):
        self.progressivity = self.net_distance / self.total_distance
        divbyzero = self.total_distance==0
        self.progressivity[divbyzero] = 0.
        return

    def _turn_matrix(self):
        '''Measure turn angles'''
        self.turn_matrix = turn_matrix(self.X[:,:,0], self.X[:,:,1])
        print('Turn Matrix')
        print(self.turn_matrix[:5])
        self.turn_means = np.mean(self.turn_matrix, axis=1)
        self.turn_mag_means = np.mean(np.abs(self.turn_matrix), axis=1)
        self.turn_stds = np.std(self.turn_matrix, axis=1)
        return

    def _linearity(self):
        '''Calculate track linearity'''
        self.linearity = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            x = self.X[i, :, 0]
            y = self.X[i, :, 1]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            if np.isfinite(r_value**2):
                self.linearity[i] = r_value**2
            else:
                self.linearity[i] = 0
        return

    def _spearman(self):
        '''Calculates spearman coefficients'''
        self.spearman = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            rho, _ = stats.spearmanr(self.X[i,:,0], self.X[i,:,1])
            if np.isfinite(rho):
                self.spearman[i] = rho
            else:
                self.spearman[i] = 0.
        return

    def _convex_hull_area(self):
        '''Computes convex hull areas'''
        self.convex_hull_area = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            hull = spatial.ConvexHull(self.X[i, :, :])
            self.convex_hull_area[i] = hull.area
        return

    def __call__(self, X):
        '''
        Extract features from X.

        Parameters
        ----------
        X : ndarray.
            N x T x 2, array of motion coordinates where (n,t,0) is `x`,
            (n,t,1) is `y`.

        Returns
        -------
        features : ndarray.
            N x M array of features.
            (mean_displacement,
            var_displacement,
            min_displacement,
            max_displacement,
            turn_angle_mean,
            turn_mag_mean,
            turn_angle_std,
            total_distance,
            net_distance,
            progressivity,
            linearity,
            spearmanr,
            convex_hull_area)
        '''
        self.X = X

        self._displacements()
        self._mean_displacement()
        self._var_displacement()
        self._total_distance()
        self._net_distance()
        self._progressivity()
        self._linearity()
        self._spearman()
        print('Calculating turns...')
        self._turn_matrix()
        self._convex_hull_area()

        features = np.stack([self.mean_disp,
                             self.var_disp,
                             np.min(self.disp, axis=1),
                             np.max(self.disp, axis=1),
                             self.turn_means,
                             self.turn_mag_means,
                             self.turn_stds,
                             self.total_distance,
                             self.net_distance,
                             self.progressivity,
                             self.linearity,
                             self.spearman,
                             self.convex_hull_area,
                             ], -1)

        if np.sum(np.isnan(features)) > 0:
            print(np.sum(np.isnan(features), axis=0))

        return features
