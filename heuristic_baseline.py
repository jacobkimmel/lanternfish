'''
Heuristic Classifier Baseline

Implements a heuristic feature extractor paired to a simple SVM as a baseline
classification model.
'''
import numpy as np
from sklearn.preprocessing import scale

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
            (mean_displacement, var_displacement, total_distance, net_distance)
        '''
        self.X = X

        self._displacements()
        self._mean_displacement()
        self._var_displacement()
        self._total_distance()
        self._net_distance()

        features = np.stack([self.mean_disp, self.var_disp, self.total_distance, self.net_distance], -1)
        return features
