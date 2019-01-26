'''
Linear motion predictions

Performs motion predictions based on linear kinematics
'''

import numpy as np

class LinearMotionModel(object):

    def __init__(self):
        '''
        Linear motion prediction model
        '''

    def predict(self, X, t_out=10, tau=1):
        '''
        Predicts future positions from a set of sample coordinates
        using linear kinematics

        Parameters
        ----------
        X : ndarray.
            (N, T, 2) array of `x` and `y` coordiantes,
            with `x` in [:,:,0] and `y` in [:,:,1]
        t_out : integer.
            time steps in the future to predict.
        tau : integer.
            time interval for estimating cell velocity.
        '''

        dv = (X[:,-1,:] - X[:,-(tau+1),:])/tau # <velocity> per cell over `tau`
        dv = np.expand_dims(dv, 1)

        # create replicated arrays of `dv` and a temporal sequence of the
        # shape (n_cells, t_out, 2)
        dv_mat = np.tile(dv, (1, t_out, 1))
        t_mat = np.tile(np.arange(1,t_out+1).reshape(1,t_out,1), (X.shape[0],1,2))

        # add displacements multiplied by the time step to an array of
        # of the final given position
        last_mat = np.tile( np.expand_dims(X[:,-1,:], 1), (1, t_out, 1))
        y = last_mat + (dv_mat * t_mat)
        return y
