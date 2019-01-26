'''
Baseline classification experiments
'''
import itertools
import numpy as np
import pandas as pd
import pickle
import os
import os.path as osp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from heuristic_baseline import HeuristicMotionClassifier, MotionFeatureExtractor

out_dir = 'classif_baselines'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

n_estimators = [5, 25, 100, 200]
max_features = ["auto", "log2", 0.5, 0.75, 0.9]
min_samples_split = [2, 5, 10]
rf_parameters = list(
    itertools.product(n_estimators, max_features, min_samples_split))


def train_rf_models(rf_parameters: list,
                X: np.ndarray,
                y: np.ndarray) -> list:
    task_scores = []
    for i, params in enumerate(rf_parameters):
        clf = RandomForestClassifier(n_estimators=params[0],
                                    max_features=params[1],
                                    min_samples_split=params[2],
                                    n_jobs=-1)
        scores = cross_val_score(clf, X, y, cv=5)
        task_scores.append(scores)
    best_acc = 0.
    for i in range(len(task_scores)):
        acc = np.mean(task_scores[i])
        if acc > best_acc:
            best_acc = acc
    print('Best accuracy: %f' % best_acc)
    return task_scores

musc_x = np.loadtxt('../data/musc/all_tracksX.csv', delimiter=',')
musc_y = np.loadtxt('../data/musc/all_tracksY.csv', delimiter=',')

myo_x = np.loadtxt('../data/myoblast/all_tracksX.csv', delimiter=',')
myo_y = np.loadtxt('../data/myoblast/all_tracksY.csv', delimiter=',')

mef_wt_x = np.loadtxt('../data/wt_mef/all_tracksX.csv', delimiter=',')
mef_wt_y = np.loadtxt('../data/wt_mef/all_tracksY.csv', delimiter=',')

mef_mr_x = np.loadtxt('../data/myc_mef/all_tracksX.csv', delimiter=',')
mef_mr_y = np.loadtxt('../data/myc_mef/all_tracksY.csv', delimiter=',')


data_sets = {}

# MuSC / MEF classif
print('Loading MuSC v MEF data...')
n_train = 405
n_test = 50
n_val = 50
n_per_class = n_train + n_test + n_val
min_t = np.min([musc_x.shape[1], mef_wt_x.shape[1], mef_mr_x.shape[1]])

ridx_musc = np.random.choice(range(musc_x.shape[0]),
                            size=n_per_class,
                            replace=False)
X_musc = np.stack([musc_x, musc_y], -1)[ridx_musc,:min_t,:]

X_mef_wt = np.stack([mef_wt_x, mef_wt_y], -1)[:,:min_t,:]
X_mef_mr = np.stack([mef_mr_x, mef_mr_y], -1)[:,:min_t,:]
X_mef = np.concatenate([X_mef_wt, X_mef_mr], axis=0)
ridx_mef = np.random.choice(range(X_mef.shape[0]),
                            size=n_per_class,
                            replace=False)
X_mef = X_mef[ridx_mef, :, :]
X = np.concatenate([X_musc, X_mef], axis=0)
y = np.array([0]*X_musc.shape[0] + [1]*X_mef.shape[0])

data_sets['musc_mef'] = (X, y)

# MuSC / Myo classif
print('Loading MuSC v Myoblast data...')
n_train = 200
n_test = 50
n_val = 27
n_per_class = n_train + n_test + n_val
min_t = np.min([musc_x.shape[1], myo_x.shape[1]])

ridx_musc = np.random.choice(range(musc_x.shape[0]), size=n_per_class, replace=False)
ridx_myo = np.random.choice(range(myo_x.shape[0]), size=n_per_class, replace=False)

X_musc = np.stack([musc_x, musc_y], axis=-1)
X_myo = np.stack([myo_x, myo_y], axis=-1)
X_musc = X_musc[ridx_musc, :min_t, :]
X_myo = X_myo[ridx_myo, :min_t, :]

X = np.concatenate([X_musc, X_myo], axis=0)
y = np.array([0]*X_musc.shape[0] + [1]*X_myo.shape[0])

data_sets['musc_myo'] = (X, y)

# MEF Type classif
print('Loading MEF type data...')
n_train = 160
n_test = 30
n_val = 30
n_per_class = n_train + n_test + n_val
min_t = np.min([ mef_wt_x.shape[1], mef_mr_x.shape[1] ])

ridx_wt = np.random.choice(range(mef_wt_x.shape[0]), size=n_per_class, replace=False)
X_mef_wt = np.stack([mef_wt_x, mef_wt_y], -1)[ridx_wt,:min_t,:]

ridx_mr = np.random.choice(range(mef_mr_x.shape[0]), size=n_per_class, replace=False)
X_mef_mr = np.stack([mef_mr_x, mef_mr_y], -1)[ridx_mr,:min_t,:]

X = np.concatenate([X_mef_wt, X_mef_mr], axis=0)
y = np.array([0]*X_mef_wt.shape[0] + [1]*X_mef_mr.shape[0])

data_sets['mef_type'] = (X, y)

# Simulation Classification
sim_data_dir = '../data/sim_data/'
print('Loading simulated data...')
rw_x = np.loadtxt(os.path.join(sim_data_dir, 'rw_X_100k.csv'), delimiter=',')
rw_y = np.loadtxt(os.path.join(sim_data_dir, 'rw_Y_100k.csv'), delimiter=',')
pf_x = np.loadtxt(os.path.join(sim_data_dir, 'pf_X_100k_mu5.csv'), delimiter=',')
pf_y = np.loadtxt(os.path.join(sim_data_dir, 'pf_Y_100k_mu5.csv'), delimiter=',')
fbm_x = np.loadtxt(os.path.join(sim_data_dir, 'fbm_X_100k_mu5.csv'), delimiter=',')
fbm_y = np.loadtxt(os.path.join(sim_data_dir, 'fbm_Y_100k_mu5.csv'), delimiter=',')
print('Data loaded.')

n_use = 15000 # use the same number of examples as 3D CNNs
xs = np.concatenate([rw_x[:n_use, :], pf_x[:n_use, :], fbm_x[:n_use, :]], axis=0)
ys = np.concatenate([rw_y[:n_use, :], pf_y[:n_use, :], fbm_y[:n_use, :]], axis=0)
X = np.stack([xs, ys], axis=-1)
y = np.array([0]*n_use + [1]*n_use + [2]*n_use)

data_sets['simulations'] = (X, y)

mfe = MotionFeatureExtractor()
# Train all RF Models
all_scores = {}
for ds in data_sets:
    print('Training %s RF models' % ds)
    X, y = data_sets[ds]
    features = mfe(X)
    scores = train_rf_models(rf_parameters, features, y)
    scores_array = np.array(scores)
    np.savetxt(osp.join(out_dir, ds + '_score_array.csv'),
        scores_array, delimiter=',')
    all_scores[ds] = scores_array

print('Saving all scores...')
with open(osp.join(out_dir, 'all_scores.pkl'), 'wb') as f:
    pickle.dump(all_scores, f)

print('Finding best scores...')
# Determine the best hyperparameter settings
best_scores = np.zeros((len(all_scores), 5))
tasks = []
for i, ds in enumerate(all_scores):
    tasks.append(ds)
    best_params = np.argmax(np.mean(all_scores[ds], axis=1))
    best_scores[i, :] = all_scores[ds][best_params, :]

best_scores = pd.DataFrame(best_scores)
best_scores.index = tasks
best_scores.columns = ['CV' + str(i).zfill(2) for i in range(5)]
best_scores.to_csv(osp.join(out_dir, '00_best_scores.csv'))
print('Scores saved.')
