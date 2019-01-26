'''Lanternfish CLI'''
import numpy as np
import keras
from .bestiary import xy_ts_rnn_model as clf_rnn_cnn
from .bestiary import rnn_baseline_model as clf_rnn_base
from .bestiary import xy_ts_rnn_ae as ae_rnn
from .bestiary import rnn_motion_pred as rnn_prediction
import argparse
import datetime

def train_rnn_clf_cv(
                   model,
                   X: np.ndarray,
                   y: np.ndarray,
                   out_dir: str,
                   exp_name: str=None,
                   n_splits: int=5,
                   n_classes: int=2,
                   batch_size: int=32,
                   w0=None,
                   ae_w=None,
                   reg=keras.regularizers.l2(1e-5),
                   use_generator: bool=False,
                   patience: int=15) -> None:

    if exp_name is None:
        exp_name = os.path.basename(out_dir)
    print('Training experiment:')
    print(exp_name)

    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    kf = KFold(n_splits=n_splits, shuffle=True)
    kf_idx = list(kf.split(X))
    print('%d k-fold splits.' % len(kf_idx))

    runs = n_splits
    for i in range(runs):

        traintest_idx = kf_idx[i][0]
        val_idx = kf_idx[i][1]

        train_idx = traintest_idx[:int(0.9*len(traintest_idx))]
        test_idx  = traintest_idx[int(0.9*len(traintest_idx)):]

        file_name_save = os.path.join(out_dir,
            exp_name + '_run_' + str(i).zfill(2) + '.h5')

        callbacks = [CSVLogger(file_name_save[:-3] + '_train_log.csv'),
        ModelCheckpoint(file_name_save, monitor='val_loss', save_best_only=True),
        EarlyStopping(patience=patience, monitor='val_loss')]

        batch_size = batch_size
        opt = keras.optimizers.Adam(lr=1e-3,)
        model.compile(metrics=['accuracy'],
            loss='categorical_crossentropy',
            optimizer=opt)

        # Transfer weights
        if w0 is not None:
            w1 = model.get_weights()
            w = w0
            w[-2:] = w1[-2:] # final Dense layers are different sizes
            model.set_weights(w)
            print('Weights transferred.')
        if ae_w is not None:
            w1 = model.get_weights()
            w = w1
            w[:4] = ae_w[:4] # only first 4 layers are the same
            model.set_weights(w)

        if use_generator:
            transform = TrackAugment(T=T)
            train_gen = TrackGen(X[train_idx, :, :], y[train_idx, :],
                                transform=transform)
            test_gen  = TrackGen(X[test_idx, :T, :], y[test_idx, :],
                                transform=None)
            model.fit_generator(
                generator=train_gen,
                validation_data=test_gen,
                epochs=1000,
                callbacks=callbacks,
            )
        else:
            model.fit(
                X[train_idx, ...], y[train_idx],
                batch_size=batch_size,
                shuffle=True,
                epochs=1000,
                validation_data=(X[test_idx, ...], y[test_idx]),
                callbacks=callbacks)

        # reload model with best weights
        model = load_model(file_name_save)
        evaluate = model.evaluate(X[val_idx, :T, :], y[val_idx])

        print('Finished experiment:')
        print(exp_name)
        print('Val Loss : %f | Val Accuracy : %f' % (evaluate[0], evaluate[1]))
        np.savetxt(file_name_save[:-3]+'_evaluation.csv', evaluate, delimiter=',')
    return

def train_rnn_ae(model, X, out_path, exp_name):

    traintest_ridx = np.random.choice(np.arange(X.shape[0]),
                    size=int(0.9*X.shape[0]),
                    replace=False).astype(np.int32)
    val_ridx = np.setdiff1d(np.arange(X.shape[0]), traintest_ridx)

    np.savetxt(os.path.join(save_dir, 'traintest_idx_run_' + str(i).zfill(2) + '.csv'),
        traintest_ridx)
    np.savetxt(os.path.join(save_dir, 'val_idx_run_' + str(i).zfill(2) + '.csv'),
        val_ridx)

    file_name_save = os.path.join(save_dir,
        exp_name + '_rnn_ae' + '.h5')

    callbacks = [CSVLogger(file_name_save[:-3] + '_train_log.csv'),
    ModelCheckpoint(file_name_save, monitor='val_loss', save_best_only=True),
    EarlyStopping(patience=20, monitor='val_loss')]

    batch_size = 64
    print(model.summary())
    opt = keras.optimizers.Adadelta(lr=0.1)
    model.compile(loss='mse',
                    optimizer=opt)
    model.fit(
        X[traintest_ridx, :, :][:int(0.9*len(traintest_ridx)),...],
        X[traintest_ridx, :, :][:int(0.9*len(traintest_ridx)),...],
         batch_size=batch_size,
         shuffle=True,
         epochs=1000,
         validation_data=(X[traintest_ridx,:][int(0.9*len(traintest_ridx)):,...],
                        X[traintest_ridx,:][int(0.9*len(traintest_ridx)):,...]),
         callbacks=callbacks)

    # reload best Weights
    model = load_model(file_name_save)
    evaluate = model.evaluate(X[val_ridx,:, :], X[val_ridx,:, :])
    pred = model.predict(X[val_ridx, :, :])

    np.savetxt(os.path.join(save_dir,
            'track_predictionsX.csv'),
        pred[:,:,0], delimiter=',')
    np.savetxt(os.path.join(save_dir,
            'track_predictionsY.csv'),
        pred[:,:,1], delimiter=',')
    print('Val Loss : %f ' % evaluate)
    np.savetxt(file_name_save[:-3]+'_evaluation.csv', np.array(evaluate).reshape(1,1), delimiter=',')

    print('Computing AE latents...')
    bottleneck = keras.models.Model(inputs=model.inputs,
                            outputs=model.layers[7].output)
    features = bottleneck.predict(X)
    np.savetxt(os.path.join(save_dir, 'ae_features.csv'),
        features,
        delimiter=',')
    print('Saved.')

def main():

    date = datetime.datetime.today().strftime('%Y%m%d')

    parser = argparse.ArgumentParser(
        description='Lanternfish tools for cell motility classification, \
            latent space learning, and motility prediction')
    # Add training options
    parser.add_argument('task', type=str, default='train',
        help='task to perform. {"train","predict"}')
    parser.add_argument('tracksX',
        type=str,
        default=None,
        help='path to [N, T] CSV of track X coordinates.')
    parser.add_argument('tracksY',
        type=str,
        default=None,
        help='path to [N, T] CSV of track Y coordinates.')
    parser.add_argument('--labels', type=str, default=None,
        help='path to class labels CSV. \
            required for training classification models.')
    parser.add_argument('--model_type',
        type=str,
        default='classification_rnn_cnn',
        help='type of model to train')
    parser.add_argument('--out_path',
        type=str,
        default='./',
        help='directory for outputs')
    parser.add_argument('--weights', type=str,
        default=None,
        help='path to pretrained weights for prediction')
    args = parser.parse_args()

    if args.task not in ['train', 'predict']:
        raise ValueError('only "train" and "predict" are supported tasks.')
    if args.task == 'predict':
        assert args.weights is not None, \
            'must supply model weights for prediction'

    print('Loading data...')
    tx = np.loadtxt(args.tracksX, delimiter=',')
    ty = np.loadtxt(args.tracksY, delimiter=',')
    X = np.stack([tx, ty], -1)
    print('Data loaded.')
    print('%d samples and %d timepoints.' % X.shape[:2])
    N, T = X.shape[:2]

    if args.model_type.split('_')[0] == 'classification':
        assert args.labels is not None, 'must supply class labels'
        L = np.loadtxt(args.labels, delimiter=',')
        assert L.shape[0] == X.shape[0], 'unequal number of tracks and classes'
        n_classes = len(np.unique(L))
        y = keras.utils.to_categorical(L)

    reg = keras.regularizers.l2(1e-5)
    if args.model_type == 'classification_rnn_cnn':
        model = clf_rnn_cnn(T, n_classes=n_classes, reg=reg)
    elif args.model_type == 'classification_rnn_baseline':
        model = clf_rnn_base(T, n_classes=n_classes, reg=reg)
    elif args.model_type == 'autoencoder_rnn':
        model = ae_rnn(T, reg=reg, min_dim=16)
    else:
        raise ValueError('invalid model_type')

    if args.task == 'train' and args.model_type.split('_')[0] == 'classification':
        train_rnn_clf_cv(model, X, y, out_path=args.out_path, exp_name=date)
    elif args.task == 'train' and args.model_type.split('_')[0] == 'autoencoder':
        train_rnn_ae(model, X, out_path=args.out_path, exp_name=date)
    else:
        raise ValueError()

    if args.task == 'predict' and args.model_type.split('_')[0] == 'classification':
        model = keras.models.load_model(args.weights)
        preds = model.predict(X)
        np.savetxt(osp.join(args.out_dir, 'predictions.csv'),
            preds,
            delimiter=',')
    elif args.task == 'predict' and args.model_type.split('_')[0] == 'autoencoder':
        model = keras.models.load_model(args.weights)
        predictions = model.predict(X)
        for c in range(predictions.shape[2]):
            np.savetxt(osp.join(args.out_dir, 'decoded_c'+str(c).zfill(2)+'.csv'),
                predictions[:,:,c],
                delimiter=',')

        enc = keras.models.Model(model.inputs, model.layers[7].output)
        latent = enc.predict(X)
        np.savetxt(osp.join(args.out_path, 'latent.csv'), delimiter=',')
    else:
        raise ValueError()
