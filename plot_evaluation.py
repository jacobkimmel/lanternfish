'''
Plots evaluation metrics of a Keras model
'''
from keras.models import load_model
from motcube_preprocessing import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

def predict_val(val_dir, model, batch_size = 12, target_size=(156,156,101), print_eval=False):
    '''
    Predicts classes from a validation set using a Keras model

    Parameters
    ----------
    val_dir : string.
        path to directory of validation samples, stored in class specific
        dir's per Keras standard practice.
    model : Keras model object.
        model used to generate predictions.
    batch_size : integer.
        size of batches for prediction.
    target_size : tuple of integers.
        tuple specifying the target sample size for prediction.
    print_eval : boolean, optional.
        perform and print evaluation.

    Returns
    -------
    ytrue : ndarray.
        1D ndarray of ground truth class labels.
    ypred : ndarray.
        1D ndarray of predicted class labels.
    classes : dict.
        dictionary keyed by class names, valued with integer representations.
    '''

    valgen = MotcubeDataGenerator()
    val_generator = valgen.flow_from_directory(val_dir, class_mode='categorical', color_mode='grayscale', target_size = target_size, batch_size = batch_size, shuffle=False)

    ytrue = val_generator.classes
    classes = val_generator.class_indices

    ypred_softmax = model.predict_generator(val_generator, val_samples=val_generator.nb_sample)

    ypred = np.argmax(ypred_softmax, axis = 1)

    if print_eval:
        print("Accuracy : ", np.sum(ypred == ytrue)/ypred.shape)

    return ytrue, ypred, classes

def plot_confusion(ytrue, ypred, classes,
                   title='Confusion matrix', cmap=plt.cm.Blues,
                   normalize=False, exp_name='model', save=False):
    '''
    Plots a confusion matrix from a set of true and predicted classes

    Parameters
    ----------
    ytrue : ndarray.
        1D ndarray of ground truth class labels.
    ypred : ndarray.
        1D ndarray of predicted class labels.
    classes : list.
        list of class names.
    title : string, optional.
        plot title.
    cmap : matplotlib color map object, optional.
    normalize : boolean, optional.
        normalizes confusion matrix to show proportions of class predictions.
    exp_name : string.
        exp_name for plot subtitle and file name save.
    save : string, optional.
        path to a directory for plot output.

    Returns
    -------
    None.
    '''
    cm = confusion_matrix(ytrue, ypred)
    np.set_printoptions(precision = 3)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.figure(figsize=(7,5))
    df = pd.DataFrame(cm, index = [i for i in classes], columns = [i for i in classes])
    sns.heatmap(df, annot=True, cmap=cmap, square=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)

    if save:
        plt.savefig(os.path.join(save, exp_name + '_conf_mat.png'))

    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('val_dir', help = 'path to csv')
    parser.add_argument('model_path', help = 'dir for saved figs')
    parser.add_argument('exp_name', help = 'name for the experiment')
    parser.add_argument('save_dir', help = 'path to save plots')
    parser.add_argument('--batch_size', default = 12, help = 'batch size for model predictions')
    parser.add_argument('--target_size', nargs = '+', type=int, default = [156,156,101])
    args = parser.parse_args()
    target_size = tuple(args.target_size)
    print(args.val_dir, args.model_path, args.exp_name, args.save_dir, args.batch_size)
    print(target_size)

    print('Loading model...')
    model = load_model(args.model_path)
    print('Predicting classes...')
    ytrue, ypred, classes = predict_val(args.val_dir, model, args.batch_size, target_size=target_size)
    print('Plotting confusion...')
    plot_confusion(ytrue, ypred, classes, title = 'Confusion Matrix', cmap=plt.cm.PiYG, normalize=True, exp_name=args.exp_name, save=args.save_dir)

    return

if __name__ == '__main__':
    main()
