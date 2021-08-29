import os
import sys
import time

#from collections import Counter
import argparse
import numpy as np
import loaddata
import utils

from sklearn.ensemble import IsolationForest
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.pca import PCA


def run_benchmark(
        model,
        training_normal_data,
        testing_normal_data,
        testing_abnormal_data,
        window_size,
        n_samples_train = None,
        n_samples_eval = None,
        percentile_th_on_validation = None,
        validation_normal_data = None,
        verbose = True
    ):

    # Normalize training data
    training_normal_data_mean = utils.get_mean(training_normal_data)
    training_normal_data_std = utils.get_std(training_normal_data)

    training_normal_data = utils.normalize(
        training_normal_data, training_normal_data_mean, training_normal_data_std
    )

    testing_normal_data = utils.normalize(
        testing_normal_data, training_normal_data_mean, training_normal_data_std
    )
    testing_abnormal_data = utils.normalize(
        testing_abnormal_data, training_normal_data_mean, training_normal_data_std
    )

    if validation_normal_data is not None:
        validation_normal_data = utils.normalize(
            validation_normal_data, training_normal_data_mean, training_normal_data_std
        )

    if verbose:
        print("training_normal_data.shape", training_normal_data.shape)
        print("testing_normal_data.shape", testing_normal_data.shape)
        print("testing_abnormal_data.shape", testing_abnormal_data.shape)


    training_normal_data = utils.seq_win_vectorize(
        seq = training_normal_data,
        window_size = window_size,
        n_samples = n_samples_train,
    )
    testing_normal_data = utils.seq_win_vectorize(
        seq = testing_normal_data,
        window_size = window_size,
        n_samples = n_samples_eval,
    )
    testing_abnormal_data = utils.seq_win_vectorize(
        seq = testing_abnormal_data,
        window_size = window_size,
        n_samples = n_samples_eval,
    )

    if validation_normal_data is not None:
        validation_normal_data = utils.seq_win_vectorize(
            seq = validation_normal_data,
            window_size = window_size,
            n_samples = n_samples_eval,
        )

    if verbose:
        print("Vectorized training_normal_data.shape", training_normal_data.shape)
        print("Vectorized testing_normal_data.shape", testing_normal_data.shape)
        print("Vectorized testing_abnormal_data.shape", testing_abnormal_data.shape)
        if validation_normal_data is not None:
            print("Vectorized validation_normal_data.shape", validation_normal_data.shape)

    # +1 is normal, -1 is abnormal
    true_label_normal = np.zeros(len(testing_normal_data))
    true_label_abnormal = np.ones(len(testing_abnormal_data))
    true_label = np.concatenate(
        (
            true_label_normal,
            true_label_abnormal
        ),
        axis=0
    )

    training_data_run = training_normal_data
    testing_data_run = np.concatenate(
        (
            testing_normal_data,
            testing_abnormal_data
        ),
        axis=0
    )
    if validation_normal_data is not None:
        validation_normal_data_run = validation_normal_data

    assert len(testing_data_run) == len(true_label)

    if model == 'IF':
        cls = IsolationForest(n_estimators=1000, contamination = 0.1)
        pred_score_is_anomaly_score = False
        normal_label = 1
        abnormal_label = -1

    elif model == 'OCSVM':
        cls = OCSVM(kernel='rbf', nu=0.1, contamination=0.1)
        pred_score_is_anomaly_score = True
        normal_label = 0
        abnormal_label = 1

    elif model == 'LOF':
        cls = LOF(n_neighbors=500, algorithm='brute', contamination=1e-4)
        pred_score_is_anomaly_score = True
        normal_label = 0
        abnormal_label = 1

    elif model == 'PCA':
        cls = PCA(contamination=0.1)
        pred_score_is_anomaly_score = True
        normal_label = 0
        abnormal_label = 1
    else:
        print("Model not support")
        exit(1)

    time_start = time.time()
    cls.fit(training_data_run)

    time_train_finish = time.time()
    print("Training takes {time} seconds".format(
        time=time_train_finish-time_start
        ))

    # Models may follow different convention, i.e. 0/1 and -1/+1
    # Convert them to 0: normal and 1: abnormal, respectively
    pred_raw = cls.predict(testing_data_run)
    pred = np.copy(pred_raw)
    pred[pred_raw == normal_label] = 0
    pred[pred_raw == abnormal_label] = 1

    time_eval_finish = time.time()
    print("Evaluation takes {time} seconds".format(
        time=time_eval_finish-time_train_finish
        ))

    pred_score = cls.decision_function(testing_data_run)
    pred_score_train = cls.decision_function(training_data_run)
    if validation_normal_data is not None:
        pred_score_val = cls.decision_function(validation_normal_data_run)

    if verbose:
        print ("Raw unique pred labels", np.unique(pred_raw))
        print ("Raw pred labels", pred_raw)
        print ("pred_score", pred_score)

    if not pred_score_is_anomaly_score:
        anomaly_score = -pred_score
        anomaly_score_train = -pred_score_train
        if validation_normal_data is not None:
            anomaly_score_val = -pred_score_val
    else:
        anomaly_score = pred_score
        anomaly_score_train = pred_score_train
        if validation_normal_data is not None:
            anomaly_score_val = pred_score_val

    if percentile_th_on_validation is not None:
        # Use percentile threshold on training data
        preset_th = np.percentile(anomaly_score_val, percentile_th_on_validation)
        print(f"preset_th {preset_th}")
    else:
        # Use EER threshold on testing data
        preset_th = None

    # Pay special attention here the score is the anomaly score
    tp, fp, fn, tn, acc, prec, rec, f1, fpr, tpr, thresholds, roc_auc = \
    utils.eval_metrics(
        truth = true_label,
        pred = pred,
        anomaly_score = anomaly_score,
        preset_th = preset_th,
        verbose = verbose
    )

    return fpr, tpr, thresholds, roc_auc, anomaly_score, true_label

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type = str, default = "all", help='Anomaly detection models')

    # Loaddata
    # Sequential data in the form of (Timeframe, Features)
    # Training only leverages normal data. Abnormaly data only for testing.
    parser.add_argument('--data_dir', type = str, default = "../perf/data/core0/100us/", help='The directory of data')
    parser.add_argument('--normal_data_name_train', type = str, default = "train_normal.npy", help='The file name of training normal data')
    parser.add_argument('--normal_data_name_test', type = str, default = "test_normal.npy", help='The file name of testing normal data')
    parser.add_argument('--abnormal_data_name', type = str, default = "test_abnormal.npy", help='The file name of testing abnormal data')

    # Window size
    parser.add_argument('--window_size', type = int, default = 10, help='Window size for vectorization')

    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Whether debug information will be printed')
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    #feature_list = [0,2,6,8,14,18,20,21,22]
    feature_list = [0,2,6,8,14,18,20,21,22,31]
    train_normal = np.load(args.data_dir + args.normal_data_name_train)[:, feature_list]
    test_normal = np.load(args.data_dir + args.normal_data_name_test)[:, feature_list]
    test_abnormal = np.load(args.data_dir + args.abnormal_data_name)[:, feature_list]

    print("feature_list", feature_list)
    print("train_normal.shape", train_normal.shape)
    print("test_normal.shape", test_normal.shape)
    print("test_abnormal.shape", test_abnormal.shape)

    model_options = {
        'all': ['LOF', 'OCSVM', 'IF', 'PCA'],
        'LOF': ['LOF'],
        'OCSVM': ['OCSVM'],
        'IF': ['IF'],
        'PCA': ['PCA'],
     }

    for model in model_options[args.model]:
        print("Model: ", model)
        fpr, tpr, thresholds, roc_auc = run_benchmark(
            model = model,
            training_normal_data=train_normal,
            testing_normal_data=test_normal,
            testing_abnormal_data=test_abnormal,
            window_size=args.window_size,
            n_samples_train=20000,   # Randomly sample 20,000 samples for training
            n_samples_eval=10000,
            verbose = args.verbose
        )

    save_roc_dir = 'temp/'
    os.system('mkdir -p {dir}'.format(dir=save_roc_dir))
    np.save(save_roc_dir + model + '_fpr', fpr)
    np.save(save_roc_dir + model + '_tpr', tpr)
