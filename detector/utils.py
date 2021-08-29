from __future__ import print_function
import collections
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn import metrics

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class FeatureSelect:
    #feature_list = range(31)
    #feature_list = range(34)
    #feature_list = [0,1,2,3,6,8,20,28,30]#14,18]#,20,21,22]
    #feature_list = [0, 1, 27, 13, 3] #, 3, 19, 6, 15, 2, 31]
    #feature_list = [0, 1, 27, 13, 28, 3, 19, 6, 15, 2]
    #feature_list = [0, 1, 27, 13, 28, 3, 19, 6, 15, 2, 31]
    #feature_list = [0, 1, 27, 13, 28, 3, 19, 6, 15, 2, 31]

    feature_list = [0, 32, 33, 27, 1, 13, 3, 19, 15, 28, 2, 6, 31]

def read_npy_data_single_flle(filename):
    print("Reading Data: " + filename)
    data = np.load(filename)
    return data

def write_npy_data_single_file(filename, data):
    print("Writing Data: " + filename)
    np.save(filename, data)
    return

def get_mean(data):
    return np.mean(data, axis = 0, keepdims = True)

def get_std(data):
    return np.std(data, axis = 0, keepdims = True)

def normalize(data, mean, std):

    assert data.shape[1] == mean.shape[1], "Feature size should match mean size"
    assert data.shape[1] == std.shape[1], "Feature size should match std size"

    eps = 1e-8
    return (data - mean) / (std + eps)

def calculate_eval_metrics(truth, pred, verbose=True):
    eps = 1e-12
    tp = np.sum( np.multiply((pred == 1) , (truth == 1)), axis=0 , dtype=np.float32)
    fp = np.sum( np.multiply((pred == 1) , (truth == 0)), axis=0 , dtype=np.float32)
    fn = np.sum( np.multiply((pred == 0) , (truth == 1)), axis=0 , dtype=np.float32)
    tn = np.sum( np.multiply((pred == 0) , (truth == 0)), axis=0 , dtype=np.float32)
    acc = np.sum( pred == truth , axis=0 ) / (1. * truth.shape[0])

    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)
    prec = tp / ( ( tp + fp + eps) * 1. )
    rec =  tp / ( ( tp + fn + eps) * 1. )
    f1 = 2. * prec * rec / ( prec + rec + eps )

    if verbose:
        print('----------------Detection Results------------------')
        print('False positives: ', fp)
        print('False negatives: ', fn)
        print('True positives: ', tp)
        print('True negatives: ', tn)
        print('False Positive Rate: ', fpr)
        print('False Negative Rate: ', fnr)
        print('Accuracy: ', acc)
        print('Precision: ', prec)
        print('Recall: ', rec)
        print('F1: ', f1)
        print('---------------------------------------------------')
    return tp, fp, fn, tn, acc, prec, rec, f1, fpr, fnr

def eval_metrics(truth, pred, anomaly_score=None, preset_th=None, verbose=True):
    tp, fp, fn, tn, acc, prec, rec, f1, fpr, fnr = calculate_eval_metrics(
        truth, pred, verbose=verbose)
    roc, roc_auc, tpr, thresholds = None, None, None, None

    if anomaly_score is not None:
        fpr, tpr, thresholds = metrics.roc_curve(truth, anomaly_score)
        roc_auc = metrics.roc_auc_score(truth, anomaly_score)
        fnr = 1 - tpr
        if verbose:

            if preset_th is None:
                # EER
                use_eer = True
                idx = np.argmin(np.abs(fpr-fnr))
                preset_th = thresholds[idx]
            else:
                use_eer = False
                preset_th = preset_th

            pred_on_th = np.zeros(truth.shape)
            pred_on_th[anomaly_score > preset_th] = 1

            if use_eer:
                print('\n')
                print('----------------------At EER-----------------------')
                print("Threshold at approx EER:", preset_th)
            else:
                print('\n')
                print(f'----------------------At threshold {preset_th}-----------------------')
                print(f"Threshold at threshold :", preset_th)

            calculate_eval_metrics(truth, pred_on_th, verbose=verbose)
            print("ROC AUC: ", roc_auc)

    return tp, fp, fn, tn, acc, prec, rec, f1, fpr, tpr, thresholds, roc_auc


def setLearningRate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def plotsignal(sigs):

    T = len(sigs[0])
    t = np.linspace(0, T, num = T)
    signal0 = sigs[0][:,0,:]
    print("signal0.shape", signal0.shape)

    signal1 = sigs[1][:,0,:]
    print("signal1.shape", signal1.shape)

    plt.plot(t, signal0)
    plt.plot(t, signal1)
    plt.ylim(-2, 2)
    plt.show()

def seq_win_vectorize(seq, window_size, n_samples=None):

    res = []

    if n_samples is None:
        for i in range(len(seq)-window_size+1):
            res.append(seq[i: i+window_size,:].reshape((-1)))
    else:
        while len(res) < n_samples:
            start = np.random.randint(low=0, high=len(seq)-window_size+1)
            res.append(seq[start: start+window_size,:].reshape((-1)))

    return np.array(res)

def plot_seq(seqs, T=None, start=0, xlabel=None, ylabel=None, title=None, figsize=None,
    include_legend=True, fname=None, sci_notation=True, **kwargs):
    """
        Plot squences for visualization and debug.
        Args:
            Seqs: a (ordered) dictionary of sequences (key, value). Key is
            squence label, value is the sequence to plot.
            T: plot partial sequence, i.e., seq[:T]. If None, plot the
            whole sequence.
    """
    fig = plt.figure(figsize=figsize)

    if 'linewidths' in kwargs:
        linewidths = kwargs['linewidths']
    else:
        linewidths = [2]*len(seqs.items())


    for i, item in enumerate(seqs.items()):
        k, v = item
        end = T if T is not None else len(v)
        t = np.arange(start, end, 1.0)

        if 'markers' in kwargs:
            seq_plot, = plt.plot(t, v[start:end], kwargs['markers'][i], linewidth=linewidths[i])
        else:
            seq_plot, = plt.plot(t, v[start:end])

        seq_plot.set_label(k)

    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'][0], kwargs['ylim'][1])


    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.xlabel(ylabel)

    if title is not None:
        plt.title(title)

    if include_legend:
        if 'legend_location' in kwargs:
            plt.legend(loc=kwargs['legend_location'])
        else:
            plt.legend(loc='upper right')

    if 'xticks' in kwargs.keys():
        if 'xticks_locations' in kwargs.keys():
            plt.xticks(kwargs['xticks_locations'], kwargs['xticks'])
        else:
            plt.xticks(range(len(kwargs['xticks'])), kwargs['xticks'])

    if fname is not None:
        plt.savefig(fname=fname, dpi=fig.dpi)

    if not sci_notation:
        plt.ticklabel_format(style='plain')

    plt.show(block = False)

def plot_hist(data, xlabel=None, ylabel=None, title=None):
    """
        Plot histograms for visualization and debug.
        Args:
            data: a dictionary of sequences (key, value). Key is label,
            value is the points to generate a historam.
    """
    plt.figure()
    for k, v in data.items():
        plt.hist(v, label=k)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.xlabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.legend(loc="lower right")
    plt.show(block = False)


def plot_cdf(data, xlabel=None, ylabel=None, title=None):
    """
        Plot cumulative distribution function for visualization and debug.
        Args:
            data: a dictionary of sequences (key, value). Key is label,
            value is the points to generate the cdf.
    """
    plt.figure()
    for k, v in data.items():
        plt.plot(np.sort(v), np.linspace(0, 1, len(v), endpoint=False), label=k)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.xlabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.legend(loc="lower right")
    plt.show(block = False)


def read_csv_file(filename, split=' ', remove_end=True, dtype=np.float32):
    """
        Read csv file.
        Args:
            split: split of entries in a line.
            remove_end: remove the '\n' at the end of the line.
        Returns:
            A np.array of shape [TimeFrame, Features]
    """

    with open(filename, 'r') as f:
        data = []
        for linenum, line in enumerate(f):
            line_list = line.split(split)
            if remove_end:
                # Remove '\n' at the end
                line_list = line_list[:-1]
            data.append(line.split(split)[:-1])
        data = np.array(data, dtype=dtype)
    return data

def p_to_anomaly_score(p_value):
    return -np.log10(p_value+1e-300)

def create_parser():
    import argparse
    parser = argparse.ArgumentParser()

    # Training or Testing?
    parser.add_argument('--training', dest='training', action='store_true', help='Flag for training')
    parser.add_argument('--testing', dest='training', action='store_false', help='Flag for testing')
    parser.set_defaults(training=True)

    parser.add_argument('--finetune', dest='finetune', action='store_true', help='Flag for fine tune')
    parser.add_argument('--nofinetune', dest='finetune', action='store_false', help='Flag for fine tune')
    parser.set_defaults(finetune=False)

    parser.add_argument('--allanomalyscores', dest='allanomalyscores', action='store_true', help='Eval anomaly scores for all scenarios')
    parser.set_defaults(allanomalyscores=False)
    parser.add_argument('--useexistinganomalyscores', dest='useexistinganomalyscores', action='store_true', help='use existing anomaly scores')
    parser.set_defaults(useexistinganomalyscores=False)

    # Real data (private) or dummy data?
    parser.add_argument('--dummy', dest='dummydata', action='store_true', help='If dummy data is used instead of an input file')
    parser.set_defaults(dummydata=False)

    # LSTM Network config, can use other sequential models
    parser.add_argument('--Nhidden', type = int, default = 64, help='Number of hidden nodes in a LSTM cell')

    # Training parameters
    parser.add_argument('--Nbatches', type = int, default = 100, help='Number of batches in training')
    parser.add_argument('--BatchSize', type = int, default = 16, help='Size of a batch in training')
    parser.add_argument('--ChunkSize', type = int, default = 500, help='The length of a chunk in training')
    parser.add_argument('--SubseqLen', type = int, default = 5000, help='The length of the randomly selected sequence for training')
    parser.add_argument('--LearningRate', type = float, default = 1e-2, help='The initial learning rate of the Adam optimizer')
    parser.add_argument('--AMSGrad', type = bool, default = True, help='Whether the AMSGrad variant is used')
    parser.add_argument('--Eps', type = float, default = 1e-3, help='The term added to the denominator to improve numerical stability')
    parser.add_argument('--LRdecrease', type = int, default = 10, help='The number of batches that are processed each time before the learning rate is divided by 2')

    # Statistic test config
    parser.add_argument('--RED_collection_len', type = int, default = 1, help='The number of readings whose prediction errors are added as a data point')
    parser.add_argument('--RED_points', type = int, default = 100, help='The number of data points that are collected at a time on the testing data to form a testing RED')
    parser.add_argument('--Pvalue_th', type = float, default = 0.05, help='The threshold of p-value in KS test')

    # Loaddata
    # Sequential data in the form of (Timeframe, Features)
    # Training only leverages normal data. Abnormaly data only for testing.
    parser.add_argument('--data_dir', type = str, default = "data/", help='Data directory')
    parser.add_argument('--normal_data_name_train', type = str, default = "train_normal.npy", help='The file name of training normal data')
    parser.add_argument('--normal_data_name_test', type = str, default = "test_normal.npy", help='The file name of testing normal data')
    parser.add_argument('--normal_data_name_ref_and_val', type = str, default = "ref_and_val_normal.npy", help='The file name of testing normal data')
    parser.add_argument('--abnormal_data_name', type = str, default = "test_abnormal.npy", help='The file name of abnormal data')

    # Save and load model. Save after training. Load before testing.
    parser.add_argument('--save_model_dir', type = str, default = "checkpoints/", help='The directory to save the model')
    parser.add_argument('--save_model_name', type = str, default = "AnomalyDetector.ckpt", help='The file name of the saved model')

    parser.add_argument('--load_model_dir', type = str, default = "checkpoints/", help='The directory to load the model')
    parser.add_argument('--load_model_name', type = str, default = "AnomalyDetector.ckpt", help='The file name of the model to be loaded')

    # Debug and GPU config
    parser.add_argument('--debug', dest='debug', action='store_true', help='Whether debug information will be printed')
    parser.set_defaults(debug=False)
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Whether GPU acceleration is enabled')
    parser.set_defaults(gpu=False)

    args = parser.parse_args()
    return args

# [0, 32, 33, 27, 1, 13, 3, 19, 15, 28, 2, 6, 31]
id_to_feature = {
    0: 'Ins',
    1: 'L1D read access (# load)',
    2: 'L1D read miss',
    3: 'L1D write access (# store)',
    4: 'L1D write miss',
    5: 'L1D prefetch miss',
    6: 'L1I read miss',
    7: 'LLC read access',
    8: 'LLC read miss',
    9: 'LLC write access',
    10: 'LLC write miss',
    11: 'LLC prefetch access',
    12: 'LLC prefetch miss',
    13: 'DTLB read access',
    14: 'DTLB read miss',
    15: 'DTLB write access',
    16: 'DTLB write miss',
    17: 'ITLB read access',
    18: 'ITLB read miss',
    19: 'BPU read access',
    20: 'BPU read miss',
    21: 'Cache node read access',
    22: 'Cache node read miss',
    23: 'Cache node write access',
    24: 'Cache node write miss',
    25: 'Cache node prefetch access',
    26: 'Cache node prefetch miss',
    27: 'cycles',
    28: 'branch instructions',
    29: 'branch prediction miss',
    30: 'page faults',
    31: 'context switch',
    32: 'stall_during_issue',
    33: 'stall_during_retirement',
    34: 'Time stamp',
}

spec_benchmarks = ('perlbench', 'bzip2', 'gcc', 'mcf', 'milc', 'namd',
    'gobmk', 'soplex', 'povray', 'hmmer', 'sjeng', 'libquantum',
    'h264ref', 'lbm', 'omnetpp', 'astar')

def f_score(pos, neg):
    neg_mean = np.mean(neg)
    neg_var = np.var(neg)
    pos_mean = np.mean(pos)
    pos_var = np.var(pos)

    eps = 1e-8
    return ((pos_mean-neg_mean) ** 2) / (neg_var+pos_var+eps)
