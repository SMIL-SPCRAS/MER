import numpy as np
from sklearn import metrics

def weighted_accuracy(test_truth_emo, test_preds_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))

    return (tp * (n / (p + 1e-16)) + tn) / (2 * n + 1e-16)

def cmu_weighted_accuracy(targets, predicts):
    targets = np.asarray(targets)
    predicts = np.asarray(predicts)
    wa_scores = []
    for i in range(0, 6):
        wa_scores.append(weighted_accuracy(targets[:, i], predicts[:, i]))
    return wa_scores

def cmu_micro_f1(targets, predicts):
    targets = np.asarray(targets)
    predicts = np.asarray(predicts)

    f1_macro_scores = []
    for i in range(0, 6):
        cr = metrics.classification_report(targets[:, i], predicts[:, i], output_dict=True, zero_division=0)
        f1_macro_scores.append(cr['micro avg']['f1-score'])
    return f1_macro_scores

def cmu_accuracy(targets, predicts):
    temp = 0
    for i in range(targets.shape[0]):
        numerator = sum(np.logical_and(targets[i], predicts[i]))
        denominator = sum(np.logical_or(targets[i], predicts[i]))
        if numerator == 0 and denominator == 0:
            numerator = 1
            denominator = 1
        temp += numerator / denominator
    return temp / targets.shape[0]

def cmu_macro_f1(targets, predicts):
    temp = 0
    for i in range(targets.shape[0]):
        
        if (sum(targets[i]) == 0) and (sum(predicts[i]) == 0):
            temp += 1
        else:
            temp+= (2*sum(np.logical_and(targets[i], predicts[i])))/ (sum(targets[i])+sum(predicts[i]))
    return temp/ targets.shape[0]

def cmu_uar(targets, predicts):
    targets = np.asarray(targets)
    predicts = np.asarray(predicts)

    uar_scores = []
    counter = []
    for i in range(0, 6):
        cr = metrics.classification_report(targets[:, i], predicts[:, i], output_dict=True, zero_division=0)
        # print(cr)
        counter.append(cr['1']['support'])
        uar_scores.append(cr['1']['recall'])
    return uar_scores, counter

def emo_accuracy(targets, predicts):
    cr = metrics.classification_report(targets, predicts, output_dict=True, zero_division=0)
    return cr['weighted avg']['recall']

def emo_uar(targets, predicts):
    cr = metrics.classification_report(targets, predicts, output_dict=True, zero_division=0)
    return cr['macro avg']['recall']


def emo_macro_f1(targets, predicts):
    cr = metrics.classification_report(targets, predicts, output_dict=True, zero_division=0)
    return cr['macro avg']['f1-score']


def emo_micro_f1(targets, predicts):
    cr = metrics.classification_report(targets, predicts, output_dict=True, zero_division=0)
    return cr['weighted avg']['f1-score']