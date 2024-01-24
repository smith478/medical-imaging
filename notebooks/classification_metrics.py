import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve


def true_positives(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 1))


def true_negatives(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 0))


def false_negatives(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 1))


def false_positives(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 0))


def get_performance_metrics(y, pred, class_labels, tp=true_positives,
                            tn=true_negatives, fp=false_positives,
                            fn=false_negatives,
                            acc=None, prevalence=None, spec=None,
                            sens=None, ppv=None, npv=None, auc=None, f1=None,
                            thresholds=[]):
    if len(thresholds) != len(class_labels):
        thresholds = [.5] * len(class_labels)

    metrics = [tp, tn, fp, fn, acc, prevalence, sens, spec, ppv, npv, auc, f1]
    metric_names = ["TP", "TN", "FP", "FN", "Accuracy", "Prevalence",
                    "Sensitivity", "Specificity", "PPV", "NPV", "AUC", "F1"]

    df = pd.DataFrame(index=class_labels, columns=metric_names + ["Threshold"])

    for i, label in enumerate(class_labels):
        for metric, name in zip(metrics, metric_names):
            if metric is not None:
                try:
                    if name in ["AUC", "F1"]:
                        df.loc[label, name] = round(metric(y[:, i], pred[:, i]), 3)
                    elif name == "Prevalence":
                        df.loc[label, name] = round(metric(y[:, i]), 3)
                    else:
                        df.loc[label, name] = round(metric(y[:, i], pred[:, i], thresholds[i]), 3)
                except Exception as e:
                    print(f"Exception occurred in {name}: {e}")
                    df.loc[label, name] = np.NAN
            else:
                df.loc[label, name] = "Not Defined"
        df.loc[label, "Threshold"] = round(thresholds[i], 3)

    return df

def print_confidence_intervals(class_labels, statistics):
    df = pd.DataFrame(columns=["Mean AUC (CI 5%-95%)"])
    for i in range(len(class_labels)):
        mean = statistics.mean(axis=1)[i]
        max_ = np.quantile(statistics, .95, axis=1)[i]
        min_ = np.quantile(statistics, .05, axis=1)[i]
        df.loc[class_labels[i]] = ["%.2f (%.2f-%.2f)" % (mean, min_, max_)]
    return df


def get_curve(gt, pred, target_names, curve='roc'):
    for i in range(len(target_names)):
        if curve == 'roc':
            curve_function = roc_curve
            auc_roc = roc_auc_score(gt[:, i], pred[:, i])
            label = target_names[i] + " AUC: %.3f " % auc_roc
            xlabel = "False positive rate"
            ylabel = "True positive rate"
            a, b, _ = curve_function(gt[:, i], pred[:, i])
            plt.figure(1, figsize=(7, 7))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(a, b, label=label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)
        elif curve == 'prc':
            precision, recall, _ = precision_recall_curve(gt[:, i], pred[:, i])
            average_precision = average_precision_score(gt[:, i], pred[:, i])
            label = target_names[i] + " Avg.: %.3f " % average_precision
            plt.figure(1, figsize=(7, 7))
            plt.step(recall, precision, where='post', label=label)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)
            

def get_accuracy(y, pred, th=0.5):
    """
    Compute accuracy of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        accuracy (float): accuracy of predictions at threshold
    """
    TP = true_positives(y, pred, th)
    FP = false_positives(y, pred, th)
    TN = true_negatives(y, pred, th)
    FN = false_negatives(y, pred, th)

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0.0
    
    return accuracy

def get_prevalence(y):
    """
    Compute prevalence of positive cases.

    Args:
        y (np.array): ground truth, size (n_examples)
    Returns:
        prevalence (float): prevalence of positive cases
    """
    total = len(y)
    prevalence = np.sum(y) / total if total else 0.0
    
    return prevalence

def get_sensitivity(y, pred, th=0.5):
    """
    Compute sensitivity of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        sensitivity (float): probability that our test outputs positive given that the case is actually positive
    """
    TP = true_positives(y, pred, th)
    FN = false_negatives(y, pred, th)

    total = TP + FN
    sensitivity = TP / total if total else 0.0
    
    return sensitivity

def get_specificity(y, pred, th=0.5):
    """
    Compute specificity of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        specificity (float): probability that the test outputs negative given that the case is actually negative
    """
    TN = true_negatives(y, pred, th)
    FP = false_positives(y, pred, th)
    
    total = TN + FP
    specificity = TN / total if total else 0.0
    
    return specificity

def get_ppv(y, pred, th=0.5):
    """
    Compute PPV of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        PPV (float): positive predictive value of predictions at threshold
    """
    TP = true_positives(y, pred, th)
    FP = false_positives(y, pred, th)

    total = TP + FP
    PPV = TP / total if total else 0.0
    
    return PPV

def get_npv(y, pred, th=0.5):
    """
    Compute NPV of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        NPV (float): negative predictive value of predictions at threshold
    """
    TN = true_negatives(y, pred, th)
    FN = false_negatives(y, pred, th)

    total = TN + FN
    NPV = TN / total if total else 0.0
    
    return NPV


def bootstrap_auc(y, pred, classes, bootstraps=100, fold_size=1000):
    statistics = np.full((len(classes), bootstraps), 0.0)

    for c in range(len(classes)):
        df = pd.DataFrame({'y': y[:, c], 'pred': pred[:, c]})
        prevalences = df.groupby('y').size() / len(df)

        for i in range(bootstraps):
            samples = df.groupby('y').apply(lambda group: group.sample(n=int(fold_size * prevalences[group.name]), replace=True))
            y_sample = samples.y.values
            pred_sample = samples.pred.values

            try:
                statistics[c, i] = roc_auc_score(y_sample, pred_sample)
            except ValueError:
                pass  # Keep default score of 0 if AUC cannot be calculated

    return statistics


def plot_calibration_curve(y, pred, target_columns):
    plt.figure(figsize=(20, 20))
    for i in range(len(target_columns)):
        plt.subplot(4, 4, i + 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(y[:,i], pred[:,i], n_bins=20)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(mean_predicted_value, fraction_of_positives, marker='.')
        plt.xlabel("Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.title(target_columns[i])
    plt.tight_layout()
    plt.show()