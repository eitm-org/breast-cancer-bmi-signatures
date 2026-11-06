#!/usr/bin/env python3
# Library of utility functions to fit ML models from scikit-learn using their default hyperparameters.
# The main functions in the library is:
#      build_ml_models3()
# This function trains the following models: logistic regression (log), k-nearest neighbor (knn),
# multi-layer perceptron (mlp), support vector machines (svm), decision trees (dtr), XGBoost (xgb),
# random forest (rfo), XGBoost with grid search (xgb_gs), and decision trees with grid search (dtr_gs)
#
# The function returns 3 objects: a results dataframe with performance metrics for the models, a
# key/value dictionary with the trained models, and a key/value dictionary with model predictions
# using the test dataset.
#
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              matthews_corrcoef, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ----- Constants ------------------------------------------------------------------
NUM_DEC = 2   # Number of decimals for rounding results
NUM_LOG_ITER = 1000
NUM_MLP_ITER = 1000
BASELINE_TRIAL_SEED = 15
NUM_CROSSVAL_FOLDS = 4

XGB_SEARCH_PARAMS = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.0001, 0.001, 0.01]
}

DTR_SEARCH_PARAMS = {
    "random_state": list(range(10, 30)),
    "criterion": ["gini", "entropy"],
    "max_depth": list(range(2, 7)),
    "min_samples_split": list(range(3, 5)),
    "min_samples_leaf": list(range(2, 5))
}

# ----- Functions ------------------------------------------------------------------------

def build_ml_models3(Xtrn: pd.DataFrame, ytrn: pd.DataFrame,
                     Xtst: pd.DataFrame, ytst: pd.DataFrame,
                     verbose=True, grid_search=True) -> tuple[pd.DataFrame, dict, dict]:
    """Build baseline models necessary to perform ML analysis using scikit-learn methods with default parameters.
    The function has been designed to operate with data matrices that represent foundation model vectors.
    Args:
        Xtrn (pd.DataFrame): Training data of size M x N, where M is the number of training samples and N is the
        number of vector elements. Each row of Xtrn represents a slide vector of size N.
        ytrn (pd.DataFrame): The binary outcome for samples in the training set. This dataframe/series has a size of M.
        Xtst (pd.DataFrame): Test data of size P x N, where P is the number of test samples and N is the number of
        vector elements. Each row of Xtst represents a slide vector of size N.
        ytst (pd.DataFrame): The binary outcome for samples in the test set. This dataframe/series has a size of M.
        verbose (boolean): If True, it prints results as they are computed.
        grid_search (boolean): If True, it runs grid search algorithms.
    Returns:
        A tuple of 3 objects: a pandas dataframe with perf results, a dictionary with trained models, and a
        dictionary with predictions for the test dataset.
        The dictionaries use the following access keys: 'log', 'knn', 'svc', 'mlp', 'svm', 'dtr', 'xgb',
        'rfo', 'xgb_gs', and 'dtr_gs'.

    """

    # Build default scikit-learn models using the test and train datasets
    log = LogisticRegression(max_iter=NUM_LOG_ITER, random_state=BASELINE_TRIAL_SEED)
    knn = KNeighborsClassifier()
    mlp = MLPClassifier(max_iter=NUM_MLP_ITER, random_state=BASELINE_TRIAL_SEED)
    svm = SVC(probability=True, random_state=BASELINE_TRIAL_SEED)
    dtr = DecisionTreeClassifier(random_state=BASELINE_TRIAL_SEED)
    xgb = XGBClassifier(random_state=BASELINE_TRIAL_SEED)
    rfo = RandomForestClassifier(random_state=BASELINE_TRIAL_SEED)

    log.fit(Xtrn, ytrn)
    knn.fit(Xtrn, ytrn)
    mlp.fit(Xtrn, ytrn)
    svm.fit(Xtrn, ytrn)
    dtr.fit(Xtrn, ytrn)
    xgb.fit(Xtrn, ytrn)
    rfo.fit(Xtrn, ytrn)

    log_pred = log.predict(Xtst)
    log_prob = log.predict_proba(Xtst)[:, 1]

    knn_pred = knn.predict(Xtst)
    knn_prob = knn.predict_proba(Xtst)[:, 1]

    mlp_pred = mlp.predict(Xtst)
    mlp_prob = mlp.predict_proba(Xtst)[:, 1]

    svm_pred = svm.predict(Xtst)
    svm_prob = svm.predict_proba(Xtst)[:, 1]

    dtr_pred = dtr.predict(Xtst)
    dtr_prob = dtr.predict_proba(Xtst)[:, 1]

    xgb_pred = xgb.predict(Xtst)
    xgb_prob = xgb.predict_proba(Xtst)[:, 1]

    rfo_pred = rfo.predict(Xtst)
    rfo_prob = rfo.predict_proba(Xtst)[:, 1]

    res_all = dict()
    res_all["log"] = perf_metrics(log_pred, ytst, log_prob, title="LOG", verbose=verbose)
    res_all["knn"] = perf_metrics(knn_pred, ytst, knn_prob, title="KNN", verbose=verbose)
    res_all["mlp"] = perf_metrics(mlp_pred, ytst, mlp_prob, title="MLP", verbose=verbose)
    res_all["svm"] = perf_metrics(svm_pred, ytst, svm_prob, title="SVM", verbose=verbose)
    res_all["dtr"] = perf_metrics(dtr_pred, ytst, dtr_prob, title="DTR", verbose=verbose)
    res_all["xgb"] = perf_metrics(xgb_pred, ytst, xgb_prob, title="XGB", verbose=verbose)
    res_all["rfo"] = perf_metrics(rfo_pred, ytst, rfo_prob, title="RFO", verbose=verbose)

    models_dict = dict()
    models_dict["log"] = log
    models_dict["knn"] = knn
    models_dict["mlp"] = mlp
    models_dict["svm"] = svm
    models_dict["dtr"] = dtr
    models_dict["xgb"] = xgb
    models_dict["rfo"] = rfo

    preds_dict = dict()
    preds_dict["log"] = [int(val) for val in log_pred]
    preds_dict["knn"] = [int(val) for val in knn_pred]
    preds_dict["mlp"] = [int(val) for val in mlp_pred]
    preds_dict["svm"] = [int(val) for val in svm_pred]
    preds_dict["dtr"] = [int(val) for val in dtr_pred]
    preds_dict["xgb"] = [int(val) for val in xgb_pred]
    preds_dict["rfo"] = [int(val) for val in rfo_pred]

    print(">> Generated models with default sklearn parameters...")

    if grid_search:
        print(">> Starting grid search for XGBoost...")
        start1 = time.time()
        xgb2 = XGBClassifier()
        grid_xgb = GridSearchCV(estimator=xgb2, param_grid=XGB_SEARCH_PARAMS, cv=NUM_CROSSVAL_FOLDS,
                                scoring='accuracy', verbose=1, n_jobs=-1)

        grid_xgb.fit(Xtrn, ytrn)

        if verbose:
            print("\nXGBoost Grid Search cross-validation results:")
            print("Best parameters:", grid_xgb.best_params_)
            print("Best estimator:", grid_xgb.best_estimator_)
            print("Best score from cross-validation:", round(grid_xgb.best_score_, NUM_DEC))
            print()

        # Compute test scores
        preds_xgb_gs = grid_xgb.best_estimator_.predict(Xtst)
        probs_xgb_gs = grid_xgb.best_estimator_.predict_proba(Xtst)[:, 1]
        res_all["xgb_gs"] = perf_metrics(preds_xgb_gs, ytst, probs_xgb_gs, title="XGB_GS:", verbose=verbose)

        elapsed1 = int(time.time() - start1)
        print(f">> Finished grid search for XGBoost in {elapsed1} seconds.")

        print(">> Starting grid search for Decision Trees...")
        start2 = time.time()
        dtr2 = DecisionTreeClassifier()
        grid_dtr = GridSearchCV(estimator=dtr2, param_grid=DTR_SEARCH_PARAMS, cv=NUM_CROSSVAL_FOLDS,
                                scoring='accuracy', verbose=1, n_jobs=-1)

        grid_dtr.fit(Xtrn, ytrn)

        if verbose:
            print("\nGrid search cross-validation results:")
            print("Best parameters:", grid_dtr.best_params_)
            print("Best estimator:", grid_dtr.best_estimator_)
            print("Best score from cross-validation:", round(grid_dtr.best_score_, NUM_DEC))
            print()

        preds_dtr_gs = grid_dtr.best_estimator_.predict(Xtst)
        probs_dtr_gs = grid_dtr.best_estimator_.predict_proba(Xtst)[:, 1]
        res_all["dtr_gs"] = perf_metrics(preds_dtr_gs, ytst, probs_dtr_gs, title="DTR_GS", verbose=verbose)

        elapsed2 = int(time.time() - start2)
        print(f">> Finished grid search for Dec. Trees in {elapsed2} seconds.")

        models_dict["xgb_gs"] = grid_xgb.best_estimator_
        preds_dict["xgb_gs"] = preds_xgb_gs
        models_dict["dtr_gs"] = grid_dtr.best_estimator_
        preds_dict["dtr_gs"] = preds_dtr_gs

    dframe = results_to_dframe(res_all)
    return dframe, models_dict, preds_dict


def results_to_dframe(results: dict) -> pd.DataFrame:
    """Convert results from the build_ml_models() function into a dataframe that shows 4 performance
    metrics: acc, auc, mcc and f1. """
    methods = list(results.keys())
    acc_data = []
    auc_data = []
    mcc_data = []
    f1_data = []
    spe_data = []
    sen_data = []
    c00_data = []
    c01_data = []
    c10_data = []
    c11_data = []

    for method in methods:
        acc_data.append(results[method]['acc'])
        auc_data.append(results[method]['auc'])
        mcc_data.append(results[method]['mcc'])
        f1_data.append(results[method]['f1s'])
        spe_data.append(results[method]['spe'])
        sen_data.append(results[method]['sen'])
        c00_data.append(results[method]['cm'][0][0])
        c01_data.append(results[method]['cm'][0][1])
        c10_data.append(results[method]['cm'][1][0])
        c11_data.append(results[method]['cm'][1][1])

    res_dict = {'method': methods, 'acc': acc_data, 'auc': auc_data, 'mcc': mcc_data, 'f1': f1_data,
                'spe': spe_data, 'sen': sen_data,
                'c00': c00_data, 'c01': c01_data, 'c10': c10_data, 'c11': c11_data}
    res_df = pd.DataFrame(res_dict)

    return res_df

def perf_metrics(y_pred, y_true, y_prob_class1, title="", verbose=True):
    """Compute performance metrics given a binary prediction (y_pred) and the equivalent
    prediction probabilities (y_prob) by comparing the results with the true labels (y_true).

    The y_pred, t_true, and y_prob arrays (typically numpy arrays extracted from scikit-learn functions) must
    have the same size.
    Args:
        y_pred (numpy array): predicted class labels
        y_true (numpy array): true class labels
        y_prob_class1 (numpy array): predicted class probabilities for the positive class. In scikit-learn, the
        predict_proba functions return an Mx2 matrix with the second column containing class probabilities
        for the positive class (class 1). In scikit-learn predict_proba functions it corresponds to [:, 1].
        title (string): title added to the printed results if using the verbose option.
        verbose (boolean): If set to True, it prints the results as soon as they are computed.

    Returns:
        A key/value dictionary where the keys represent performance metrics and the values represent
        computed scores.
    """

    acc = round(accuracy_score(y_true, y_pred), NUM_DEC)
    auc = round(roc_auc_score(y_true, y_prob_class1), NUM_DEC)
    mcc = round(matthews_corrcoef(y_true, y_pred), NUM_DEC)
    pre = round(precision_score(y_true, y_pred), NUM_DEC)
    rec = round(recall_score(y_true, y_pred), NUM_DEC)
    f1s = round(f1_score(y_true, y_pred), NUM_DEC)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spe = round(tn / (tn + fp), NUM_DEC)
    sen = round(tp / (tp + fn), NUM_DEC)

    if verbose:
        print(f"--- {title} ---")
        print(f"ACC: {acc}, AUC: {auc}, MCC: {mcc}, F1: {f1s}, PRE: {pre}, REC: {rec}, SPE: {spe}, SEN: {sen}")
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in cm]))

    return {
        "acc": float(acc),
        "auc": float(auc),
        "mcc": float(mcc),
        "f1s": float(f1s),
        "pre": float(pre),
        "rec": float(rec),
        "spe": float(spe),
        "sen": float(sen),
        "cm": cm
    }

def build_ml_models3_unsupervised(Xtrn: pd.DataFrame, ytrn: pd.DataFrame,
                     Xtst: pd.DataFrame, verbose=True, grid_search=True) -> tuple[pd.DataFrame, dict, dict]:
    """Build baseline models necessary to perform ML analysis using scikit-learn methods with default parameters.
    The function has been designed to operate with data matrices that represent foundation model vectors.
    Args:
        Xtrn (pd.DataFrame): Training data of size M x N, where M is the number of training samples and N is the
        number of vector elements. Each row of Xtrn represents a slide vector of size N.
        ytrn (pd.DataFrame): The binary outcome for samples in the training set. This dataframe/series has a size of M.
        Xtst (pd.DataFrame): Test data of size P x N, where P is the number of test samples and N is the number of
        vector elements. Each row of Xtst represents a slide vector of size N.
        ytst (pd.DataFrame): The binary outcome for samples in the test set. This dataframe/series has a size of M.
        verbose (boolean): If True, it prints results as they are computed.
        grid_search (boolean): If True, it runs grid search algorithms.
    Returns:
        A tuple of 3 objects: a pandas dataframe with perf results, a dictionary with trained models, and a
        dictionary with predictions for the test dataset.
        The dictionaries use the following access keys: 'log', 'knn', 'svc', 'mlp', 'svm', 'dtr', 'xgb',
        'rfo', 'xgb_gs', and 'dtr_gs'.

    """

    # Build default scikit-learn models using the test and train datasets
    log = LogisticRegression(max_iter=NUM_LOG_ITER, random_state=BASELINE_TRIAL_SEED)
    knn = KNeighborsClassifier()
    mlp = MLPClassifier(max_iter=NUM_MLP_ITER, random_state=BASELINE_TRIAL_SEED)
    svm = SVC(probability=True, random_state=BASELINE_TRIAL_SEED)
    dtr = DecisionTreeClassifier(random_state=BASELINE_TRIAL_SEED)
    xgb = XGBClassifier(random_state=BASELINE_TRIAL_SEED)
    rfo = RandomForestClassifier(random_state=BASELINE_TRIAL_SEED)

    log.fit(Xtrn, ytrn)
    knn.fit(Xtrn, ytrn)
    mlp.fit(Xtrn, ytrn)
    svm.fit(Xtrn, ytrn)
    dtr.fit(Xtrn, ytrn)
    xgb.fit(Xtrn, ytrn)
    rfo.fit(Xtrn, ytrn)

    log_pred = log.predict(Xtst)
    log_prob = log.predict_proba(Xtst)[:, 1]

    knn_pred = knn.predict(Xtst)
    knn_prob = knn.predict_proba(Xtst)[:, 1]

    mlp_pred = mlp.predict(Xtst)
    mlp_prob = mlp.predict_proba(Xtst)[:, 1]

    svm_pred = svm.predict(Xtst)
    svm_prob = svm.predict_proba(Xtst)[:, 1]

    dtr_pred = dtr.predict(Xtst)
    dtr_prob = dtr.predict_proba(Xtst)[:, 1]

    xgb_pred = xgb.predict(Xtst)
    xgb_prob = xgb.predict_proba(Xtst)[:, 1]

    rfo_pred = rfo.predict(Xtst)
    rfo_prob = rfo.predict_proba(Xtst)[:, 1]

    res_all = dict()
    res_all["log"] = log_prob
    res_all["knn"] = knn_prob
    res_all["mlp"] = mlp_prob
    res_all["svm"] = svm_prob
    res_all["dtr"] = dtr_prob
    res_all["xgb"] = xgb_prob
    res_all["rfo"] = rfo_prob

    models_dict = dict()
    models_dict["log"] = log
    models_dict["knn"] = knn
    models_dict["mlp"] = mlp
    models_dict["svm"] = svm
    models_dict["dtr"] = dtr
    models_dict["xgb"] = xgb
    models_dict["rfo"] = rfo

    preds_dict = dict()
    preds_dict["log"] = [int(val) for val in log_pred]
    preds_dict["knn"] = [int(val) for val in knn_pred]
    preds_dict["mlp"] = [int(val) for val in mlp_pred]
    preds_dict["svm"] = [int(val) for val in svm_pred]
    preds_dict["dtr"] = [int(val) for val in dtr_pred]
    preds_dict["xgb"] = [int(val) for val in xgb_pred]
    preds_dict["rfo"] = [int(val) for val in rfo_pred]

    print(">> Generated models with default sklearn parameters...")

    #dframe = results_to_dframe(res_all)
    dframe = pd.DataFrame(res_all)
    return dframe, models_dict, preds_dict