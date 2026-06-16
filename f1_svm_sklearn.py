import warnings
warnings.filterwarnings('ignore')

#CS5811 section 4 - sklearn SVM for F1 finishing-position prediction
#RBF SVM tuned by stratified grid search, with linear-SVM and logreg baselines.

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (f1_score, classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support,
                             cohen_kappa_score, roc_auc_score)
from sklearn.inspection import permutation_importance

#For google colab

# try:
#     from google.colab import files
#     uploaded = files.upload()
# except ImportError:
#     pass



# DATA_PATH = '/content/f1_final_dataset.csv'

DATA_PATH = 'f1_final_dataset.csv'
SEED = 42
KFOLDS = 5
LOG_PATH = 'sklearn_results.txt'

#RBF on 6800 rows + OHE circuits is 25s/fit. Full grid is 4*3*5 = 60 fits ≈ 25 min.
#Set FAST_MODE=True for a 2*2 x 3-fold sweep (5 min) when iterating.
FAST_MODE = False



mpl.rcParams.update({
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

NUMERIC_COLS =['grid', 'q1_gap_to_pole',
                'constructor_standing', 'constructor_points', 'constructor_wins',
                'driver_standing', 'driver_points', 'driver_wins',
                'alt', 'year', 'round']
CAT_COLS= ['circuit_name']
TARGET = 'final_position'
CLASS_ORDER = ['1st', '2nd','3rd', '4th', 'Outside Top 4']


def get_era(year):
    if year <= 2013:
        return 'V8 (2005-2013)'
    if year <= 2021:
        return 'Hybrid (2014-2021)'
    return 'Ground Effect (2022-2025)'




class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()


def load_and_split(path):
    df = pd.read_csv(path)
    df['era'] = df['year'].apply(get_era)

    # grid==0 is a pit-lane start, recode to max+1 so it ranks worst not best



    mg = df.loc[df['grid'] > 0, 'grid'].max()
    df.loc[df['grid'] == 0, 'grid'] = mg + 1

    train = df[df['year'] <= 2022].copy()
    val   = df[df['year'] == 2023].copy()
    test  = df[df['year'] >= 2024].copy()

    print(f'Loaded {len(df):,} rows.')
    print(f'  Train (2005-2022): {len(train):,}')
    print(f'  Val   (2023):      {len(val):,}')
    print(f'  Test  (2024-2025): {len(test):,}')

    print('\nTrain class balance:')
    for cls in CLASS_ORDER:
        n = (train[TARGET] == cls).sum()
        print(f'  {cls:<15} {n:>5}  ({n/len(train)*100:.1f}%)')
    return train, val, test


def build_preprocessor():
    # SVM is scale-sensitive so numeric cols get z-scored.


    return ColumnTransformer([
        ('num', StandardScaler(), NUMERIC_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
    ], remainder='drop')


def tune_svm_rbf(X_train, y_train):
    pipe = Pipeline([
        ('prep', build_preprocessor()),
        ('svm', SVC(kernel='rbf', class_weight='balanced',
                    decision_function_shape='ovr', probability=True,
                    random_state=SEED)),
    ])

    if FAST_MODE:
        param_grid = {'svm__C': [1.0, 5.0],
                      'svm__gamma': ['scale', 0.1]}
        n_splits = 3
    else:
        param_grid = {'svm__C': [0.5, 1.0, 5.0, 10.0],
                      'svm__gamma': ['scale', 0.01, 0.1]}
        n_splits = KFOLDS

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    grid = GridSearchCV(pipe, param_grid, scoring='f1_macro',
                        cv=cv, n_jobs=-1, verbose=1, refit=True)

    n_combos = len(param_grid['svm__C']) * len(param_grid['svm__gamma'])
    print(f"\ngrid search: {n_combos} combos x {n_splits} folds")
    t0 = time.time()
    grid.fit(X_train, y_train)
    print(f'finished in {time.time()-t0:.1f}s')
    print(f'best CV macro-F1: {grid.best_score_:.4f}')
    print(f'best params:      {grid.best_params_}')

    cv_df = pd.DataFrame(grid.cv_results_)[
        ['param_svm__C', 'param_svm__gamma', 'mean_test_score', 'std_test_score']
    ].sort_values('mean_test_score', ascending=False)
    print('\nfull CV results:')
    print(cv_df.to_string(index=False))
    return grid


def fit_baseline(X_train, y_train, kind='linear_svm'):
    if kind == 'linear_svm':
        clf = SVC(kernel='linear', class_weight='balanced',
                  decision_function_shape='ovr', probability=True,
                  random_state=SEED)
    elif kind == 'logreg':
        clf = LogisticRegression(class_weight='balanced',
                                 max_iter=2000, random_state=SEED)
    else:
        raise ValueError(kind)
    pipe = Pipeline([('prep', build_preprocessor()), ('clf', clf)])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate(model, X, y, name):
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, labels=CLASS_ORDER,
                                                  average='macro', zero_division=0)
    wf1 = f1_score(y, y_pred, labels=CLASS_ORDER, average='weighted')

    # macro specificity - per class TNR averaged
    cm = confusion_matrix(y, y_pred, labels=CLASS_ORDER)
    specs = []
    for i in range(len(CLASS_ORDER)):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    spec = float(np.mean(specs))

    auc = np.nan
    try:
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(X)
        else:
            scores = model.decision_function(X)
        auc = roc_auc_score(y, scores, labels=CLASS_ORDER,
                            multi_class='ovr', average='macro')
    except Exception as e:
        print('  AUC failed:', e)

    print(f'\n=== {name} ===')
    print(f'  Accuracy:    {acc:.4f}')
    print(f'  Kappa:       {kappa:.4f}')
    print(f'  Precision:   {p:.4f}')
    print(f'  Sensitivity: {r:.4f}')
    print(f'  Specificity: {spec:.4f}')
    print(f'  Macro-F1:    {f1:.4f}')
    print(f'  AUC:         {auc:.4f}')
    print(f'  W-F1:        {wf1:.4f}')
    print(classification_report(y, y_pred, labels=CLASS_ORDER, digits=3))

    return {'name': name, 'accuracy': acc, 'kappa': kappa,
            'macro_precision': p, 'macro_sensitivity': r,
            'macro_specificity': spec, 'macro_f1': f1, 'auc': auc,
            'weighted_f1': wf1, 'y_pred': y_pred}


def plot_confusion(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER, rotation=30, ha='right')
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)',
                    ha='center', va='center', color=color, fontsize=8.5)
    plt.colorbar(im, ax=ax, shrink=0.7, label='row-normalised %')
    ax.set_title(title, loc='left', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print('saved', out_path)


def plot_method_comparison(results, out_path):
    names = [r['name'] for r in results]
    macro = [r['macro_f1'] for r in results]
    acc   = [r['accuracy'] for r in results]

    x = np.arange(len(names))
    w = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w/2, macro, w, label='Macro-F1',
                color='#3498db', edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + w/2, acc, w, label='Accuracy',
                color='#95a5a6', edgecolor='black', linewidth=0.5)
    for bar, v in zip(b1, macro):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f'{v:.3f}', ha='center', fontsize=9)
    for bar, v in zip(b2, acc):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f'{v:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')
    ax.set_title('Method comparison on test set (2024-2025)', loc='left', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print('saved', out_path)


def plot_permutation_importance(model, X, y, out_path):
    #using permutation importance (not coefficients) because RBF SVM has no
    #run it on the full pipeline so circuit_name shows up as one feature



    print('\ncomputing permutation importance...')
    feat_names = NUMERIC_COLS + CAT_COLS
    t0 = time.time()
    res = permutation_importance(model, X, y, n_repeats=10, random_state=SEED,
                                 scoring='f1_macro', n_jobs=-1)
    print(f'  done in {time.time()-t0:.1f}s')

    imp_df = pd.DataFrame({
        'feature': feat_names,
        'importance_mean': res.importances_mean,
        'importance_std': res.importances_std,
    }).sort_values('importance_mean', ascending=True)
    print('\npermutation importance (drop in macro-F1 when shuffled):')
    print(imp_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp_df['feature'], imp_df['importance_mean'],
            xerr=imp_df['importance_std'],
            color='#3498db', edgecolor='black', linewidth=0.5,
            error_kw={'ecolor': '#2c3e50', 'elinewidth': 1})
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_xlabel('drop in macro-F1 when feature shuffled (mean ± std, 10 repeats)')
    ax.set_title('Permutation importance - SVM RBF on test set', loc='left', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print('saved', out_path)


def era_stratified_eval(model, train, test_df, feat_cols, out_path):
    #tells us whether the same hyperparams generalise across regulation regimes


    eras = ['V8 (2005-2013)', 'Hybrid (2014-2021)', 'Ground Effect (2022-2025)']
    by_era = {}

    for era in eras:
        era_train = train[train['era'] == era]
        if len(era_train) < 200:
            by_era[era] = (np.nan, 0)
            continue
        X = era_train[feat_cols]
        y = era_train[TARGET]
        pipe = Pipeline([
            ('prep', build_preprocessor()),
            ('svm', SVC(kernel='rbf', class_weight='balanced',
                        C=0.5, gamma=0.01,                        #matches the grid-search winner from tune_svm_rbf
                        decision_function_shape='ovr', probability=True,
                        random_state=SEED)),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
        by_era[era] = (scores.mean(), scores.std())
        print(f'  {era}: macro-F1 = {scores.mean():.4f} ± {scores.std():.4f}  '
              f'n_train={len(era_train)}')
        

        #do

    fig, ax = plt.subplots(figsize=(8, 4.2))
    valid = [e for e in eras if not np.isnan(by_era[e][0])]
    means = [by_era[e][0] for e in valid]
    stds  = [by_era[e][1] for e in valid]
    cols = ['#f39c12', '#3498db', '#27ae60']
    bars = ax.bar(valid, means, yerr=stds, color=cols[:len(valid)],
                  edgecolor='black', linewidth=0.5,
                  error_kw={'ecolor': '#2c3e50', 'elinewidth': 1.5, 'capsize': 4})
    for bar, v, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, v + s + 0.01,
                f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Macro-F1 (5-fold CV within era)')
    ax.set_ylim(0, max(means) + max(stds) + 0.1)
    ax.set_title('Within-era SVM RBF performance (same hyperparams)', loc='left', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print('saved', out_path)





def main():
    import sys
    log_file = open(LOG_PATH, 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)

    print('CS5811 section 4 - sklearn SVM')

    train, val, test = load_and_split(DATA_PATH)
    feat_cols = NUMERIC_COLS + CAT_COLS
    X_train, y_train = train[feat_cols], train[TARGET]
    X_test,  y_test  = test[feat_cols],  test[TARGET]

    #tune RBF SVM


    grid = tune_svm_rbf(X_train, y_train)
    best_rbf = grid.best_estimator_
    rbf_test = evaluate(best_rbf, X_test, y_test, 'SVM (RBF) - test set')
    plot_confusion(y_test, rbf_test['y_pred'],
                   'Confusion matrix - SVM RBF on test set',
                   'fig_confusion_matrix_rbf.png')
    
    #

    print('\nbaselines')
    lin_svm = fit_baseline(X_train, y_train, kind='linear_svm')
    lin_test = evaluate(lin_svm, X_test, y_test, 'SVM (linear) - test set')

    logreg = fit_baseline(X_train, y_train, kind='logreg')
    log_test = evaluate(logreg, X_test, y_test, 'Logistic Regression - test set')


    plot_method_comparison([
        {'name': 'Logistic\n(baseline)', 'macro_f1': log_test['macro_f1'], 'accuracy': log_test['accuracy']},
        {'name': 'SVM\n(linear)',        'macro_f1': lin_test['macro_f1'], 'accuracy': lin_test['accuracy']},
        {'name': 'SVM\n(RBF)',           'macro_f1': rbf_test['macro_f1'], 'accuracy': rbf_test['accuracy']},
    ], 'fig_method_comparison.png')


    print('\n   permutation importance    ')
    plot_permutation_importance(best_rbf, X_test, y_test, 'fig_permutation_importance.png')


    print('\n   era stratified   ')
    era_stratified_eval(best_rbf, train, test, feat_cols, 'fig_era_macro_f1.png')

##

    out = test.copy()
    out['pred_svm_rbf']    = rbf_test['y_pred']
    out['pred_svm_linear'] = lin_test['y_pred']
    out['pred_logreg']     = log_test['y_pred']
    out.to_csv('sklearn_test_predictions.csv', index=False)
    print('saved sklearn_test_predictions.csv')



    print('\nfor comparison.xlsx:')
    for k in ['accuracy', 'kappa', 'macro_precision', 'macro_sensitivity',
              'macro_specificity', 'macro_f1', 'auc']:
        print(f'  {k:20s} {rbf_test[k]:.4f}')

    print('\ndone.')
    log_file.close()


if __name__ == '__main__':
    main()
