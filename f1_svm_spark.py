
#For google colab

# try:
#     from google.colab import files
#     uploaded = files.upload()
# except ImportError:
#     pass  




#section 5, Spark version of the LinearSVC.
#part A: same train/test as section 4, just on Spark
#part B: scale up with bootstrap resampling, time sklearn vs spark




import warnings
warnings.filterwarnings('ignore')

import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline as SparkPipeline
from pyspark.ml.feature import (VectorAssembler, StandardScaler as SparkScaler,
                                StringIndexer, OneHotEncoder as SparkOneHotEncoder)
from pyspark.ml.classification import LinearSVC, OneVsRest

# sklearn used for the part B timing comparison
from sklearn.svm import LinearSVC as SkLinearSVC
from sklearn.preprocessing import StandardScaler as SkScaler, OneHotEncoder as SkOneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline

# DATA_PATH = '/content/f1_final_dataset.csv'

DATA_PATH = 'f1_final_dataset.csv'
SEED = 42
LOG_PATH = 'spark_results.txt'

#tried 10M but my laptop down, 1M is enough to see the crossover


SCALING_SIZES = [10_000, 100_000, 1_000_000]

mpl.rcParams.update({
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

NUM_COLS = ['grid', 'q1_gap_to_pole',
            'constructor_standing', 'constructor_points', 'constructor_wins',
            'driver_standing', 'driver_points', 'driver_wins',
            'alt', 'year', 'round']
CAT_COL = 'circuit_name'
TARGET = 'final_position'
CLASS_ORDER = ['1st', '2nd', '3rd', '4th', 'Outside Top 4']


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()


def build_spark_session():
    spark = (SparkSession.builder
             .appName('CS5811-F1-LinearSVC')
             .master('local[*]')
             .config('spark.driver.memory', '4g')
             .config('spark.sql.shuffle.partitions', '8')
             .config('spark.ui.showConsoleProgress', 'false')
             .getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    print('Spark version:', spark.version)
    print('master:', spark.sparkContext.master)
    print('parallelism:', spark.sparkContext.defaultParallelism)
    return spark






def load_and_split_spark(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=True)

    #grid==0 is a pit-lane start, treat as worst grid slot


    max_grid = df.filter(F.col('grid') > 0).agg(F.max('grid')).collect()[0][0]
    df = df.withColumn('grid',
                       F.when(F.col('grid') == 0, max_grid + 1).otherwise(F.col('grid')))

    #mllib won't take string labels

    label_map = {lab: float(i) for i, lab in enumerate(CLASS_ORDER)}
    expr = F.when(F.col(TARGET) == CLASS_ORDER[0], label_map[CLASS_ORDER[0]])
    for c in CLASS_ORDER[1:]:
        expr = expr.when(F.col(TARGET) == c, label_map[c])
    df = df.withColumn('label', expr)

    train = df.filter(F.col('year') <= 2022).cache()
    val   = df.filter(F.col('year') == 2023).cache()
    test  = df.filter(F.col('year') >= 2024).cache()

    print(f'\ntrain {train.count():,} | val {val.count():,} | test {test.count():,}')
    return df, train, val, test, label_map





def build_spark_pipeline():
    indexer = StringIndexer(inputCol=CAT_COL, outputCol='circuit_idx',
                            handleInvalid='keep')
    encoder = SparkOneHotEncoder(inputCols=['circuit_idx'], outputCols=['circuit_ohe'])
    assembler = VectorAssembler(inputCols=NUM_COLS + ['circuit_ohe'],
                                outputCol='features_unscaled',
                                handleInvalid='keep')
    


    #withMean=False: sparse OHE can't be centred in spark, will throw

    scaler = SparkScaler(inputCol='features_unscaled', outputCol='features',
                         withMean=False, withStd=True)
    base = LinearSVC(featuresCol='features', labelCol='label',
                     maxIter=100, regParam=0.1, standardization=False)
    ovr = OneVsRest(classifier=base, labelCol='label',
                    featuresCol='features', predictionCol='prediction')
    return SparkPipeline(stages=[indexer, encoder, assembler, scaler, ovr])


def evaluate_spark(model, test_df, label_map):

    #using sklearn metrics on the collected predictions, way easier than fighting
    #MulticlassClassificationEvaluator for kappa/specificity



    from sklearn.metrics import (f1_score, classification_report, accuracy_score,
                                 precision_recall_fscore_support, confusion_matrix,
                                 cohen_kappa_score, roc_auc_score)

    pred = model.transform(test_df).select('label', 'prediction').cache()
    n = pred.count()
    pdf = pred.toPandas()

    inv = {v: k for k, v in label_map.items()}
    pdf['true_label'] = pdf['label'].map(inv)
    pdf['pred_label'] = pdf['prediction'].map(inv)

    y_true = pdf['true_label']
    y_pred = pdf['pred_label']

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_ORDER,
                                                  average='macro', zero_division=0)
    wf1 = f1_score(y_true, y_pred, labels=CLASS_ORDER, average='weighted')

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    specs = []
    for i in range(len(CLASS_ORDER)):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    spec = float(np.mean(specs))

    #AUC: rawPrediction is per-class margins not probabilities, sklearn won't





    auc = np.nan
    try:
        sdf = model.transform(test_df).select('label', 'rawPrediction').toPandas()
        scores = np.array([row.toArray() for row in sdf['rawPrediction']])
        y_int = sdf['label'].astype(int).values
        scores = scores - scores.max(axis=1, keepdims=True)
        proba = np.exp(scores)
        proba = proba / proba.sum(axis=1, keepdims=True)
        auc = roc_auc_score(y_int, proba, multi_class='ovr', average='macro',
                            labels=list(range(proba.shape[1])))
    except Exception as e:
        print('  AUC failed:', e)

    print(f'  rows: {n:,}')
    print(f'  acc:         {acc:.4f}')
    print(f'  kappa:       {kappa:.4f}')
    print(f'  precision:   {p:.4f}')
    print(f'  recall:      {r:.4f}')
    print(f'  specificity: {spec:.4f}')
    print(f'  macro-F1:    {f1:.4f}')
    print(f'  AUC:         {auc:.4f}')
    print(f'  weighted-F1: {wf1:.4f}')
    print(classification_report(y_true, y_pred, labels=CLASS_ORDER, digits=3))

    return {'accuracy': acc, 'kappa': kappa,
            'macro_precision': p, 'macro_sensitivity': r,
            'macro_specificity': spec, 'macro_f1': f1, 'auc': auc,
            'weighted_f1': wf1, 'predictions_pdf': pdf}


def part_a_same_data(spark):
    print('\n== PART A: same-data spark vs sklearn ==')
    df, train, val, test, label_map = load_and_split_spark(spark, DATA_PATH)
    pipeline = build_spark_pipeline()

    print('\nfitting...')
    t0 = time.time()
    model = pipeline.fit(train)
    fit_time = time.time() - t0
    print(f'done in {fit_time:.1f}s')

    print('\n--- test set ---')
    metrics = evaluate_spark(model, test, label_map)

    #save preds for the group comparison sheet
    out = metrics['predictions_pdf'][['true_label', 'pred_label']].copy()
    out.columns = ['final_position', 'pred_spark_linearsvc']
    out.to_csv('spark_test_predictions.csv', index=False)

    # paste-ready numbers for comparison.xlsx (HCPI column)
    print('\nfor comparison.xlsx:')
    for k in ['accuracy', 'kappa', 'macro_precision', 'macro_sensitivity',
              'macro_specificity', 'macro_f1', 'auc']:
        print(f'  {k:20s} {metrics[k]:.4f}')

    return metrics, fit_time


def fit_sklearn(X_tr, y_tr):
    pipe = SkPipeline([
        ('prep', ColumnTransformer([
            ('num', SkScaler(), NUM_COLS),
            ('cat', SkOneHotEncoder(handle_unknown='ignore', sparse_output=False), [CAT_COL]),
        ])),
        ('clf', SkLinearSVC(C=1.0, class_weight='balanced',
                            max_iter=2000, random_state=SEED)),
    ])
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    return time.time() - t0


def fit_spark(spark_df):
    pipeline = build_spark_pipeline()
    t0 = time.time()
    pipeline.fit(spark_df)
    return time.time() - t0


def part_b_scaling(spark):
    print('\n== PART B: scaling experiment ==')

    df = pd.read_csv(DATA_PATH)
    mg = df.loc[df['grid'] > 0, 'grid'].max()
    df.loc[df['grid'] == 0, 'grid'] = mg + 1
    train_pdf = df[df['year'] <= 2022].copy()
    print(f'base train set: {len(train_pdf):,} rows')

    feat_cols = NUM_COLS + [CAT_COL]
    rows = []

    for n in SCALING_SIZES:
        print(f'\n>> resample to {n:,}')
        boot = train_pdf[feat_cols + [TARGET]].sample(
            n=n, replace=True, random_state=SEED).reset_index(drop=True)

        # sklearn
        try:
            sk_t = fit_sklearn(boot[feat_cols], boot[TARGET])
            print(f'  sklearn: {sk_t:.1f}s')
        except Exception as e:
            print('  sklearn FAILED:', e)
            sk_t = np.nan

        #spark - need numeric label column and a forced cache before timing
        try:
            label_map = {lab: float(i) for i, lab in enumerate(CLASS_ORDER)}
            b2 = boot.copy()
            b2['label'] = b2[TARGET].map(label_map)
            sdf = spark.createDataFrame(b2)
            sdf = sdf.repartition(spark.sparkContext.defaultParallelism).cache()
            _ = sdf.count()  # materialise so we don't time the lazy eval
            sp_t = fit_spark(sdf)
            print(f'  spark:   {sp_t:.1f}s')
            sdf.unpersist()
        except Exception as e:
            print('  spark FAILED:', e)
            sp_t = np.nan

        rows.append({'n_rows': n, 'sklearn_seconds': sk_t, 'spark_seconds': sp_t})

    timing_df = pd.DataFrame(rows)
    print('\ntimings:')
    print(timing_df.to_string(index=False))
    timing_df.to_csv('scaling_timings.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    v = timing_df.dropna()
    ax.plot(v['n_rows'], v['sklearn_seconds'], marker='o', linewidth=2,
            color='#3498db', markersize=10, label='sklearn LinearSVC')
    ax.plot(v['n_rows'], v['spark_seconds'], marker='s', linewidth=2,
            color='#e74c3c', markersize=10, label='Spark LinearSVC + OvR')
    for _, row in v.iterrows():
        if not np.isnan(row['sklearn_seconds']):
            ax.text(row['n_rows'], row['sklearn_seconds']*1.15,
                    f"{row['sklearn_seconds']:.1f}s", ha='center', fontsize=9, color='#3498db')
        if not np.isnan(row['spark_seconds']):
            ax.text(row['n_rows'], row['spark_seconds']*0.75,
                    f"{row['spark_seconds']:.1f}s", ha='center', fontsize=9, color='#e74c3c')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('training rows (log)')
    ax.set_ylabel('fit time, seconds (log)')
    ax.set_title('sklearn vs Spark training time', loc='left', fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_scaling_curve.png', dpi=180, bbox_inches='tight')
    plt.close()
    print('saved fig_scaling_curve.png')
    return timing_df


def plot_spark_vs_sklearn(metrics, out_path):
    bars = [
        {'name': 'sklearn\nLinearSVC', 'macro_f1': None, 'accuracy': None},
        {'name': 'Spark\nLinearSVC',   'macro_f1': metrics['macro_f1'],
                                       'accuracy': metrics['accuracy']},
    ]

    #try to grab section 4's numbers from its log if it ran in this folder
    if os.path.exists('sklearn_results.txt'):
        try:
            with open('sklearn_results.txt') as f:
                txt = f.read()
            blk = txt.split('SVM (linear) - test set')[1]
            acc_line = [l for l in blk.splitlines() if 'Accuracy' in l][0]
            f1_line  = [l for l in blk.splitlines() if 'Macro-F1' in l][0]
            bars[0]['accuracy'] = float(acc_line.split(':')[1].strip())
            bars[0]['macro_f1'] = float(f1_line.split(':')[1].strip())
            print(f"loaded sklearn from log: acc={bars[0]['accuracy']:.3f} f1={bars[0]['macro_f1']:.3f}")
        except Exception as e:
            print('could not parse sklearn log:', e)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [b['name'] for b in bars]
    macro = [b['macro_f1'] for b in bars]
    acc   = [b['accuracy']  for b in bars]
    x = np.arange(len(names))
    w = 0.36

    if all(v is not None for v in macro):
        ax.bar(x - w/2, macro, w, label='Macro-F1', color='#3498db',
               edgecolor='black', linewidth=0.5)
    if all(v is not None for v in acc):
        ax.bar(x + w/2, acc, w, label='Accuracy', color='#95a5a6',
               edgecolor='black', linewidth=0.5)
    for i, val in enumerate(macro):
        if val is not None:
            ax.text(i - w/2, val + 0.01, f'{val:.3f}', ha='center', fontsize=9)
    for i, val in enumerate(acc):
        if val is not None:
            ax.text(i + w/2, val + 0.01, f'{val:.3f}', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('score')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')
    ax.set_title('sklearn vs Spark LinearSVC (same data)', loc='left', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print('saved', out_path)


##

def main():
    log_file = open(LOG_PATH, 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)

    print('CS5811 section 5 - Spark LinearSVC')

    spark = build_spark_session()
    try:
        metrics, fit_time = part_a_same_data(spark)
        plot_spark_vs_sklearn(metrics, 'fig_spark_method_comparison.png')
        timing_df = part_b_scaling(spark)
    finally:
        spark.stop()

    print('\ndone.')
    log_file.close()


if __name__ == '__main__':
    main()
