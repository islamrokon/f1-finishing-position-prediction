CS5811 Distributed Data Analysis
Author: Md. Rokon Islam Emon
Student ID: 2565915
Submission: 4 May 2026


Files

f1_eda_emon.py        Section 3 EDA figures (driver strength, era
                      constructor, PCA)
f1_svm_sklearn.py     Section 4 SVM machine learning pipeline
f1_svm_spark.py       Section 5 Spark distributed Linear SVC and
                      scaling experiment
f1_final_dataset.csv  Cleaned working dataset (8,072 rows)


Requirements

Python 3.10 or higher.
pandas, numpy, scikit-learn, matplotlib, pyspark.

Install with:

    pip install pandas numpy scikit-learn matplotlib pyspark


How to run

Put all the .py files and f1_final_dataset.csv in the same folder, then
run whichever script you want.

1. EDA figures (about 3 seconds):

       python f1_eda_emon.py

   Produces fig_1_driver_strength.png, fig_2_era_constructor.png,
   and fig_3_pca.png.

2. Section 4 SVM pipeline (about 25 minutes for the full grid search):

       python f1_svm_sklearn.py

   Produces sklearn_results.txt plus the confusion matrix, permutation
   importance, and era-stratified macro-F1 figures.

3. Section 5 Spark implementation and scaling experiment (about 5 minutes
   for Part A, another 5 for Part B):

       python f1_svm_spark.py

   Produces spark_results.txt plus the Spark vs sklearn comparison and
   the scaling curve figures.


Notes

The scripts were developed in Google Colab and tested locally on Python
3.10 with PySpark 4.0.2.

If you are running locally, remove the Colab-specific upload block at
the top of each script (the one that does `from google.colab import
files`). The scripts expect f1_final_dataset.csv to already be in the
working directory.

Original raw data is on Kaggle:
https://www.kaggle.com/datasets/jtrotman/formula-1-race-data


Figures

The report has 10 figures. All are reproducible from the scripts here
except Figures 2 and 3, which come from a teammate's script.

   1   Class distribution                       (Section 1)
   2   Grid position win rate                   (Section 3, teammate)
   3   Q1 gap to pole boxplot                   (Section 3, teammate)
   4   Driver strength three-panel              (Section 3)
   5   Era-stratified constructor               (Section 3)
   6   PCA scree and biplot                     (Section 3)
   7   Permutation importance                   (Section 4)
   8   Era-stratified within-era macro-F1       (Section 4)
   9   Spark vs sklearn comparison              (Section 5)
   10  Scaling curve                            (Section 5)

Figures 1, 4, 5, 6 come from f1_eda_emon.py.
Figures 7, 8 come from f1_svm_sklearn.py.
Figures 9, 10 come from f1_svm_spark.py.

