import pandas as p
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.ml.linalg import Vectors, DenseVector, SparseVector
from pyspark.sql.types import IntegerType,StringType,DoubleType
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


#DataFrame (use basic pyspark.sql)
# 1. Load the "Bank-marketing" dataset into a DataFrame, check the schema
# and dislay the first 20 rows

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df0 = spark.read.csv("/Users/hoangphuonganhnguyen/Downloads/BankMarketing.csv",header=True,inferSchema=True)

print("df0 schema is: ")
df0.printSchema()

df0.show(20)

# 2. groupBy education
education = df0.groupBy("education").count()
education.show(20)
# 3. col "age", most targeted range
age = df0.groupBy("age").count()
age_sort = age.orderBy("count", ascending=False)
age_sort.show(20)

#4 model report of selected cols
selected_cols = df0.select("age",'education','marital',"duration")

numerical_info = selected_cols.describe("age","duration","marital","education")
numerical_info.show()

marital_info = selected_cols.groupBy("marital").count().show()
education_info = selected_cols.groupBy("education").count().show()

#5. automatically identify the features and filter them based on their types (numerical or categorical)
numerical_col = [field.name for field in df0.schema.fields
                 if isinstance(field.dataType, (IntegerType, DoubleType))]
categorical_col = [field.name for field in df0.schema.fields
                   if isinstance(field.dataType,(StringType))]
print("Numerical columns: ")
print(numerical_col)
print("Categorical columns: ")
print(categorical_col)


#Feature transformation (use pyspark.ml.feature)
marital_Index = StringIndexer(inputCol="marital", outputCol="maritalIndex")
df1 = marital_Index.fit(df0).transform(df0)
print("df1 marital index schema is: ")
df1.printSchema()

marital_vector = OneHotEncoder(inputCol="maritalIndex", outputCol="maritalVector")
df2 = marital_vector.fit(df1).transform(df1)
print("df2 marital index schema is: ")
df2.printSchema()
print("Basic info of the vector: ")
df2.select("maritalVector").show(5, truncate=False)
#define function to measure the vector size (features' sizes)


#3. For education feature
education_Index = StringIndexer(inputCol="education", outputCol="educationIndex")
df3 = education_Index.fit(df2).transform(df2)
print("vectors sizes of education col are: ")
df3.select("educationIndex").printSchema()


education_vector = OneHotEncoder(inputCol="educationIndex", outputCol="educationVector")
df4 = education_vector.fit(df3).transform(df3)
df4.select("educationIndex").show(5, truncate=False)

#4. Vector assembler
assembler = VectorAssembler(
    inputCols=["age", "maritalVector",'educationVector'],
    outputCol="vector",
)
df5 = assembler.transform(df4)
print("SHOW df5: ")
df5.select("age","maritalVector","educationVector","vector").show(truncate=False)
df5.printSchema

#5.
pipeline = Pipeline(stages=[marital_Index,education_Index,marital_vector,education_vector,assembler])
pipeline_model = pipeline.fit(df0)
df5 = pipeline_model.transform(df0)

df5.select("age","marital","education","vector").show(truncate=False)

#6. Y to label, using pipeline
print("Step 6: ")
label_indexer = StringIndexer(inputCol="y", outputCol="label")
df6 = label_indexer.fit(df5).transform(df5)
df6.select("y","label").show(truncate= False)

#7. Features col
print("Step 7: ")
numerical_features = [
    "age",
    "duration",
    "campaign",
    "maritalVector",
    "educationVector"
]
assembler7 = VectorAssembler(inputCols= numerical_features,outputCol="features")
df_features = assembler7.transform(df6)
df_features.select("age","features").show(truncate=False)

#Statistical Classification
print("STATISTICAL CLASSIFICATION")
#1. Logistic regression -  add on appropriate hyperparams
print("step 1: ")
logistic_regression = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.8)

pipeline = Pipeline(stages=[logistic_regression])
pipeline_model = pipeline.fit(df_features)
df_predictions = pipeline_model.transform(df_features)
df_predictions.filter(df_predictions["prediction"] == 0).select("label","features","prediction").show(truncate=False)
df_predictions.filter(df_predictions["prediction"] == 1).select("label","features","prediction").show(truncate=False)

# 2. Train and test the model
pipeline_model.write().overwrite().save("/Users/hoangphuonganhnguyen/Desktop/intern/TP/[PRJ] Bank marketing/logistic_regression_pipeline")
print("Pipeline model saved successfully!")

#loaded_pipeline_model = PipelineModel.load("/Users/hoangphuonganhnguyen/Desktop/intern/TP/[PRJ] Bank marketing/logistic_regression_pipeline")
#print("Pipeline model loaded successfully!")

#3. Apply the model to the dataset to generate predictions
print("step 3: ")
df_predictions = pipeline_model.transform(df_features)
df_predictions.filter(df_predictions["prediction"] == 0).select("label","features","prediction").show(truncate=False)
df_predictions.filter(df_predictions["prediction"] == 1).select("label","features","prediction").show(truncate=False)

#4 Extracting the Logistic regression (classifier) from the pipeline
print("Pipeline model classifier: ")
print(type(pipeline_model.stages[-1])) #<class 'pyspark.ml.classification.LogisticRegressionModel'>
logistic_regression = LogisticRegression(featuresCol="features",
                                         labelCol="label",
                                         maxIter=10)
logistic_model = logistic_regression.fit(df_features)

#display the loss history
loss_history = logistic_model.summary.objectiveHistory
print("Loss History: ")
print(loss_history)

#Check convergence
print("Check CONVERGENCE: ")
if len(loss_history) > 1:
    print("Loss changed between iterations are: ")
    for i in range(1, len(loss_history)):
        print(f"Iteration {i}: Loss difference = {loss_history[i]-loss_history[i-1]}")
    else:
        print("Not enough iterations to check convergence")

#Performance and Evaluation
#1. Print AUC and assess the evaluation of of model classification
print("Step 1: ")
#Initialize the evaluator
binary_evaluator = BinaryClassificationEvaluator(labelCol="label",rawPredictionCol="rawPrediction",metricName="areaUnderROC")

#Calculate the AUC
auc = binary_evaluator.evaluate(df_predictions)
print(f"AUC: {auc}")

#Visualize ROC
roc = logistic_model.summary.roc.toPandas()
plt.figure(figsize=(8, 6))
plt.plot(roc['FPR'], roc['TPR'], label="ROC Curve", color="blue")
plt.plot([0, 1], [0, 1], "r--", label="Random Classifier")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc="lower right")
plt.grid(True)
#plt.show()

#2. F-measures
print("Retrieve F-Measure for different threshold: ")
binary_summary = logistic_model.summary
#extract data
precision = binary_summary.precisionByThreshold
recall = binary_summary.recallByThreshold
f1_score = binary_summary.fMeasureByThreshold
#convert to pandas df
f1_df = f1_score.toPandas()
precision_df = precision.toPandas()
recall_df = recall.toPandas()

print("F1 DataFrame:")
print(f1_df.head())
print("Precision DataFrame:")
print(precision_df.head())
print("Recall DataFrame:")
print(recall_df.head())

#merge
metrics_df = f1_df.merge(precision_df, on="threshold").merge(recall_df, on="threshold")
metrics_df.columns = ["Threshold", "F1-Score", "Precision", "Recall"]

#find the best threshold
best_row = metrics_df.loc[metrics_df["F1-Score"].idxmax()]
best_threshold = best_row["Threshold"]
best_f1 = best_row["F1-Score"]

print(f"Best F1-Score: {best_f1}")
print(f"Best threshold: {best_threshold}")

#apply threshold to the model
logistic_model.setThreshold(best_threshold)
print(f"Best classifier threshold set to: {best_threshold}")
logistic_model.write().overwrite().save("optimized_logistic_model")

#3. PLot the F-measure evaluation
print("F-Measure PLot: ")
thresholds = metrics_df["Threshold"]
f1_scores = metrics_df["F1-Score"]

plt.figure(figsize=(10,6))
plt.plot(thresholds, f1_scores, label="F1-Score", color="blue", linewidth=2)
plt.xlabel("Threshold", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.title("F1-Score vs Threshold", fontsize=14)
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f"Best Threshold: {best_threshold:.4f}")
plt.legend()
plt.grid()
plt.show()

#4, Train model on 5-folds approach
print("Step 4: ")
paramGrid = ParamGridBuilder()\
    .addGrid(logistic_model.regParam, [0.1, 0.3, 0.5]) \
        .addGrid(logistic_model.elasticNetParam, [0.0,0.5,1.0])\
            .build()

#set up binary classification
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC", rawPredictionCol="rawPrediction")
CrossValidator = CrossValidator(estimator=logistic_regression,
                                estimatorParamMaps=paramGrid,
                                evaluator=evaluator,
                                numFolds= 5)
#Perform cross-validation
cvModel = CrossValidator.fit(df_features)

#Get and evaluate the best model
bestModel = cvModel.bestModel
auc = evaluator.evaluate(bestModel.transform(df_features))
print(f"Best model after cross-validation is: {auc}")

#Best model after cross-validation is: 0.8375070526053597
#the difference between the number from step 1 and the number from step 4

