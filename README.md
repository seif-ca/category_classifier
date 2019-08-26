# category_classifier


#local run: 
mlflow run . --no-conda --experiment-id 3833022

#remote run 
mlflow run https://github.com/seif-ca/category_classifier.git --experiment-id 3833022 -b databricks -c cluster.json -P remote=1