# category_classifier


#local run: 
mlflow run . --no-conda --experiment-id 3681976

#remote run 
mlflow run . --no-conda --experiment-id 3681976 -b databricks -c cluster.json -P remote=1
