# category_classifier


#local run: 
mlflow run . --no-conda --experiment-id 4979191

#remote run 
mlflow run https://github.com/seif-ca/category_classifier.git --experiment-id 4979191 -b databricks -c cluster.json -P remote=1