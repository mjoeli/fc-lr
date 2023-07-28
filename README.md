# FeatureCloud Linear Regression App

## Description
A Linear Regression FeautureCloud app, allowing a federated computation of the linear regression algorithm.

## Input
- train.csv containing the local training data (columns: features; rows: samples)
- test.csv containing the local test data

## Output
- output.csv containing the Intercept and Coefficients from the model 

## Config
Use the config file to customize your training. Just upload it together with your training data as `config.yml`
```
fc_linear_regression
 train: train.csv
 test: centralised_test.csv
 target_value: target_variable
 sep: ','
 max_iter: 1
 exact: True (If True, the aggregation step weights the parameters by the exact datasize) 
 output: output.csv
```

