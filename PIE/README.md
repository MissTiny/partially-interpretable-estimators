# Partially Interpretable Estimators

This repository is the official implementation of [Partially Interpretable Estimators](https://arxiv.org/abs/2030.12345). 


## Requirements

Our PIE model is constructed based on R language. However, some baselines presented in the paper is done by Python. Navigate to **baseline** folder if needed.

To install requirements for PIE model:

```setup
Rscript requirements.R
```
## Regression
Folder **Regression** contains the training process of regression task presented in the paper.
In the folder, "parkinsons_load.R" is provided as an example. To get the RData file of the dataset, run "parkinsons_load.R". 
For other datasets, please see the pretrained model, or create similar data loading R file.

Following are the parameters setting and training command:

```train
##load functions
Rscript load_functions.R
echo "function load finished"

#Parameter Tuning set
count=0
Nrounds=(200)
Lambda1=(0.0001 0.0005 0.001 0.005 0.01 0.1 1 10 100)
Lambda2=(1)
Eta=(0.05)
Iteration=(500)
Stepsize=(0.01 0.05 0.1 0.25 0.5 0.75 1)


for i in {1,2,3,4,5}
do
for nrounds in ${Nrounds[@]}
do
for lambda1 in ${Lambda1[@]}
do
for lambda2 in ${Lambda2[@]}
do
for eta in ${Eta[@]}
do
for iteration in ${Iteration[@]}
do
for stepsize in ${Stepsize[@]}
do

((count+=1))
##do parallel functions
name="output_${count}.txt"
##Use CASP Dataset as an example
Rscript sparsity_parallelCV.R load_CASP.RData $i $count $lambda1 $lambda2 $stepsize $iteration $eta $nrounds CASP > $name &

done
done
done
done
done
done
wait
done
wait

## data_name fold
echo "start summing all outcomes finished"
Rscript sparsity_parallelFinal.R CASP $count CASP_sparsity.csv CASP_sparsityCV.csv
```

## Classification
Folder **Classification** contains the training processof classification task presented in the paper.
In the folder, "adult_load.R" is provided as an example. To get the RData file of the dataset, run "adult_load.R". 
For other datasets, please see the pretrained model, or create similar data loading R file.

Following are the parameters setting and training command:
```
##load functions
Rscript load_parallel_functions.R
echo "function load finished"

count=0
Nrounds=(200)
Lambda1=(0.0001 0.0005 0.001 0.005 0.01 0.1)
Lambda2=(0.0001 0.001 0.01 0.1 1)
Eta=(0.05)
Iteration=(500)
Stepsize=(1)
Tree_Nrounds=(200)

for i in {1,2,3,4,5}
do
for nrounds in ${Nrounds[@]}
do
for lambda2 in ${Lambda2[@]}
do
for lambda1 in ${Lambda1[@]}
do
for eta in ${Eta[@]}
do
for iteration in ${Iteration[@]}
do
for stepsize in ${Stepsize[@]}
do
for tree_nrounds in ${Tree_Nrounds[@]}
do
((count+=1))
##do parallel functions
name="output_${count}.txt"
Rscript parallelCV.R load_adult.RData $i $count $lambda1 $lambda2 $eta $iteration $nrounds adult $stepsize $tree_nrounds > $name &
done
done
done
done
done
done
done
wait
done
wait
## data_name fold
echo "start summing all outcomes finished"
Rscript parallelFinal.R adult $count adult_errorMat.csv
```
## Result
From the command above, you can easily see the parameter tuning range.
Since we applied 5-fold cross-validation on each dataset, the final result reported in the paper is the average of performance for each fold and the standard deviation of the five result is provided as error bars as well. Thus, for each fold, we select model with the best performance on the validation data and then predict with the test data.

The selection standard for PIE result is high validation RPE and a relative good interpretability.
For Sparse PIE result, the selection standard is similar to the PIE selection standard, plus a limitation of less than or equal to 8 features.

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
