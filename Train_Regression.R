library(xgboost)
library(MASS)
library(splines)
library(gglasso)

args<-commandArgs()
#load PIE algorithm functions
load("functions.RData")


#load data RData file
load_rdata_filename<-args[6]
load(load_rdata_filename)


lambda1 <-as.numeric(args[7])
lambda2<-as.numeric(args[8])
stepsize<-as.numeric(args[9])
iter <-as.numeric(args[10])
eta <-as.numeric(args[11])
nrounds <-as.numeric(args[12])
fold <-as.numeric(args[13])
parameter_set = paste("lambda1:",lambda1,"lambda2:",lambda2,
                      "iter:",iter,"nrounds:",nrounds,"eta:",eta,
                      "stepsize:",stepsize,"fold",fold,sep=" ")
print(parameter_set)

print("------------------PIE model starts training-----------------")
fit = GAMtree_fit(X=spl_Validation_train_X[[fold]],
                  y=validation_train_y[[fold]],
                  lasso_group,
                  X_orig=Validation_train_X_orig[[fold]],
                  lambda1,lambda2, iter, stepsize,eta, nrounds)
print("-----------------PIE model finishes training----------------")
save.image(file="train.RData")