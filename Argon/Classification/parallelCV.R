rm(list=ls())
###Loading packges#######
library(splines)
library(MASS)
library(fastDummies)
library(pROC)
library(xgboost)
#########################################################
load("functions.RData")
args<-commandArgs()
load_rdata_filename<-args[6]
load(load_rdata_filename)

fold<-as.numeric(args[7])
count<-args[8]
lambda1 <-as.numeric(args[9])
lambda2<-as.numeric(args[10])
eta<-as.numeric(args[11])
iter <-as.numeric(args[12])
nrounds <-as.numeric(args[13])
name<-args[14]
stepsize<-as.numeric(args[15])
tree_nrounds<-as.numeric(args[16])
# fold=1
# count = 0
# lambda1=0.01
# lambda2=0.1
# iter = 20
# nrounds=20
# eta=0.05
# name="XXX"
# stepsize=1
# tree_nrounds = 200
parameter_set = paste("count:",count,"lambda1:",lambda1,"lambda2:",lambda2,"eta:",eta,"iter:",iter,"nrounds:",nrounds,"name:",name,"stepsize:",stepsize,"tree_nrounds",tree_nrounds,sep=" ")
print(parameter_set)
#1. stepsize
n = length(validation_train_y[[fold]])
d_j = c()
for(J in 1:p){
  E = eigen(crossprod(as.matrix(spl_Validation_train_X[[fold]][,(((J-1)*5)+1):(J*5)]),as.matrix(spl_Validation_train_X[[fold]][,(((J-1)*5)+1):(J*5)])))
  d_j = c(d_j,(1/(4*n))*(max(E$values)))
}

d_c = c()
n_dummy = c()
d_c_count = 0 #number of columns for categorical vars after dummy transfer
for(C in 1:q){
  #n_levels = length(levels(X_cat[,C]))-1 - prewriter in data load
  n_dummy = c(n_dummy,n_levels[[C]])
  d_c_count = d_c_count + n_levels[[C]]
  E = eigen(crossprod(as.matrix(spl_Validation_train_X[[fold]][,(d_c_count-n_levels[[C]]):d_c_count]),as.matrix(spl_Validation_train_X[[fold]][,(d_c_count-n_levels[[C]]):d_c_count])))
  d_c = c(d_c,(1/(4*n))*(max(E$values)))
}

d_0 = 1/(4*n)

#2.parameters to tune



k=5
#######################
##Update
set.seed(100)
fit = Piano_fit(spl_Validation_train_X[[fold]],Validation_train_X_orig[[fold]],validation_train_y[[fold]],iter,lambda1,lambda2,eta,nrounds,d_j,d_c,d_0,n_dummy,nc_spline,stepsize,tree_nrounds)
pred = Piano_predict(spl_Validation_test_X[[fold]], Validation_test_X_orig[[fold]], validation_test_y[[fold]], alpha=fit$alpha, beta0=fit$beta0, Tree_list=fit$Trees,stepsize)
val_interpret = pred$auc_lasso/pred$auc
sparsity=sparsity_count(fit$alpha,n_levels,p)
test_pred = Piano_predict(spl_test_X[[fold]], test_X_orig[[fold]], test_y[[fold]], alpha=fit$alpha, beta0=fit$beta0, Tree_list=fit$Trees,stepsize)
interpret = test_pred$auc_lasso/test_pred$auc
errorMat = c(count,lambda1,lambda2,fit$best_iter,nrounds,stepsize,tree_nrounds,sparsity,fit$auc, fit$auc_lasso, 
              pred$auc,pred$auc_lasso,val_interpret,test_pred$auc,test_pred$auc_lasso,interpret)

#names(errorMat) = c( "Lambda1", "Lambda2","Gamma","Iteration","Nrounds"
#                   ,"auc_train","auc_lasso_train","auc_test",
#                   "auc_lasso_test","loss_train", "loss_lasso_train",
#                   "loss_test","loss_lasso_test")
print(errorMat)


filename=paste(name,"_",fold,"_",count,".RData",sep="")
save.image(file=filename)
#################
















