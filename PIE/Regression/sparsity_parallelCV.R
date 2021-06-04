library(xgboost)
library(MASS)
library(splines)
library(gglasso)
load("functions.RData")

args<-commandArgs()
load_rdata_filename<-args[6]
load(load_rdata_filename)

fold<-as.numeric(args[7])
count<-args[8]
lambda1 <-as.numeric(args[9])
lambda2<-as.numeric(args[10])
stepsize<-as.numeric(args[11])
iter <-as.numeric(args[12])
eta <-as.numeric(args[13])
nrounds <-as.numeric(args[14])
name<-args[15]
parameter_set = paste("count:",count,"lambda1:",lambda1,"lambda2:",lambda2,
                      "iter:",iter,"nrounds:",nrounds,"name:",name,"eta:",eta,
                      "stepsize:",stepsize,sep=" ")
print(parameter_set)
k=5
#######################
##Update
set.seed(100)

fit = GAMtree_fit(X=spl_Validation_train_X[[fold]],y=validation_train_y[[fold]],lasso_group,X_orig=Validation_train_X_orig[[fold]],lambda1,lambda2, iter, stepsize,eta, nrounds)
best_iter = fit[[6]]
dim(fit[[1]][,c(1:best_iter)])
Betas = rowSums(fit[[1]][,c(1:best_iter),drop=FALSE])
sparsity = sparsity_count(Betas,lasso_group)
pred = GAMtree_predict(X=spl_Validation_test_X[[fold]], X_orig = Validation_test_X_orig[[fold]],Betas, Trees = fit[[2]][1:best_iter],stepsize)
val_rrmse_test =my_error(pred[[1]],validation_test_y[[fold]])
val_rrmse_Gam = my_error(pred[[2]],validation_test_y[[fold]])
val_int = (1-val_rrmse_Gam)/(1-val_rrmse_test)
test_pred = GAMtree_predict(X=spl_test_X[[fold]], X_orig = test_X_orig[[fold]],Betas, Trees = fit[[2]][1:best_iter],stepsize)
print("test_pred finished")
test_rrmse=my_error(test_pred[[1]],test_y[[fold]])
test_Gam_rrmse = my_error(test_pred[[2]],test_y[[fold]])
Interpretability = (1-test_Gam_rrmse)/(1-test_rrmse)
errorMat = c(count,fold,best_iter,lambda1,lambda2,stepsize,eta,nrounds,sparsity,fit[[3]][2,best_iter],
             val_rrmse_test,val_rrmse_Gam,val_int,test_rrmse,
            test_Gam_rrmse,Interpretability)
print(errorMat)


filename=paste(name,"_",fold,"_",count,".RData",sep="")
save.image(file=filename)
#################
