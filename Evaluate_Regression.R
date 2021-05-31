library(xgboost)
library(MASS)
library(splines)
library(gglasso)

args<-commandArgs()
#load PIE algorithm functions####
load_model<-args[6]
load(load_model)

#----------The sparsity of the model----------####
best_iter = fit[[6]]
dim(fit[[1]][,c(1:best_iter)])
Betas = rowSums(fit[[1]][,c(1:best_iter),drop=FALSE])
sparsity = sparsity_count(Betas,lasso_group)

#-------predict with Validation Test dataset--------------####
pred = GAMtree_predict(X=spl_Validation_test_X[[fold]], 
                       X_orig = Validation_test_X_orig[[fold]],
                       Betas, 
                       Trees = fit[[2]][1:best_iter],stepsize)
val_rrmse_test =my_error(pred[[1]],validation_test_y[[fold]])
val_rrmse_Gam = my_error(pred[[2]],validation_test_y[[fold]])
val_int = (1-val_rrmse_Gam)/(1-val_rrmse_test)

#-------predict with Real Test dataset--------------####
test_pred = GAMtree_predict(X=spl_test_X[[fold]], 
                            X_orig = test_X_orig[[fold]],Betas, 
                            Trees = fit[[2]][1:best_iter],stepsize)
test_rrmse=my_error(test_pred[[1]],test_y[[fold]])
test_Gam_rrmse = my_error(test_pred[[2]],test_y[[fold]])
Interpretability = (1-test_Gam_rrmse)/(1-test_rrmse)


#-------Result Statement--------------####
errorMat = paste("lambda1:",lambda1,
                 "lambda2:",lambda2,
                 "iter:",iter,
                 "nrounds:",nrounds,
                 "eta:",eta,
                 "stepsize:",stepsize,
                 "best_iter",best_iter,
                 "sparsity",sparsity,
                 "RPE_validation_train",fit[[3]][2,best_iter],
                 "RPE_validation_test",val_rrmse_test,
                 "RPE_Validation_GAM_only",val_rrmse_Gam,
                 "Val_Interpretability",val_int,
                 "RPE_test",test_rrmse,
                 "RPE_test_GAM_only",test_Gam_rrmse,
                 "Interpretability",Interpretability,sep=" ")
print(errorMat)