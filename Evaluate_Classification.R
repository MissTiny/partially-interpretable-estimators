
args<-commandArgs()
load_model<-args[6]
load(load_model)

#-------predict with Validation Test dataset--------------####
pred = Piano_predict(spl_Validation_test_X[[fold]], Validation_test_X_orig[[fold]], 
                     validation_test_y[[fold]], alpha=fit$alpha, beta0=fit$beta0, 
                     Tree_list=fit$Trees,stepsize)

val_interpret = pred$auc_lasso/pred$auc
sparsity=sparsity_count(fit$alpha,n_levels,p)
#-------predict with Real Test dataset--------------####
test_pred = Piano_predict(spl_test_X[[fold]], test_X_orig[[fold]], test_y[[fold]], 
                          alpha=fit$alpha, beta0=fit$beta0, 
                          Tree_list=fit$Trees,stepsize)
interpret = test_pred$auc_lasso/test_pred$auc
#-------Result Statement--------------####
errorMat=paste("count",count,"Lambda1", lambda1,"Lambda2",lambda2,"Gamma",gamma,"Iteration",
      fit$best_iter,"Nrounds",nrounds,"auc_train",fit$auc,"auc_lasso_train",
      fit$auc_lasso,"auc_val_test",pred$auc,"auc_val_lasso_test",pred$auc_lasso,
      "val_interpret",val_interpret,"auc_test",test_pred$auc,"auc_lasso_test",
      test_pred$auc_lasso,"interpret",interpret,
      "stepsize",stepsize,"tree_nrounds",tree_nrounds,"sparsity",sparsity,
      sep="")

print(errorMat)

