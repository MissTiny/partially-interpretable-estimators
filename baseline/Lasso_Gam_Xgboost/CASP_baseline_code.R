rm(list = ls())
data_file_train = "D:/Miss Tiny/Research/Piano Classification/Regression/No_orthogonal/Data/CASP_train.csv"
data_file_test="D:/Miss Tiny/Research/Piano Classification/Regression/No_orthogonal/Data/CASP_test.csv"
full_file_path="D:/Miss Tiny/Research/Piano Classification/Regression/No_orthogonal/baseline_code/CASP_baseline.xlsx"
full_RData_path="D:/Miss Tiny/Research/Piano Classification/Regression/No_orthogonal/baseline_code/CASP.RData"
#result(full_data, train_data, test_data)
result = load_data_without_index(data_file_train,data_file_test)
normalized_result = normalize(result$full_data) #normalize data
names(normalized_result)<-c("x1","x2","x3","x4","x5","x6","x7","x8","x9","y")
Y_position = ncol(normalized_result)

#normalize_X = normalized_result[,1:(ncol(result$full_data)-1)] #Features 
#normalize_y = normalized_result[,ncol(result$full_data)]      #target 
#train_X_orig = normalize_X[c(1:nrow(result$train_data)),] #not splined 
#test_X_orig = normalize_X[-c(1:nrow(result$train_data)),]
#train_y = normalize_y[1:nrow(result$train_data)]
#test_y = normalize_y[-1:-nrow(result$train_data)]
formula = as.formula(y ~ x1+x2+x3+x4+x5+x6+x7+x8+x9)
#smoother_formula = as.formula(y ~s(x1)+s(x2)+s(x3)+s(x4)+s(x5)+s(x6)+s(x7)+s(x8)+s(x9))
##library####



##Decision Tree Baseline ####
Decision_error = DecisonTree_CV(formula = formula,data = 
                                  normalized_result,nfold = 5)

Lasso_error = Lasso_or_Ridge_CV(normalized_result,alpha = 1,nfold = 5)

Ridge_error = Lasso_or_Ridge_CV(normalized_result,alpha = 0,nfold = 5)


#best_mtry=best_mtry_trainning(nfold=5,mtry_range=c(1:10),formula,normalized_result[c(1:nrow(result$train_data)),])
#best_ntree = best_ntree(6,nfold =5,ntree_range = c(250,300),formula,normalized_result[c(1:nrow(result$train_data)),])
#results_tree <- resamples(store_maxtrees)
#results = summary(results_tree)
#RForest_error = RForest(best_mtry = 6,best_ntree = 500,nfold = 5 ,
#                        formula = formula ,
#                        data = normalized_result[c(1:nrow(result$train_data)),],
#                        train_X_orig,train_y,test_X_orig, test_y)
#//todo: need double check

Gam_error = Gam_CV(formula=formula,
                   data=normalized_result,
                   nfold = 5)




#//todo: no splined, need check 
params <- list(booster = "gbtree", objective = "reg:squarederror", eta=0.05, 
               max_depth=6)
Xgboost_error = XgBoost_summary_CV(data=normalized_result,params,nrounds_max = 10000, nfold=5)

#save results
result_error = rbind(Decision_error,Lasso_error,Ridge_error,Gam_error,Xgboost_error)
export(full_file_path, result_error, "Sheet1",FALSE)
save.image(file=full_RData_path)

