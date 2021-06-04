rm(list = ls())
data_file_train = "D:/Miss Tiny/Research/Piano Classification/Classification/Dataset/adult/adult"
result = load_data_with_fold(data_file_train)

##fold structure####
fold1_normalized_result = normalize(result$fold1$full_data) #normalize data
fold1_normalized_result = fold1_normalized_result[,-c(16,32,39,54,60,64,66,109)]
data_name<-c()
Y_position = 7
for (i in 1:(ncol(fold1_normalized_result[c(1:nrow(result$fold1$train_data)),])-1)){
  name = paste("x",i,sep="")
  if (i ==Y_position){
    data_name=c(data_name,"y")
  }
  data_name=c(data_name,name)
}
names(fold1_normalized_result)<-data_name
fold1_train = fold1_normalized_result[c(1:nrow(result$fold1$train_data)),]
fold1_test = fold1_normalized_result[-c(1:nrow(result$fold1$train_data)),]

fold2_normalized_result = normalize(result$fold2$full_data)
fold2_normalized_result = fold2_normalized_result[,-c(16,32,39,54,60,64,66,109)]
names(fold2_normalized_result)<-data_name
fold2_train = fold2_normalized_result[c(1:nrow(result$fold2$train_data)),]
fold2_test = fold2_normalized_result[-c(1:nrow(result$fold2$train_data)),]

fold3_normalized_result = normalize(result$fold3$full_data)
fold3_normalized_result = fold3_normalized_result[,-c(16,32,39,54,60,64,66,109)]
names(fold3_normalized_result)<-data_name
fold3_train = fold3_normalized_result[c(1:nrow(result$fold3$train_data)),]
fold3_test = fold3_normalized_result[-c(1:nrow(result$fold3$train_data)),]

fold4_normalized_result = normalize(result$fold4$full_data)
fold4_normalized_result = fold4_normalized_result[,-c(16,32,39,54,60,64,66,109)]
names(fold4_normalized_result)<-data_name
fold4_train = fold4_normalized_result[c(1:nrow(result$fold4$train_data)),]
fold4_test = fold4_normalized_result[-c(1:nrow(result$fold4$train_data)),]

fold5_normalized_result = normalize(result$fold5$full_data)
fold5_normalized_result = fold5_normalized_result[,-c(16,32,39,54,60,64,66,109)]
names(fold5_normalized_result)<-data_name
fold5_train = fold5_normalized_result[c(1:nrow(result$fold5$train_data)),]
fold5_test = fold5_normalized_result[-c(1:nrow(result$fold5$train_data)),]

formula = as.formula(y ~.)
levels=c(0,1)
Decision_error = DecisonTree_CV(formula = formula,fold1_train=fold1_train,fold1_test,fold2_train,fold2_test,
                                fold3_train,fold3_test,fold4_train,fold4_test,
                                fold5_train,fold5_test,nfold = 5)

Lasso_error = Lasso_or_Ridge_CV(fold1_train=fold1_train,fold1_test,fold2_train,fold2_test,
                                fold3_train,fold3_test,fold4_train,fold4_test,
                                fold5_train,fold5_test,alpha = 1,nfold = 5)

Ridge_error = Lasso_or_Ridge_CV(fold1_train=fold1_train,fold1_test,fold2_train,fold2_test,
                                fold3_train,fold3_test,fold4_train,fold4_test,
                                fold5_train,fold5_test,alpha = 0,nfold = 5)

Gam_error = Gam_CV(formula=formula,
                   fold1_train,fold1_test,fold2_train,fold2_test,
                   fold3_train,fold3_test,fold4_train,fold4_test,
                   fold5_train,fold5_test,
                   nfold = 5)




#//todo: no splined, need check 
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.05, 
               max_depth=6)
Xgboost_error = XgBoost_summary_CV(data=normalized_result,params,nrounds_max = 10000, nfold=5)

#save results
result_error = rbind(Decision_error,Lasso_error,Ridge_error,Gam_error,Xgboost_error)
full_file_path="D:/Miss Tiny/Research/Piano Classification/Classification/baseline_code/adult_baseline.xlsx"
export(full_file_path, result_error, "Sheet1",FALSE)
full_RData_path="D:/Miss Tiny/Research/Piano Classification/Classification/baseline_code/adult_baseline.RData"
save.image(file=full_RData_path)


