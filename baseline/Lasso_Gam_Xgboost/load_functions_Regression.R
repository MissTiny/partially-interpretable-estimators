#library####
library(splines)
library(glmnet)
library(MLmetrics)
library(rpart)
library(mgcv)
library(caret)
library(xgboost)
library(openxlsx)
##Load Data####
load_data_with_index<-function(data_file_train,data_file_test){
  train_data <- read.csv(data_file_train)[,-1]
  test_data <- read.csv(data_file_test)[,-1]
  full_data <- rbind(train_data, test_data)
  result <- list("full_data" = full_data,"train_data"=train_data,"test_data" = test_data)
  return(result)
}
load_data_with_index_onefold<-function(data_file){
  data <- read.csv(data_file)[,-1]
  result <- list("full_data" = data)
  return(result)
}

load_data_without_index<-function(data_file_train,data_file_test){
  train_data <- read.csv(data_file_train)
  test_data <- read.csv(data_file_test)
  full_data <- rbind(train_data, test_data)
  result <- list("full_data" = full_data,"train_data"=train_data,"test_data" = test_data)
  return(result)
}
##Normalization####
normalize<-function(data){
  for(i in 1:ncol(data)){
    data[,i] = (data[,i]-min(data[,i]))/(max(data[,i])-min(data[,i]))
  }
  return (data)
}

##measure ####
my_error<-function(pred, true){#mse/variance, which is 1-R^2
  sum((true-pred)^2)/sum((true-mean(true))^2)
}

DecisonTree_CV<-function(formula,data,nfold){
  set.seed(100)
  folds <- sample(rep(1:nfold, length.out = nrow(data)))#all data
  fold_errorMat<-data.frame(matrix(NA,ncol=8,nrow=nfold+1))
  names(fold_errorMat)<-c("method_name","fold_num","rrmse_train","rrmse_test", "mse_train","mse_test","parameter","parameter range")
  fold_num = 0
  for(fold in 1:nfold){
    #data split####
    #cross validation
    fold_num =fold_num +1
    print(paste("fold_num:",fold_num,sep=""))
    idx <- which(folds==fold)
    train_orig = data[-idx,]
    test_orig  = data[idx,]
    
    train_X_orig = train_orig[,-Y_position]
    test_X_orig = test_orig[,-Y_position]
  
    #Y
    train_y =train_orig[,Y_position]
    test_y = test_orig[,Y_position]
    errorMat = DecisionTree(formula,train_orig,train_X_orig,train_y,test_X_orig,test_y,nfold)
    fold_errorMat[fold_num,] = c("Decision_Tree",fold_num,errorMat)
  }
  fold_errorMat[fold_num+1,] = c("Decision_Tree","average",colSums(fold_errorMat[3],na.rm=TRUE)/nfold,colSums(fold_errorMat[4],na.rm=TRUE)/nfold,colSums(fold_errorMat[5],na.rm=TRUE)/nfold,colSums(fold_errorMat[6],na.rm=TRUE)/nfold,"","")
  fold_errorMat
  }
##baseline Functions####
DecisionTree<-function(formula,data,train_X_orig,train_y,test_X_orig,test_y,nfold){
  #random
  set.seed(100)
  ##train####
  #xval number of corss validations, anova = regression
  train_tree<-rpart(formula,
                    data = data.frame(data),method = "anova",
                    control=rpart.control(xval=nfold))
  
  #pruned tree
  bestcp <- train_tree$cptable[which.min(train_tree$cptable[,"xerror"]),"CP"]
  print(bestcp)
  train_pruned_tree <-prune(train_tree,bestcp)
  train_predict = predict(train_pruned_tree)
  
  ##train error rate
  rrMSE_train = my_error(train_predict,train_y)
  MSE_train = MSE(train_predict,train_y)
  
  ##Test####
  test_predict<-predict(train_pruned_tree,newdata=test_X_orig)
  
  #test error rate
  rrMSE_test = my_error(test_predict,test_y)
  MSE_test = MSE(test_predict,test_y)
  
  errorMat<-c(rrMSE_train,rrMSE_test,MSE_train,MSE_test)
  #errorMat<-c(errorMat,paste("cp=",bestcp,sep=""),"package auto find")
  #names(errorMat) = c("rrmse_train","rrmse_test", "mse_train","mse_test","parameter","parameter range")
  errorMat_result<-list("rrmse_train"=errorMat[1],"rrmse_test"=errorMat[2],
                        "mse_train"=errorMat[3],"mse_test"=errorMat[4],
                        "parameter"=paste("cp=",bestcp,sep=""),"parameter range"="package auto find")
  #errorMat
  errorMat_result
}

#####
Lasso_or_Ridge_CV<-function(data,alpha,nfold){
  set.seed(100)
  folds <- sample(rep(1:nfold, length.out = nrow(data)))#all data
  fold_errorMat<-data.frame(matrix(NA,ncol=8,nrow=nfold+1))
  names(fold_errorMat)<-c("method_name","fold_num","rrmse_train","rrmse_test", "mse_train","mse_test","parameter","parameter range")
  fold_num = 0
  method="Lasso"
  if(alpha==0){
    method="Ridge"
  }
  for(fold in 1:nfold){
    #data split####
    #cross validation
    fold_num =fold_num +1
    print(paste("fold_num:",fold_num,sep=""))
    idx <- which(folds==fold)
    train_orig = data[-idx,]
    test_orig  = data[idx,]
    
    train_X_orig = train_orig[,-Y_position]
    test_X_orig = test_orig[,-Y_position]
    
    #Y
    train_y =train_orig[,Y_position]
    test_y = test_orig[,Y_position]
    errorMat = Lasso_or_Ridge(train_X_orig,train_y,alpha,test_X_orig,test_y,nfold)
    fold_errorMat[fold_num,] = c(method,fold_num,errorMat)
  }
  
  fold_errorMat[fold_num+1,] = c(method,"average",colSums(fold_errorMat[3],na.rm=TRUE)/nfold,colSums(fold_errorMat[4],na.rm=TRUE)/nfold,colSums(fold_errorMat[5],na.rm=TRUE)/nfold,colSums(fold_errorMat[6],na.rm=TRUE)/nfold,"","")
  fold_errorMat  
}
Lasso_or_Ridge<-function(train_X_orig,train_y,alpha,test_X_orig,test_y,nfold){
  #train model
  set.seed(100)
  lambda_seq <- 10^seq(2, -10, by = -.1)
  #alpha = 1 for lasso
  cv_output<-cv.glmnet(data.matrix(train_X_orig),train_y
                       ,alpha = alpha, lambda = lambda_seq,nfolds=nfold)
  best_lam<-cv_output$lambda.min
  print(best_lam)
  lasso_or_ridge_best <- glmnet(data.matrix(train_X_orig),train_y, alpha = alpha, lambda = best_lam)
  
  #train
  train_predict <- predict(lasso_or_ridge_best, s = best_lam, newx =data.matrix(train_X_orig))
  rrMSE_train = my_error(train_predict,train_y)
  MSE_train = MSE(train_predict,train_y)
  
  #test
  test_predict <- predict(lasso_or_ridge_best, s = best_lam, newx = data.matrix(test_X_orig))
  rrMSE_test = my_error(test_predict,test_y)
  MSE_test = MSE(test_predict,test_y)
  
  errorMat<-c(rrMSE_train,rrMSE_test,MSE_train,MSE_test)
  #errorMat<-c(errorMat,paste("lambda = ",best_lam,sep=""),"lambda = 10^seq(2, -10, by = -.1)")
  errorMat_result<-list("rrmse_train"=errorMat[1],"rrmse_test"=errorMat[2],
                        "mse_train"=errorMat[3],"mse_test"=errorMat[4],
                        "parameter"=paste("lambda = ",best_lam,sep=""),"parameter range"="lambda = 10^seq(2, -10, by = -.1)")
  #names(errorMat) = c("rrmse_train","rrmse_test", "mse_train","mse_test","parameter","parameter range")
  #print(errorMat)
  #errorMat
  errorMat_result
}

#Rforest####
#best_mtry_trainning<-function(nfold,mtry_range,formula,data){
#  set.seed(100)
#  #CV
#  trControl <-trainControl(method = "cv", number = nfold,search="grid") #evaluate the model with grid search of 5 fold

#  tuneGrid<- expand.grid(.mtry=mtry_range)
#  rf_mtry<- train(formula,data = data.frame(data),
#                  method="rf",metric="RMSE",tuneGrid = tuneGrid, trControl=trControl,importance=TRUE) #Best model is chosen by RMSE
#  best_mtry = rf_mtry$bestTune$mtry
#  return(best_mtry)
#}
#best_ntree<-function(best_mtry,nfold,ntree_range,formula,data){
#  set.seed(100)
#  store_maxtrees <- list()
#  trControl <-trainControl(method = "cv", number = nfold,search="grid") #evaluate the model with grid search of 5 fold
#  tuneGrid<- expand.grid(.mtry=best_mtry)
#  for (ntree in ntree_range){
#    rf_ntree<- train(formula,data = data.frame(data),
#                     method="rf",metric="RMSE",tuneGrid = tuneGrid, trControl=trControl,importance=TRUE,ntree=ntree)
#    print(ntree)
#    key <- toString(ntree)
#    store_maxtrees[[key]] <- rf_ntree
#  }
#  store_maxtrees

  #Find the best ntree by looking at the table
#}
#RForest<-function(best_mtry,best_ntree,nfold,formula,data,train_X_orig,train_y,test_X_orig, test_y){
  #random seed
#  set.seed(100)
  
  #CV
#  trControl <-trainControl(method = "cv", number = nfold,search="grid") #evaluate the model with grid search of 5 fold
  
  #random Forest
  
  ##train
  
  #search best parameter
#  tuneGrid<- expand.grid(.mtry=best_mtry)
#  fit_rf <- train(formula,
#                  data.frame(data),
#                  method = "rf",
#                  metric="RMSE",
#                  tuneGrid = tuneGrid,
#                  trControl = trControl,
#                  importance = TRUE,
#                  ntree = best_ntree)
#  ##train error rate
#  train_predict <-predict(fit_rf, train_X_orig)
#  rrMSE_train = my_error(train_predict,train_y)
#  MSE_train = MSE(train_predict,train_y)
#  
  ##Test
#  test_predict<-predict(fit_rf,test_X_orig)
  
  #test error rate
#  rrMSE_test = my_error(test_predict,test_y)
#  MSE_test = MSE(test_predict,test_y)
#  
#  errorMat<-c(rrMSE_train,rrMSE_test,MSE_train,MSE_test)
#  
  
#  names(errorMat) = c("rrmse_train","rrmse_test", "mse_train","mse_test")
#  errorMat
#}
#RandomForest_summary<-function(nfold,formula,data,train_X_orig,train_y,test_X_orig, test_y){
#  best_mtry = best_mtry_trainning(nfold,mtry_range,formula,data)
#  best_ntry = best_ntree(best_mtry,nfold,ntree_range,formula,data)
#  RForest_error = RForest(best_mtry,best_ntree,nfold,formula,data,train_X_orig,train_y,test_X_orig, test_y)
#  return(RForest_error)
#}

#Gam needs revision####
#Gam predict with trControl
Gam_CV<-function(formula,data,nfold){
  set.seed(100)
  folds <- sample(rep(1:nfold, length.out = nrow(data)))#all data
  fold_errorMat<-data.frame(matrix(NA,ncol=8,nrow=nfold+1))
  names(fold_errorMat)<-c("method_name","fold_num","rrmse_train","rrmse_test", "mse_train","mse_test","parameter","parameter range")
  fold_num = 0
  for(fold in 1:nfold){
    #data split####
    #cross validation
    fold_num =fold_num +1
    print(paste("fold_num:",fold_num,sep=""))
    idx <- which(folds==fold)
    train_orig = data[-idx,]
    test_orig  = data[idx,]
    
    train_X_orig = train_orig[,-Y_position]
    test_X_orig = test_orig[,-Y_position]
    
    #Y
    train_y =train_orig[,Y_position]
    test_y = test_orig[,Y_position]
    errorMat = Gam(formula,train_orig,nfold,train_X_orig,train_y,test_X_orig,test_y)
    fold_errorMat[fold_num,] = c("Gam",fold_num,errorMat)
  }
  fold_errorMat[fold_num+1,] = c("Gam","average",colSums(fold_errorMat[3],na.rm=TRUE)/nfold,colSums(fold_errorMat[4],na.rm=TRUE)/nfold,colSums(fold_errorMat[5],na.rm=TRUE)/nfold,colSums(fold_errorMat[6],na.rm=TRUE)/nfold,"","")
  fold_errorMat
}
Gam<-function(formula,data,nfold,train_X_orig,train_y,test_X_orig,test_y){
  set.seed(100)
  trControl <-trainControl(method = "cv", number = nfold)
  #f = ns(X_num[,1], df = k)
  #gglasso(formula,group = group1)
  #group1 <- rep(1:20,each=5)

  gam_fit <- train(formula,data = data, method="gamSpline",df = 5, trControl = trControl) #with default, no parameter tuned
  #train
  train_predict <- predict(gam_fit,train_X_orig)
  rrMSE_train = my_error(train_predict,train_y)
  MSE_train = MSE(train_predict,train_y)
  
  #test
  test_predict <- predict(gam_fit,test_X_orig)
  rrMSE_test = my_error(test_predict,test_y)
  MSE_test = MSE(test_predict,test_y)
  
  errorMat<-c(rrMSE_train,rrMSE_test,MSE_train,MSE_test)
  #errorMat<-c(errorMat,"","")
  #names(errorMat) = c("rrmse_train","rrmse_test", "mse_train","mse_test","parameter","parameter range")
  errorMat_result<-list("rrmse_train"=errorMat[1],"rrmse_test"=errorMat[2],
                        "mse_train"=errorMat[3],"mse_test"=errorMat[4],
                        "parameter"="","parameter range"="")
  #errorMat
  errorMat_result
}


#With given eta
XgBoost_summary_CV<-function(data,params,nrounds_max,nfold){
  set.seed(100)
  folds <- sample(rep(1:nfold, length.out = nrow(data)))#all data
  fold_errorMat<-data.frame(matrix(NA,ncol=8,nrow=nfold+1))
  names(fold_errorMat)<-c("method_name","fold_num","rrmse_train","rrmse_test", "mse_train","mse_test","parameter","parameter range")
  fold_num = 0
  for(fold in 1:nfold){
    #data split####
    #cross validation
    fold_num =fold_num +1
    print(paste("fold_num:",fold_num,sep=""))
    idx <- which(folds==fold)
    train_orig = data[-idx,]
    test_orig  = data[idx,]
    
    train_X_orig = train_orig[,-Y_position]
    test_X_orig = test_orig[,-Y_position]
    
    #Y
    train_y =train_orig[,Y_position]
    test_y = test_orig[,Y_position]
    errorMat = XgBoost_summary(train_X_orig,train_y, test_X_orig,test_y,params,nrounds_max,nfold)
    fold_errorMat[fold_num,] = c("XgBoost",fold_num,errorMat)
  }
  fold_errorMat[fold_num+1,] = c("XgBoost","average",colSums(fold_errorMat[3],na.rm=TRUE)/nfold,colSums(fold_errorMat[4],na.rm=TRUE)/nfold,colSums(fold_errorMat[5],na.rm=TRUE)/nfold,colSums(fold_errorMat[6],na.rm=TRUE)/nfold,"","")
  fold_errorMat
}
best_iteration<-function(dtrain,params,train_X_orig, train_y,nfold,nround_max){
  set.seed(100)
  
  #parameter not trained
  xgbcv <- xgb.cv(params = params, data = dtrain, nrounds = nround_max, nfold = 5, 
                  showsd = T, stratified = T,print_every_n = 40, early_stopping_rounds = 10, maximize = F)
  best_iteration = xgbcv$best_iteration
  return (best_iteration)
}
XgBoost<-function(dtrain,train_y,dtest,test_y,params,nrounds){
  set.seed(100)
  xgb_fit<-xgb.train (params = params, data = dtrain, nrounds =nrounds, eval_metric = "rmse")
  #train
  train_predict <- predict(xgb_fit, dtrain)
  rrMSE_train = my_error(train_predict,train_y)
  MSE_train = MSE(train_predict,train_y)
  print("train working")
  #test
  test_predict <- predict(xgb_fit, dtest)
  rrMSE_test = my_error(test_predict,test_y)
  MSE_test = MSE(test_predict,test_y)
  print("test working")
  errorMat<-c(rrMSE_train,rrMSE_test,MSE_train,MSE_test)
  names(errorMat) = c("rrmse_train","rrmse_test", "mse_train","mse_test")
  errorMat
}
XgBoost_summary<-function(train_X_orig,train_y, test_X_orig,test_y,params,nrounds_max, nfold){
  set.seed(100)
  dtrain <- xgb.DMatrix(data = data.matrix(train_X_orig),label = train_y)
  dtest <- xgb.DMatrix(data = data.matrix(test_X_orig),label = test_y)
  best_nround <- best_iteration(dtrain=dtrain,params = params,train_X_orig, train_y,nfold = nfold,nround_max = nrounds_max)
  Xg_error = XgBoost(dtrain=dtrain,train_y,dtest=dtest,test_y,params = params,nrounds = best_nround)
  #Xg_error<-c(Xg_error,paste("best_nround = ",best_nround,sep=""),paste("nround_max = ",nrounds_max,".package auto find",sep=""))
  errorMat_result<-list("rrmse_train"=Xg_error[1],"rrmse_test"=Xg_error[2],
                        "mse_train"=Xg_error[3],"mse_test"=Xg_error[4],
                        "parameter"=paste("best_nround = ",best_nround,sep=""),"parameter range"=paste("nround_max = ",nrounds_max,".package auto find",sep=""))
  #names(Xg_error) = c("rrmse_train","rrmse_test", "mse_train","mse_test","parameter","parameter range")
  #return (Xg_error)
  errorMat_result
  }



##Save Results####
export<-function(full_file_path, data, sheetName,append){
  write.xlsx( data, full_file_path,sheetName=sheetName, 
              col.names=TRUE, row.names=TRUE, append=append)
  
}


