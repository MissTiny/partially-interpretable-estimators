#library####
library(splines)
library(glmnet)
library(MLmetrics)
library(rpart)
library(mgcv)
library(caret)
library(xgboost)
library(openxlsx)
library(pROC)
library(fastDummies)
##Load Data####
load_data_with_fold<-function(data_file){
  result<-list()
  for (i in 1:5){
    name=paste("fold",i,sep="")
    train_data<-read.csv(paste(data_file,"_train",i,".csv",sep=""))
    test_data<-read.csv(paste(data_file,"_test",i,".csv",sep=""))
    full_data <- rbind(train_data, test_data)
    fold <- list("full_data" = full_data,"train_data"=train_data,"test_data" = test_data)
    result[[name]]<-fold
  }
  result
}
load_data_with_index_onefold<-function(data_file){
  data <- read.csv(data_file)[,-1]
  result <- list("full_data" = data)
  return(result)
}
load_data_without_index_onefold<-function(data_file){
  data <- read.csv(data_file)
  result <- list("full_data" = data)
  return(result)
}
load_data_with_index<-function(data_file_train,data_file_test){
  train_data <- read.csv(data_file_train)[,-1]
  test_data <- read.csv(data_file_test)[,-1]
  full_data <- rbind(train_data, test_data)
  full_data <-na.omit(full_data)
  result <- list("full_data" = full_data,"train_data"=train_data,"test_data" = test_data)
  return(result)
}
load_data_without_index<-function(data_file_train,data_file_test){
  train_data <- read.csv(data_file_train)
  test_data <- read.csv(data_file_test)
  full_data <- rbind(train_data, test_data)
  full_data <-na.omit(full_data)
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
auc_function<-function(pred,true){
  roc_result<-roc(true,pred,levels = c(0,1),direction="<")
  auc_result<-auc(roc_result)
  return(auc_result)
}

DecisonTree_CV<-function(formula,fold1_train,fold1_test,fold2_train,fold2_test,
                         fold3_train,fold3_test,fold4_train,fold4_test,
                         fold5_train,fold5_test,nfold){
  set.seed(100)
  fold_errorMat<-data.frame(matrix(NA,ncol=6,nrow=nfold+1))
  names(fold_errorMat)<-c("method_name","fold_num","auc_train","auc_test","parameter","parameter range")
  for(fold in 1:nfold){
    #data split####
    #cross validation
    print(paste("fold_num:",fold,sep=""))
    if (fold== 1){
      train_orig = fold1_train
      test_orig  = fold1_test
    }else if(fold ==2){
      train_orig = fold2_train
      test_orig  = fold2_test
    }else if(fold ==3){
      train_orig = fold3_train
      test_orig  = fold3_test
    }else if(fold ==4){
      train_orig = fold4_train
      test_orig  = fold4_test
    }else if(fold ==5){
      train_orig = fold5_train
      test_orig  = fold5_test
    }
    
    train_X_orig = train_orig[,-Y_position]
    test_X_orig = test_orig[,-Y_position]
    
    #Y
    train_y =train_orig[,Y_position]
    test_y = test_orig[,Y_position]

    errorMat = DecisionTree(formula,train_orig,train_X_orig,train_y,test_X_orig,test_y,nfold)
    fold_errorMat[fold,] = c("Decision_Tree",fold,errorMat)
  }
  fold_errorMat[fold+1,] = c("Decision_Tree","average",colSums(fold_errorMat[3],na.rm=TRUE)/nfold,colSums(fold_errorMat[4],na.rm=TRUE)/nfold,"","")
  fold_errorMat
}
##baseline Functions####
DecisionTree<-function(formula,data,train_X_orig,train_y,test_X_orig,test_y,nfold){
  #random
  set.seed(100)
  ##train####
  #xval number of corss validations, anova = regression
  train_tree<-rpart(formula,
                    data = data.frame(data),method = "class",
                    control=rpart.control(xval=nfold))
  
  #pruned tree
  bestcp <- train_tree$cptable[which.min(train_tree$cptable[,"xerror"]),"CP"]
  print(bestcp)
  train_pruned_tree <-prune(train_tree,bestcp)
  #train_predict = predict(train_pruned_tree,type="class")
  train_predict = predict(train_pruned_tree)
  ##train error rate
  #auc_train = auc_function(train_predict,train_y)
  auc_train = auc_function(train_predict[,2],train_y)
  ##Test####
  #test_predict<-predict(train_pruned_tree,newdata=test_X_orig,type="class")
  test_predict<-predict(train_pruned_tree,newdata=test_X_orig)
  #test error rate
  #auc_test = auc_function(test_predict,test_y)
  auc_test = auc_function(test_predict[,2],test_y)
  errorMat<-c(auc_train,auc_test)
  #errorMat<-c(errorMat,paste("cp=",bestcp,sep=""),"package auto find")
  #names(errorMat) = c("rrmse_train","rrmse_test", "mse_train","mse_test","parameter","parameter range")
  errorMat_result<-list("auc_train"=errorMat[1],"auc_test"=errorMat[2],
                        "parameter"=paste("cp=",bestcp,sep=""),"parameter range"="package auto find")
  #errorMat
  errorMat_result
}

#####
Lasso_or_Ridge_CV<-function(fold1_train,fold1_test,fold2_train,fold2_test,
                            fold3_train,fold3_test,fold4_train,fold4_test,
                            fold5_train,fold5_test,alpha,nfold){
  set.seed(100)
  fold_errorMat<-data.frame(matrix(NA,ncol=6,nrow=nfold+1))
  names(fold_errorMat)<-c("method_name","fold_num","auc_train","auc_test","parameter","parameter range")
  method="Lasso"
  if(alpha==0){
    method="Ridge"
  }
  for(fold in 1:nfold){
    #data split####
    #cross validation
    print(paste("fold_num:",fold,sep=""))
    if (fold== 1){
      train_orig = fold1_train
      test_orig  = fold1_test
    }else if(fold ==2){
      train_orig = fold2_train
      test_orig  = fold2_test
    }else if(fold ==3){
      train_orig = fold3_train
      test_orig  = fold3_test
    }else if(fold ==4){
      train_orig = fold4_train
      test_orig  = fold4_test
    }else if(fold ==5){
      train_orig = fold5_train
      test_orig  = fold5_test
    }
    #data split####
    #cross validation
    
    train_X_orig = train_orig[,-Y_position]
    test_X_orig = test_orig[,-Y_position]
    
    #Y
    train_y =train_orig[,Y_position]
    test_y = test_orig[,Y_position]
    errorMat = Lasso_or_Ridge(train_X_orig,train_y,alpha,test_X_orig,test_y,nfold)
    fold_errorMat[fold,] = c(method,fold,errorMat)
  }
  
  fold_errorMat[fold+1,] = c(method,"average",colSums(fold_errorMat[3],na.rm=TRUE)/nfold,colSums(fold_errorMat[4],na.rm=TRUE)/nfold,"","")
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
  lasso_or_ridge_best <- glmnet(data.matrix(train_X_orig),train_y,alpha = alpha, type.measure='auc',family="binomial",lambda = best_lam)
  
  #train
  train_predict <- predict(lasso_or_ridge_best, s = best_lam, newx =data.matrix(train_X_orig),type="response")
  auc_train = auc_function(train_predict[,1],train_y)
  
  #test
  test_predict <- predict(lasso_or_ridge_best, s = best_lam, newx = data.matrix(test_X_orig),type="response")
  auc_test = auc_function(test_predict[,1],test_y)
  
  errorMat<-c(auc_train,auc_test)
  #errorMat<-c(errorMat,paste("lambda = ",best_lam,sep=""),"lambda = 10^seq(2, -10, by = -.1)")
  errorMat_result<-list("auc_train"=errorMat[1],"auc_test"=errorMat[2],
                        "parameter"=paste("lambda = ",best_lam,sep=""),"parameter range"="lambda = 10^seq(2, -10, by = -.1)")
  #names(errorMat) = c("rrmse_train","rrmse_test", "mse_train","mse_test","parameter","parameter range")
  #print(errorMat)
  #errorMat
  errorMat_result
}



#Gam needs revision####
#Gam predict with trControl
Gam_CV<-function(formula,fold1_train,fold1_test,fold2_train,fold2_test,
                 fold3_train,fold3_test,fold4_train,fold4_test,
                 fold5_train,fold5_test,nfold){
  set.seed(100)
  
  fold_errorMat<-data.frame(matrix(NA,ncol=6,nrow=nfold+1))
  names(fold_errorMat)<-c("method_name","fold_num","auc_train","auc_test","parameter","parameter range")
  for(fold in 1:nfold){
    #data split####
    #cross validation
    print(paste("fold_num:",fold,sep=""))
    if (fold== 1){
      train_orig = fold1_train
      test_orig  = fold1_test
    }else if(fold ==2){
      train_orig = fold2_train
      test_orig  = fold2_test
    }else if(fold ==3){
      train_orig = fold3_train
      test_orig  = fold3_test
    }else if(fold ==4){
      train_orig = fold4_train
      test_orig  = fold4_test
    }else if(fold ==5){
      train_orig = fold5_train
      test_orig  = fold5_test
    }
    
    train_X_orig = train_orig[,-Y_position]
    test_X_orig = test_orig[,-Y_position]
    
    #Y
    train_y =train_orig[,Y_position]
    test_y = test_orig[,Y_position]
    train_orig[,Y_position] = as.factor(train_orig[,Y_position])
    errorMat = Gam(formula,train_orig,nfold,train_X_orig,train_y,test_X_orig,test_y)
    fold_errorMat[fold,] = c("Gam",fold,errorMat)
  }
  fold_errorMat[fold+1,] = c("Gam","average",colSums(fold_errorMat[3],na.rm=TRUE)/nfold,colSums(fold_errorMat[4],na.rm=TRUE)/nfold,"","")
  fold_errorMat
}
Gam<-function(formula,data,nfold,train_X_orig,train_y,test_X_orig,test_y){
  set.seed(100)
  trControl <-trainControl(method = "cv", number = nfold)
  gam_fit <- train(formula,data = data, method="gamSpline",df = 5, trControl = trControl,family="binomial") #with default, no parameter tuned
  #train
  train_predict <- predict(gam_fit,train_X_orig,type="prob")
  print(train_predict)
  auc_train = auc_function(train_predict[,2],train_y)
 
  
  #test
  test_predict <- predict(gam_fit,test_X_orig,type="prob")
  auc_test = auc_function(test_predict[,2],test_y)
 
  
  errorMat<-c(auc_train,auc_test)
  #errorMat<-c(errorMat,"","")
  #names(errorMat) = c("rrmse_train","rrmse_test", "mse_train","mse_test","parameter","parameter range")
  errorMat_result<-list("auc_train"=errorMat[1],"auc_test"=errorMat[2],
                        
                        "parameter"="","parameter range"="")
  #errorMat
  errorMat_result
}


#With given eta
XgBoost_summary_CV<-function(data,params,nrounds_max,nfold){
  set.seed(100)
  fold_errorMat<-data.frame(matrix(NA,ncol=6,nrow=nfold+1))
  names(fold_errorMat)<-c("method_name","fold_num","auc_train","auc_test","parameter","parameter range")
  for(fold in 1:nfold){
    #data split####
    #cross validation
    print(paste("fold_num:",fold,sep=""))
    if (fold== 1){
      train_orig = fold1_train
      test_orig  = fold1_test
    }else if(fold ==2){
      train_orig = fold2_train
      test_orig  = fold2_test
    }else if(fold ==3){
      train_orig = fold3_train
      test_orig  = fold3_test
    }else if(fold ==4){
      train_orig = fold4_train
      test_orig  = fold4_test
    }else if(fold ==5){
      train_orig = fold5_train
      test_orig  = fold5_test
    }
    train_X_orig = train_orig[,-Y_position]
    test_X_orig = test_orig[,-Y_position]
    
    #Y
    train_y =train_orig[,Y_position]
    test_y = test_orig[,Y_position]
    errorMat = XgBoost_summary(train_X_orig,train_y, test_X_orig,test_y,params,nrounds_max,nfold)
    fold_errorMat[fold,] = c("XgBoost",fold,errorMat)
  }
  fold_errorMat[fold+1,] = c("XgBoost","average",colSums(fold_errorMat[3],na.rm=TRUE)/nfold,colSums(fold_errorMat[4],na.rm=TRUE)/nfold,"","")
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
  xgb_fit<-xgb.train (params = params, data = dtrain, nrounds =nrounds, eval_metric = "auc")
  #train
  train_predict <- predict(xgb_fit, dtrain)
  auc_train = auc_function(train_predict,train_y)


  #test
  test_predict <- predict(xgb_fit, dtest)
  auc_test = auc_function(test_predict,test_y)


  errorMat<-c(auc_train,auc_test)
  names(errorMat) = c("auc_train","auc_test")
  errorMat
}
XgBoost_summary<-function(train_X_orig,train_y, test_X_orig,test_y,params,nrounds_max, nfold){
  set.seed(100)
  dtrain <- xgb.DMatrix(data = data.matrix(train_X_orig),label = train_y)
  dtest <- xgb.DMatrix(data = data.matrix(test_X_orig),label = test_y)
  best_nround <- best_iteration(dtrain=dtrain,params = params,train_X_orig, train_y,nfold = nfold,nround_max = nrounds_max)
  Xg_error = XgBoost(dtrain=dtrain,train_y,dtest=dtest,test_y,params = params,nrounds = best_nround)
  #Xg_error<-c(Xg_error,paste("best_nround = ",best_nround,sep=""),paste("nround_max = ",nrounds_max,".package auto find",sep=""))
  errorMat_result<-list("auc_train"=Xg_error[1],"auc_test"=Xg_error[2],
                        "parameter"=paste("best_nround = ",best_nround," eta =",params$eta," max_depth =",params$max_depth,sep =""),"parameter range"=paste("nround_max = ",nrounds_max,".package auto find",sep=""))
  #names(Xg_error) = c("rrmse_train","rrmse_test", "mse_train","mse_test","parameter","parameter range")
  #return (Xg_error)
  errorMat_result
}



##Save Results####
export<-function(full_file_path, data, sheetName,append){
  write.xlsx( data, full_file_path,sheetName=sheetName, 
              col.names=TRUE, row.names=TRUE, append=append)
  
}


