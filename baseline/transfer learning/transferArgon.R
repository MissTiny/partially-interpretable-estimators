#package and function loading####
library(glmnet)
library(randomForest)
library(gbm)
library(xgboost)
library(fastAdaboost)
library(e1071)
library(pROC)
#Normalization
normalize<-function(data){
  for(i in 1:ncol(data)){
    data[,i] = (data[,i]-min(data[,i]))/(max(data[,i])-min(data[,i]))
  }
  return (data)
}
#measure 
auc_function<-function(pred,true){
  roc_result<-roc(true,pred,levels = c(0,1),direction="<")
  auc_result<-auc(roc_result)
  return(auc_result)
}
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
##Data Loading####
data_file_train = "D:/Miss Tiny/Research/Piano Classification/Classification/Dataset/adult/adult"
#data_file_train="adult"
result = load_data_with_fold(data_file_train)
#fold structure###
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


set.seed(100)
nfold=1
for(fold in 1:nfold){
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
  train_y =train_orig[,Y_position]
  test_y = test_orig[,Y_position]
  #------Step1: Train simple model:Lasso--------------------####
  print("--------------simple method training-----------------")
  alpha = 1
  lambda_seq <- 10^seq(2, -10, by = -.1)
  nfold=5
  cv_output<-cv.glmnet(data.matrix(train_X_orig),train_y,
                       alpha = alpha, lambda = lambda_seq,nfolds=nfold)
  best_lam<-cv_output$lambda.min
  lasso_or_ridge_best <- glmnet(data.matrix(train_X_orig),train_y,
                                alpha = alpha, type.measure='auc',
                                family="binomial",lambda = best_lam)
  probabilities <-predict(lasso_or_ridge_best, s = best_lam, 
                          newx =data.matrix(train_X_orig),type="response")
  simple_label <- ifelse(probabilities > 0.5,1,0) 
  simple_err = sum(abs(train_y-probabilities))/length(probabilities)
  print(paste("simple_err: ",simple_err,sep=""))
  
  #------Step2: Train 5 Classifier:--------------------####
  #Classifier 1 : random forest
  #print("--------------Classifier 1: random forest  training-----------------")
  #rf_train_orig=train_orig
  #rf_train_orig[,Y_position] <- as.factor(rf_train_orig[,Y_position])
  #rf_test_orig =test_orig 
  #rf_test_orig[,Y_position] <- as.factor(rf_test_orig[,Y_position])
  #rf = randomForest(formula,data=rf_train_orig,ntree=100,proximity=T )
  #rf_pred=predict(rf,newdata=rf_train_orig,type="prob")
  #cf1_label <- ifelse(rf_pred[,2] > 0.5,1,0)
  #cf1_err=sum(abs(train_y-cf1_label))/length(train_y)
  #print(paste("\nrandom forest:" ,cf1_err,sep=""))
  
  #Classifier 2: gradient boosted Tree
  print("--------------Classifier 2: gradient boosted tree-----------------")
  bt_train_orig=train_orig
  bt_train_orig[,Y_position]<-as.character(bt_train_orig[,Y_position])
  bt=gbm(formula,data=bt_train_orig,distribution="bernoulli",n.trees=5000)
  bt_pred = predict(bt,newdata=bt_train_orig,n.trees=5000,type="response")
  cf2_label <- ifelse(bt_pred > 0.5,1,0)
  #cf2_err=sum(abs(train_y-bt_pred))/length(train_y)
  cf2_err=sum(abs(train_y-cf2_label))/length(train_y)
  print(paste("\nboosted Tree:", cf2_err,sep=""))
  
  #classifier 3: xgboost
  print("--------------Classifier 3: xgboost-----------------")
  params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.05, 
                 max_depth=6)
  dtrain <- xgb.DMatrix(data = data.matrix(train_X_orig),label = train_y)
  dtest <- xgb.DMatrix(data = data.matrix(test_X_orig),label = test_y)
  xgb_fit<-xgb.train (params = params, data = dtrain, nrounds =1000, eval_metric = "auc")
  xgb_pred <- predict(xgb_fit, dtrain)
  cf3_label <- ifelse(xgb_pred> 0.5,1,0)
  #cf3_err=sum(abs(train_y-xgb_pred))/length(train_y)
  cf3_err=sum(abs(train_y-cf3_label))/length(train_y)
  print(paste("\nxgboost Tree:",cf3_err,sep=""))
  
  #Classifier 4: Adaboost
  print("--------------Classifier 4: Adaboost-----------------")
  ada = adaboost(formula,train_orig,10)
  ada_pred = predict(ada,newdata=train_orig)
  cf4_label <- ifelse(ada_pred$prob[,2]> 0.5,1,0)
  #cf4_err=sum(abs(train_y-ada_pred$prob[,2]))/length(train_y)
  cf4_err=sum(abs(train_y-cf4_label))/length(train_y)
  print(paste("\nadaboost Tree:",cf4_err,sep=""))
  
  #Classifier 5: SVM
  print("--------------Classifier 5: svm-----------------")
  svm_train_orig=train_orig
  svm_train_orig[,Y_position]<-as.factor(svm_train_orig[,Y_position])
  s = svm(formula, data=svm_train_orig,probability=TRUE)
  s_pred = predict(s,data.matrix(train_X_orig),probability=TRUE)
  cf5_label <- ifelse(attr(s_pred,"probabilities")[,2]> 0.5,1,0)
  #cf5_err=sum(abs(train_y-attr(s_pred,"probabilities")[,2]))/length(train_y)
  cf5_err=sum(abs(train_y-cf5_label))/length(train_y)
  print(paste("\nsvm:",cf5_err,sep=""))
  
  #------Step3: Weights--------------------####
  print("--------------Weights-----------------")
  pred = list(rf_pred[,2],bt_pred,xgb_pred,ada_pred$prob[,2],attr(s_pred,"probabilities")[,2])
  err=list(cf1_err,cf2_err,cf3_err,cf4_err,cf5_err)
  w=c(0)
  count = 0
  for (i in c(1:5)){
    if (simple_err > err[[i]]){
      w =w+ pred[[i]]
      count=count+1
      print(paste(i, " is selected",sep=""))
    }
  }
  w=w/(count*probabilities)
  plot(density(w))
  a = quantile(w,probs=c(0.90))
  print(paste("90 percentile is ",a,sep=""))
  w_re = replace(w,w>min(c(a,2)),0)
  
  #------Step3: Re-train Simple model: Lasso####
  print("--------------Retrain-----------------")
  lambda_seq <- 10^seq(2, -10, by = -.1)
  new_cv_output<-cv.glmnet(data.matrix(train_X_orig),train_y,
                           alpha = alpha, lambda = lambda_seq,nfolds=nfold)
  new_best_lam<-new_cv_output$lambda.min
  new_lasso = glmnet(data.matrix(train_X_orig),c(train_y),alpha = alpha, type.measure='auc',family="binomial",weights=c(w_re),lambda = new_best_lam)
  new_pred <-predict(new_lasso, s = new_best_lam, newx =data.matrix(train_X_orig),type="response")
  new_label <- ifelse(new_pred> 0.5,1,0)
  #new_err= sum(abs(train_y-new_pred))/length(train_y)
  new_err= sum(abs(train_y-new_label))/length(train_y)
  
  test_old_pred <-predict(lasso_or_ridge_best, s = best_lam, newx =data.matrix(test_X_orig),type="response")
  test_new_pred <-predict(new_lasso, s = new_best_lam, newx =data.matrix(test_X_orig),type="response")
  new_test_label <- ifelse(test_new_pred> 0.5,1,0)
  old_test_label <- ifelse(test_old_pred> 0.5,1,0)
  #test_new_err = sum(abs(test_y-test_new_pred))/length(train_y)
  #test_old_err = sum(abs(test_y-test_old_pred))/length(train_y)
  test_new_err = sum(abs(test_y-new_test_label))/length(test_y)
  test_old_err = sum(abs(test_y-old_test_label))/length(test_y)
  
  auc_train = auc_function(new_pred,train_y)
  auc_test = auc_function(test_new_pred,test_y)
  
  print(paste("simple_err: ",simple_err,sep=""))
  print(paste("\nrandom forest:" ,cf1_err,sep=""))
  print(paste("\nboosted Tree:", cf2_err,sep=""))
  print(paste("\nxgboost Tree:",cf3_err,sep=""))
  print(paste("\nadaboost Tree:",cf4_err,sep=""))
  print(paste("\nsvm:",cf5_err,sep=""))
  print(paste("\nnew error:",new_err,sep=""))
  print(paste("\nnew simple err(test):",test_new_err,sep=""))
  print(paste("\nold simple err(test):",test_old_err,sep=""))
  print(paste("\nauc train:",auc_train,sep=""))
  print(paste("\nauc test:",auc_test,sep="" ))
}

