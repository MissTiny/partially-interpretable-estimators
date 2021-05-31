library(xgboost)
library(MASS)
library(splines)

args<-commandArgs()
data_name<-args[6]
total_count <- as.numeric(args[7])
error_filename<-args[8]
rdata = paste(data_name,"_","[0-9]*","_","[0-9]*",".RData",sep="")
files<-list.files(pattern=rdata)
print(length(files))
fold_matrix<-data.frame(matrix(NA,ncol=16,nrow=total_count))
names(fold_matrix) = c("count","Lambda1", "Lambda2","Iteration","Nrounds",
                       "stepsize","tree_nrounds","sparsity","auc_val_train","auc_lasso_val_train","auc_val_test",
                       "auc_lasso_val_test","val_interpretability","auc_test","auc_lasso_test","interpretability")
#load all RData file that belongs to the fold
num = 0
for (i in files){
  load(file=i)
  num=num+1
  fold_matrix[num,]=errorMat[1:16]
}

write.csv(fold_matrix, error_filename)

