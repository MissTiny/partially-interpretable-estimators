library(xgboost)
library(MASS)
library(splines)

args<-commandArgs()
data_name<-args[6]
total_count <- as.numeric(args[7])
error_filename<-args[8]
cv_filename<-args[9]
rdata = paste(data_name,"_","[0-9]*","_","[0-9]*",".RData",sep="")
files<-list.files(pattern=rdata)
print(length(files))
fold_matrix<-data.frame(matrix(NA,ncol=16,nrow=total_count))
names(fold_matrix) = c("count","fold_num","iter", "lambda1","lambda2","stepsize",
                       "eta","nrounds","sparsity","rrmse_validation_train",
                       "rrmse_validation_test","rrmse_Validation_GAM_only",
                       "Val_Interpretability",
                       "rrmse_test","rrmse_test_GAM_only","Interpretability")
#load all RData file that belongs to the fold
num = 0
for (i in files){
  load(file=i)
  num=num+1
  fold_matrix[num,]=errorMat[1:16]
}
cv_matrix<-data.frame(matrix(NA,ncol=16,nrow=k+1))
names(cv_matrix) = c("count","fold_num","iter", "lambda1","lambda2","stepsize",
                       "eta","nrounds","sparsity","rrmse_validation_train",
                       "rrmse_validation_test","rrmse_Validation_GAM_only",
                       "Val_Interpretability",
                       "rrmse_test","rrmse_test_GAM_only","Interpretability")
k=5
dim(fold_matrix)
for (i in c(1:k)){
  print(i)
  fold_summy = na.omit(fold_matrix[which(fold_matrix$fold_num == i),])
  dim(fold_summy)
  cv_matrix[i,] = fold_summy[which(fold_summy$Val_Interpretability==max(fold_summy[which((fold_summy$rrmse_Validation_GAM_only < 1) & (fold_summy$sparsity <= 8)),]$Val_Interpretability))[1],]
}
avg_rrmse = sum(as.numeric(cv_matrix$rrmse_test),na.rm=TRUE)/k
avg_rrmse_Gam = sum(as.numeric(cv_matrix$rrmse_test_GAM_only),na.rm=TRUE)/k
interpret_for_avg = (1-avg_rrmse_Gam)/(1-avg_rrmse)
cv_matrix[k+1,]=c("","average","","","","","","","","","","","",avg_rrmse,avg_rrmse_Gam,interpret_for_avg)

write.csv(fold_matrix, error_filename)
write.csv(cv_matrix,cv_filename)
