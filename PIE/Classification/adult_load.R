library(splines)
library(MASS)
library(fastDummies)
library(pROC)
#setwd("D:/Miss Tiny/Research/Piano Classification/Classification/Dataset/TelcoCustomer")
#data_file = "D:/Miss Tiny/Research/Piano Classification/Classification/Dataset/adult/adult"

#load data####
data_file="adult/adult"
set.seed(100)
result<-list()
for (i in 1:5){
  name=paste("fold",i,sep="")
  train_data<-read.csv(paste(data_file,"_train",i,".csv",sep=""))
  test_data<-read.csv(paste(data_file,"_test",i,".csv",sep=""))
  full_data <- rbind(train_data, test_data)
  fold <- list("full_data" = full_data,"train_data"=train_data,"test_data" = test_data)
  result[[name]]<-fold
}

##normalized####
normalize<-function(data){
  for(i in 1:ncol(data)){
    data[,i] = (data[,i]-min(data[,i]))/(max(data[,i])-min(data[,i]))
  }
  return (data)
}

normalized_result<-list()
normalized_result[[1]] = normalize(result$fold1$full_data)
normalized_result[[1]] = normalized_result[[1]][,-c(16,32,39,54,60,64,66,109)]
normalized_result[[2]] = normalize(result$fold2$full_data)
normalized_result[[2]] = normalized_result[[2]][,-c(16,32,39,54,60,64,66,109)]
normalized_result[[3]] = normalize(result$fold3$full_data)
normalized_result[[3]] = normalized_result[[3]][,-c(16,32,39,54,60,64,66,109)]
normalized_result[[4]] = normalize(result$fold4$full_data)
normalized_result[[4]] = normalized_result[[4]][,-c(16,32,39,54,60,64,66,109)]
normalized_result[[5]] = normalize(result$fold5$full_data)
normalized_result[[5]] = normalized_result[[5]][,-c(16,32,39,54,60,64,66,109)]

Y_position = 7
for (i in c(1:5)){
  normalized_result[[i]][,Y_position][which(normalized_result[[i]][,Y_position] == 0)] = -1
}
#n_levels=
df_spl<-list()
numerical_cols = c(1:6)
categorical_cols=c(7:100)
k = 5 #df
p = length(numerical_cols)
#number of categorical variable
q = 8
#dummy variable of each categorical variable
n_levels<-c(8,15,6,14,5,3,1,42)

nc_spline  = k*p
###transfer X_num to be splines
for (fold in 1:5){
  X_num= normalized_result[[fold]][,numerical_cols]
  #print(X_num)
  f = ns(X_num[,1], df = k) #df needs to be adjusted
  for(i in 2:ncol(X_num)){
    temp = ns(X_num[,i], df=5)
    f = cbind(f, temp)
  }
  N_categorical_cols = cbind(numerical_cols,Y_position)
  normalized_result[[fold]][,-N_categorical_cols] <-lapply(normalized_result[[fold]][,-N_categorical_cols],as.integer)
  X_dummy = normalized_result[[fold]][,-N_categorical_cols]
  y = normalized_result[[fold]][,Y_position]
  df_spl[[fold]] = data.frame(cbind(f, X_dummy, y))
}
  
train_orig=list()
Validation_train_orig_len=list()
Validation_train_orig=list()
Validation_test_orig=list()
test_orig =list()

Validation_train_X_orig=list()
Validation_test_X_orig=list()
test_X_orig=list()

spl_train=list()
spl_Validation_train_len=list()
spl_Validation_train = list()
spl_Validation_test=list()
spl_test=list()

spl_Validation_train_X=list()
spl_Validation_test_X=list()
spl_test_X=list()

validation_train_y=list()
validation_test_y=list()
test_y=list()

for(fold in 1:k){
  if (fold == 1){
    train_orig[[fold]] = normalized_result[[fold]][c(1:nrow(result$fold1$train_data)),]
    test_orig[[fold]]  = normalized_result[[fold]][-c(1:nrow(result$fold1$train_data)),]
    spl_train[[fold]] = df_spl[[fold]][c(1:nrow(result$fold1$train_data)),]
    spl_test[[fold]]  = df_spl[[fold]][-c(1:nrow(result$fold1$train_data)),] 
  } else if(fold == 2){
    train_orig[[fold]] = normalized_result[[fold]][c(1:nrow(result$fold2$train_data)),]
    test_orig[[fold]]  = normalized_result[[fold]][-c(1:nrow(result$fold2$train_data)),]
    spl_train[[fold]] = df_spl[[fold]][c(1:nrow(result$fold2$train_data)),]
    spl_test[[fold]]  = df_spl[[fold]][-c(1:nrow(result$fold2$train_data)),] 
  }else if(fold == 3){
    train_orig[[fold]] = normalized_result[[fold]][c(1:nrow(result$fold3$train_data)),]
    test_orig[[fold]]  = normalized_result[[fold]][-c(1:nrow(result$fold3$train_data)),]
    spl_train[[fold]] = df_spl[[fold]][c(1:nrow(result$fold3$train_data)),]
    spl_test[[fold]]  = df_spl[[fold]][-c(1:nrow(result$fold3$train_data)),] 
  }else if(fold == 4){
    train_orig[[fold]] = normalized_result[[fold]][c(1:nrow(result$fold4$train_data)),]
    test_orig[[fold]]  = normalized_result[[fold]][-c(1:nrow(result$fold4$train_data)),]
    spl_train[[fold]] = df_spl[[fold]][c(1:nrow(result$fold4$train_data)),]
    spl_test[[fold]]  = df_spl[[fold]][-c(1:nrow(result$fold4$train_data)),] 
  }else if(fold == 5){
    train_orig[[fold]] = normalized_result[[fold]][c(1:nrow(result$fold5$train_data)),]
    test_orig[[fold]]  = normalized_result[[fold]][-c(1:nrow(result$fold5$train_data)),]
    spl_train[[fold]] = df_spl[[fold]][c(1:nrow(result$fold5$train_data)),]
    spl_test[[fold]]  = df_spl[[fold]][-c(1:nrow(result$fold5$train_data)),] 
  }
  
  Validation_train_orig_len[[fold]] = ceiling(0.8 * nrow(train_orig[[fold]]))
  Validation_train_orig[[fold]] = train_orig[[fold]][c(1:Validation_train_orig_len[[fold]]),]
  Validation_test_orig[[fold]] = train_orig[[fold]][-c(1:Validation_train_orig_len[[fold]]),]
  
  Validation_train_X_orig[[fold]] = Validation_train_orig[[fold]][,-Y_position]
  Validation_test_X_orig[[fold]] = Validation_test_orig[[fold]][,-Y_position]
  test_X_orig[[fold]] = test_orig[[fold]][,-Y_position]
  
  #splined data
  
  spl_Validation_train_len[[fold]] = ceiling(0.8 * nrow(spl_train[[fold]]))
  spl_Validation_train[[fold]] = spl_train[[fold]][c(1:spl_Validation_train_len[[fold]]),]
  spl_Validation_test[[fold]] = spl_train[[fold]][-c(1:spl_Validation_train_len[[fold]]),]
  
  
  spl_Validation_train_X[[fold]] = spl_Validation_train[[fold]][,1:(ncol(spl_Validation_train[[fold]])-1)]
  spl_Validation_test_X[[fold]] = spl_Validation_test[[fold]][,1:(ncol(spl_Validation_test[[fold]])-1)]
  spl_test_X[[fold]] = spl_test[[fold]][,1:(ncol(spl_test[[fold]])-1)]
  
  #Y
  validation_train_y[[fold]] = Validation_train_orig[[fold]][,Y_position]
  validation_test_y[[fold]] =Validation_test_orig[[fold]][,Y_position]
  test_y[[fold]] = test_orig[[fold]][,Y_position]
}

save.image(file="load_adult.RData")

