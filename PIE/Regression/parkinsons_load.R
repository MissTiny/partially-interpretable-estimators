library(xgboost)
library(MASS)
library(splines)
setwd("D:/Miss Tiny/Research/Piano Classification/Regression/No_orthogonal/Data")
#data loading
df1 <- read.csv("parkinsons_train.csv")
df2 <- read.csv("parkinsons_test.csv")
df <- rbind(df1, df2)
numerical_cols = c(1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)
categorical_cols = c(2)
col_names = names(df)
names(df)<-c("x1","x2","x3","x4","x5","x6","x7","x8","x9","x10",
             "x11","x12","x13","x14","x15","x16","x17",
             "x18","x19","x20","y")
Y_position = ncol(df)
n_level=c(1)

##Normalization
for(i in 1:ncol(df)){
  df[,i] = (df[,i]-min(df[,i]))/(max(df[,i])-min(df[,i]))
}

X = as.matrix(df[,-Y_position])  #Features 
y = df[,Y_position]         #target 

f = ns(X[,numerical_cols[1]], df = 5) #df needs to be adjusted
for(i in numerical_cols[2:length(numerical_cols)]){
  temp = ns(X[,i], df=5)
  f = cbind(f, temp)
}

df_spl = as.matrix(cbind(f, X[,categorical_cols],y))
lasso_group = rep(1:ncol(X[,numerical_cols]),each=5)
lasso_group = c(lasso_group,(n_level +ncol(X[,numerical_cols])))
dim(as.matrix(X[,categorical_cols]))
dim(f)
dim(df_spl)

k=5
folds <- sample(rep(1:k, length.out = nrow(df)))#df
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
  idx <- which(folds==fold)
  train_orig[[fold]] = df[-idx,]
  Validation_train_orig_len[[fold]] = ceiling(0.8 * nrow(train_orig[[fold]]))
  Validation_train_orig[[fold]] = train_orig[[fold]][c(1:Validation_train_orig_len[[fold]]),]
  Validation_test_orig[[fold]] = train_orig[[fold]][-c(1:Validation_train_orig_len[[fold]]),]
  test_orig[[fold]]  = df[idx,]
  
  Validation_train_X_orig[[fold]] = Validation_train_orig[[fold]][,-Y_position]
  Validation_test_X_orig[[fold]] = Validation_test_orig[[fold]][,-Y_position]
  test_X_orig[[fold]] = test_orig[[fold]][,-Y_position]
  
  #splined data
  spl_train[[fold]] = df_spl[-idx,] 
  spl_Validation_train_len[[fold]] = ceiling(0.8 * nrow(spl_train[[fold]]))
  spl_Validation_train[[fold]] = spl_train[[fold]][c(1:spl_Validation_train_len[[fold]]),]
  spl_Validation_test[[fold]] = spl_train[[fold]][-c(1:spl_Validation_train_len[[fold]]),]
  spl_test[[fold]]  = df_spl[idx,] 
  
  spl_Validation_train_X[[fold]] = spl_Validation_train[[fold]][,1:(ncol(spl_Validation_train[[fold]])-1)]
  spl_Validation_test_X[[fold]] = spl_Validation_test[[fold]][,1:(ncol(spl_Validation_test[[fold]])-1)]
  spl_test_X[[fold]] = spl_test[[fold]][,1:(ncol(spl_test[[fold]])-1)]
  
  #Y
  validation_train_y[[fold]] = Validation_train_orig[[fold]][,Y_position]
  validation_test_y[[fold]] =Validation_test_orig[[fold]][,Y_position]
  test_y[[fold]] = test_orig[[fold]][,Y_position]
}

save.image(file="load_parkinsons.RData")
