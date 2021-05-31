#load PIE algorithm functions
load("functions.RData")
args<-commandArgs()
load_rdata_filename<-args[6]
load(load_rdata_filename)



lambda1 <-as.numeric(args[7])
lambda2<-as.numeric(args[8])
eta<-as.numeric(args[9])
iter <-as.numeric(args[10])
nrounds <-as.numeric(args[11])
stepsize<-as.numeric(args[12])
tree_nrounds<-as.numeric(args[13])
fold<-as.numeric(args[14])

parameter_set = paste("fold",fold,"lambda1:",lambda1,"lambda2:",lambda2,
                      "eta:",eta,"iter:",iter,"nrounds:",nrounds,
                      "stepsize:",stepsize,"tree_nrounds",tree_nrounds,
                      sep=" ")
print(parameter_set)

#1. stepsize
n = length(validation_train_y[[fold]])
d_j = c()
if(p > 0){
  for(J in 1:p){
    E = eigen(crossprod(as.matrix(spl_Validation_train_X[[fold]][,(((J-1)*5)+1):(J*5)]),as.matrix(spl_Validation_train_X[[fold]][,(((J-1)*5)+1):(J*5)])))
    d_j = c(d_j,(1/(4*n))*(max(E$values)))
  }
}

d_c = c()
n_dummy = c()
d_c_count = 0 #number of columns for categorical vars after dummy transfer
if (q!= 0){
  for(C in 1:q){
    #n_levels = length(levels(X_cat[,C]))-1 - prewriter in data load
    n_dummy = c(n_dummy,n_levels[[C]])
    d_c_count = d_c_count + n_levels[[C]]
    E = eigen(crossprod(as.matrix(spl_Validation_train_X[[fold]][,(d_c_count-n_levels[[C]]):d_c_count]),as.matrix(spl_Validation_train_X[[fold]][,(d_c_count-n_levels[[C]]):d_c_count])))
    d_c = c(d_c,(1/(4*n))*(max(E$values)))
  }
}

d_0 = 1/(4*n)

k=5

#######################

set.seed(100)
print("------------------PIE model starts training-----------------")
fit = Piano_fit(spl_Validation_train_X[[fold]],Validation_train_X_orig[[fold]],
                validation_train_y[[fold]],iter,lambda1,lambda2,eta,nrounds,
                d_j,d_c,d_0,n_dummy,nc_spline,stepsize,tree_nrounds)
print("-----------------PIE model finishes training----------------")
save.image(file="train.RData")