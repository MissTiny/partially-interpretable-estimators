rm(list=ls())
###Loading packges#######
library(splines)
library(MASS)
library(fastDummies)
library(pROC)
library(xgboost)
#function####
Gradient_alpha<-function(n, w_vec, b_vec, beta0, y, spline_j){
  w_vec = as.vector(w_vec)
  temp = (1/(1+exp((w_vec+b_vec+beta0)*y)))*y
  as.vector(-temp %*% spline_j / n)
}


Gradient_beta0<-function(w_vec, b_vec, beta0, y){
  temp = 1/(1+exp((w_vec+b_vec+beta0)*y))
  return(-colSums(temp*y)/length(y))  ##returning 2.gradient of beta0
}

relu<- function(x){
  ifelse(x>0, x, 0)
}

#Part B: Tree 
Gi <- function(y, w_vec, b_vec, beta0){ 
  #return the first order gradient of the objective function
  #remember to sort the w_vec and b_vec and beta0 along with X and y.
  temp = exp((w_vec+b_vec+beta0)*y)
  return((1/(1+temp))*(-y))
}

Obj_tree <- function(y, w_vec, b_vec, beta0, lambda2,gamma,n_nodes,wgt_levs){
  #return the objective that we want to minimize

  #p1 = log(1+exp((w_vec+b_vec+beta0)*(-y)))
  p1 = ifelse((w_vec+b_vec+beta0)*(-y) > 20,(w_vec+b_vec+beta0)*(-y), log(1+exp((w_vec+b_vec+beta0)*(-y))))
  #straightly equal to exp if 20 more

  p2 = gamma*n_nodes+sum((wgt_levs)^2)
  return(sum(p1)+lambda2*(0.5)*p2)
}

loss <- function(y, w_vec, b_vec, beta0){
  #the loss function 
  return(log(1+exp((w_vec+b_vec+beta0)*(-y))))
}

pred_prob<-function(fx){
  p_y1 = 1/(1+exp(-fx))
  p_y = 1/(1+exp(fx))
  df = data.frame("prob_positive" = p_y1, "prob_negative"= p_y)
  #pred = lapply(p_y1, function(x) if (x <0.5) {-1} else {1})
  pred = as.numeric(p_y1 > 0.5)
  pred[which(pred==0)]=-1
  return(list("prob" = df, "pred" = pred))
}

sparsity_count<-function(vec,n_levels,p){
  count = 0
  total=0
  for (i in c(0:(p-1))){
    value =  vec[c((i*5+1):((i+1)*5))]
    total=total+1
    if (sum(value) != 0){
      count=count + 1 #number of nonzero betas
    }
  }
  new_levels = c(0,n_levels)
  for (i in c(1:(length( new_levels)-1))){
    if (i == 1){
      value =  vec[c((p*5+1):(p*5+sum(n_levels[c(1:(i+1))])))]
    }else{
      value =  vec[c((p*5+sum( new_levels[c(1:i)])+1):(p*5+sum( new_levels[c(1:(i+1))])))]
    }
    total=total+1
    if (sum(value) != 0){
      count=count + 1 #number of nonzero betas
    }
  }
  print(total)
  count
}
###############################################
Piano_fit<-function(train_X,train_X_orig,train_y,iter,lambda1,lambda2,eta,nrounds,d_j,d_c,d_0,n_dummy,nc_spline,stepsize,tree_nrounds){
  obj_value = c()
  n = length(train_y)
  alpha = runif(ncol(train_X),0,1)
  w_vec = as.matrix(train_X)%*%alpha
  b_vec = rep(0, length(train_y))
  beta0 = 0
  
  train_X = as.matrix(train_X)
  
  #iter  = 10
  #A_rec = matrix(,nrow = iter, ncol = length(alpha))
  #B_rec = matrix(,nrow = iter, ncol = 1)
  Tree_list = list()
  Tree_pred = matrix(0, nrow = length(train_y),ncol = iter)
  #Obj_tree(train_y, w_vec, b_vec,beta0,lambda2,gamma, n_nodes=0, wgt_levs = 0)
  for(i in 1:iter){
    print(paste("i=",i))
    ptm1<-proc.time()
    for (r in 1: nrounds){
      #print(paste("nrounds=",r))
      for (j in 1:p){
        w_vec = as.matrix(train_X)%*%alpha
        grad_a = Gradient_alpha(n, w_vec, b_vec, beta0, train_y, train_X[,((j-1)*k+1): (j*k)])
        alpha_j = alpha[((j-1)*k+1): (j*k)]
        group_ind = relu(1-(lambda1/sqrt(sum((-grad_a+(1/d_j[j])*alpha_j)^2))))
        #if(group_ind==0){print(j, group_ind)}
        alpha[((j-1)*k+1): (j*k)] = d_j[j]*(-grad_a+(1/d_j[j])*alpha_j)*group_ind
        
      }
      
      if(is.null(d_c)==FALSE){
        n_dummy_sum = 0
        for(d in 1:q){
          #print(paste("d = ",d, sep = ""))
          w_vec = as.matrix(train_X)%*%alpha
          n_dummy_sum = n_dummy_sum+n_dummy[d]
          #print(paste("n_dummy_sum = ", n_dummy_sum, sep = ""))
          grad_a = Gradient_alpha(n, w_vec, b_vec, beta0, train_y, train_X[,(nc_spline+n_dummy_sum-n_dummy[d]+1):(nc_spline+n_dummy_sum)])
          #print(paste("grad_a = ", grad_a, sep = ""))
          alpha_d = alpha[(nc_spline+n_dummy_sum-n_dummy[d]+1):(nc_spline+n_dummy_sum)]
          #print(paste("alpha_d = ", alpha_d))
          group_ind = relu(1-(lambda1/sqrt(sum((-grad_a+(1/d_c[d])*alpha_d)^2))))
          alpha[(nc_spline+n_dummy_sum-n_dummy[d]+1):(nc_spline+n_dummy_sum)] = d_c[d]*(-grad_a+(1/d_c[d])*alpha_d)*group_ind
        }
      }
      
    }
    
    w_vec = as.matrix(train_X)%*%alpha
    grad_b = Gradient_beta0(w_vec, b_vec, beta0, train_y)
    beta0 = beta0-d_0*grad_b
    ptm2 <-proc.time()
    print(paste("Tree fit and predict:",ptm2[3]-ptm1[3],sep=""))
    # A_rec[i,] = alpha
    # B_rec[i,] = beta0
    ############
    #Fit One Tree here
    #ptm <- proc.time()
    print("Fit Tree begin start")
    ptm <- proc.time()
    gi = Gi(train_y,w_vec,b_vec,beta0)
    
    dtrain <- xgb.DMatrix(data = as.matrix(train_X_orig),label = gi)
    params <- list(booster = "gbtree", objective = "reg:squarederror", eta=eta, lambda = lambda2, gamma = lambda2,max_depth=6)
    Tree_list[[i]] <- xgb.train (params = params, data = dtrain, nrounds = tree_nrounds, eval_metric = "rmse")
    #print(Trees[[i]])
    leaves = length(xgb.dump(Tree_list[[i]],with_stats=TRUE))
    if (leaves != (2*tree_nrounds)){
       Tree_pred[,i] = predict(Tree_list[[i]],dtrain)
    
       ##########
       ptm4 <- proc.time()
       print(paste("Tree fit and predict:",ptm4[3]-ptm[3],sep=""))
    
       b_vec = b_vec - stepsize*Tree_pred[,i]
       #timer = proc.time() - ptm
       #print(timer)
       ###Test obj value updates
       print(Tree_list[[i]])
       Tree_model = xgb.model.dt.tree(model=Tree_list[[i]])
    
       n_nodes = nrow(Tree_model)#number of nodes
       #Terminal leaf value
       wgt_levs = sum((Tree_model$Quality[which(Tree_model$Feature == "Leaf")]^2))
     }else {
        print("tree not fitting")
        n_nodes = 0
        wgt_levs = 0
     }
    obj_value = c(obj_value,Obj_tree(train_y, w_vec, b_vec, beta0, lambda2, lambda2, n_nodes, wgt_levs))
    ############
    if(i>1){
      if(abs(obj_value[length(obj_value)-1]-obj_value[length(obj_value)])<0.1){
        print(obj_value)
        #objective[count,] = c(lambda1,lambda2,gamma,min(obj_value))
        best_iter = i-1
        print(paste("best_iter:",i-1,sep=""))
        break
      }else{
        best_iter = i
        if(i==iter){
          print(obj_value)
        }
      }
    }
    
  }
  fx = w_vec + beta0 +b_vec
  lasso = w_vec + beta0
  pred_train_prob <- pred_prob(fx)
  lasso_prob <- pred_prob(lasso)
  auc <- auc(roc(train_y, pred_train_prob$prob$prob_positive, smooth=FALSE, auc=TRUE, quiet = TRUE))
  auc_lasso<-auc(roc(train_y,lasso_prob$prob$prob_positive,smooth=FALSE, auc=TRUE, quiet = TRUE))
  #auc <- AUC(pred_train_prob$pred, train_y)
  #auc_lasso <- AUC(lasso_prob$pred, train_y)
  myloss<-log(1+exp((fx)*(-train_y)))
  loss_lasso <- log(1+exp((lasso)*(-train_y)))
  return(list("alpha"=alpha,"beta0"=beta0,"Trees"=Tree_list, "pred_train_prob"= pred_train_prob$prob, "pred_train" = pred_train_prob$pred, "auc" = auc, "auc_lasso" = auc_lasso, "loss" = myloss, "loss_lasso" = loss_lasso, "obj_values"=obj_value,"best_iter"=best_iter))
}



###
Piano_predict<-function(X, X_orig, y, alpha, beta0, Tree_list,stepsize){
  pred_lasso = as.matrix(X)%*%alpha+beta0
  pred_tree = rep(0,length(y))
  Tree_pred = matrix(0,nrow = length(y),ncol = iter)

  for(i in 1:length(Tree_list)){
    Tree_pred[,i] = predict(Tree_list[[i]],as.matrix(X_orig))
    #Predicted = c()
    #for(n in 1:length(y)){
    #  Predicted= c(Predicted,stepsize*Predict_tree(Tree_list[[i]],X_orig[n,]))
    #}
    pred_tree = pred_tree - stepsize*Tree_pred[,i]
  }
  #print(cbind(pred_tree, pred_tree1))
  fx = pred_lasso + pred_tree
  pred_test_prob = pred_prob(fx)
  lasso_prob = pred_prob(pred_lasso)
  auc = auc(roc(y, pred_test_prob$prob$prob_positive, smooth=FALSE, auc=TRUE, quiet = TRUE))
  auc_lasso<-auc(roc(y,lasso_prob$prob$prob_positive,smooth=FALSE, auc=TRUE, quiet = TRUE))
  #auc <- AUC(pred_test_prob$pred,y)
  #auc_lasso <- AUC(lasso_prob$pred,y)
  myloss<-log(1+exp((fx)*(-y)))
  loss_lasso <- log(1+exp((pred_lasso)*(-y)))
  return(list("auc"=auc, "auc_lasso" = auc_lasso, "loss" = myloss, "loss_lasso"=loss_lasso))
}

k=5
save.image(file="functions.RData")

