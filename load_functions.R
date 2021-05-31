library(xgboost)
library(MASS)
library(splines)
library(gglasso)
library(fastDummies)
library(pROC)

#---------------Regression------------------------------####
my_error<-function(pred, true){#mse/variance, which is 1-R^2
  sum((true-pred)^2)/sum((true-mean(true))^2)
}

#Train
GAMtree_fit<-function(X, y,lasso_group, X_orig, lambda1,lambda2, iter, stepsize, eta, nrounds){
  y_orig = y
  Betas = matrix(, nrow = ncol(X)+1, ncol =iter)
  Trees =  vector("list", iter)
  rrMSE_fit = matrix(, nrow = 2, ncol =iter)
  GAM_pred = matrix(0,nrow=length(y), ncol = iter)
  Tree_pred = matrix(0,nrow = length(y),ncol = iter)
  for(i in 1:iter){
    print(i)
    data=data.frame(cbind(X,y))
    
    #--------------Lasso-----------------###
    model = gglasso(X,y,lasso_group,loss="ls",lambda =lambda1)
    Betas[,i] = coef(model)
    GAM_pred[,i] = (as.matrix(X)%*%as.vector(Betas[2:(ncol(X)+1),i]))+Betas[1,i]
    res1 = y-GAM_pred[,i]
    rrMSE_fit[1,i] = my_error(rowSums(GAM_pred),y_orig)
    
    #--------------XGboost-----------------###
    dtrain <- xgb.DMatrix(data = as.matrix(X_orig),label = res1) 
    params <- list(booster = "gbtree", objective = "reg:squarederror", eta=eta, lambda = lambda2,max_depth=6)
    Trees[[i]] <- xgb.train (params = params, data = dtrain, nrounds = nrounds,eval_metric = "rmse")
    Tree_pred[,i] = stepsize*(predict(Trees[[i]],dtrain))

    ############################
    y = res1 - Tree_pred[,i]
    rrMSE_fit[2,i] = my_error((rowSums(Tree_pred)+rowSums(GAM_pred)),y_orig)
    if(i>1){
      if(rrMSE_fit[2,i]>=rrMSE_fit[2,i-1]){
        best_iter = i-1
        break
      }else{
        best_iter = i
      }
    }
  }
  result = list(Betas, Trees, rrMSE_fit, GAM_pred, Tree_pred, best_iter)
}


#Predict
GAMtree_predict<-function(X,X_orig,Betas,Trees,stepsize){
  G_part = (X%*%Betas[2:(ncol(X)+1)])+Betas[1]
  T_part=matrix(0,nrow = nrow(X), ncol =length(Trees))
  
  for(i in 1:length(Trees)){
    T_part[,i]= stepsize*(predict(Trees[[i]],as.matrix(X_orig)))
  }
  list(rowSums(T_part)+G_part, G_part, T_part)
}

sparsity_count<-function(Betas,lasso_group){
  count = 0
  
  for (i in c(1:lasso_group[length(lasso_group)])){
    value =  Betas[(which(lasso_group == i)+1)]
    if (sum(value) != 0){
      count=count + 1 #number of nonzero betas
    }
  }
  count
}

#-----------------Classification----------------------####
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

  Tree_list = list()
  Tree_pred = matrix(0, nrow = length(train_y),ncol = iter)
  for(i in 1:iter){
    print(paste("i=",i))
    ptm1<-proc.time()
    for (r in 1: nrounds){
      for (j in 1:p){
        w_vec = as.matrix(train_X)%*%alpha
        grad_a = Gradient_alpha(n, w_vec, b_vec, beta0, train_y, train_X[,((j-1)*k+1): (j*k)])
        alpha_j = alpha[((j-1)*k+1): (j*k)]
        group_ind = relu(1-(lambda1/sqrt(sum((-grad_a+(1/d_j[j])*alpha_j)^2))))
        alpha[((j-1)*k+1): (j*k)] = d_j[j]*(-grad_a+(1/d_j[j])*alpha_j)*group_ind
        
      }
      
      if(is.null(d_c)==FALSE){
        n_dummy_sum = 0
        for(d in 1:q){
          w_vec = as.matrix(train_X)%*%alpha
          n_dummy_sum = n_dummy_sum+n_dummy[d]
          grad_a = Gradient_alpha(n, w_vec, b_vec, beta0, train_y, train_X[,(nc_spline+n_dummy_sum-n_dummy[d]+1):(nc_spline+n_dummy_sum)])
          alpha_d = alpha[(nc_spline+n_dummy_sum-n_dummy[d]+1):(nc_spline+n_dummy_sum)]
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
    ############
    #Fit One Tree here
    print("Fit Tree begin start")
    ptm <- proc.time()
    gi = Gi(train_y,w_vec,b_vec,beta0)
    
    dtrain <- xgb.DMatrix(data = as.matrix(train_X_orig),label = gi)
    params <- list(booster = "gbtree", objective = "reg:squarederror", eta=eta, lambda = lambda2, gamma = lambda2,max_depth=6)
    Tree_list[[i]] <- xgb.train (params = params, data = dtrain, nrounds = tree_nrounds, eval_metric = "rmse")
    leaves = length(xgb.dump(Tree_list[[i]],with_stats=TRUE))
    if (leaves != (2*tree_nrounds)){
      Tree_pred[,i] = predict(Tree_list[[i]],dtrain)
      
      ##########
      ptm4 <- proc.time()
      print(paste("Tree fit and predict:",ptm4[3]-ptm[3],sep=""))
      
      b_vec = b_vec - stepsize*Tree_pred[,i]
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

    pred_tree = pred_tree - stepsize*Tree_pred[,i]
  }

  fx = pred_lasso + pred_tree
  pred_test_prob = pred_prob(fx)
  lasso_prob = pred_prob(pred_lasso)
  auc = auc(roc(y, pred_test_prob$prob$prob_positive, smooth=FALSE, auc=TRUE, quiet = TRUE))
  auc_lasso<-auc(roc(y,lasso_prob$prob$prob_positive,smooth=FALSE, auc=TRUE, quiet = TRUE))

  myloss<-log(1+exp((fx)*(-y)))
  loss_lasso <- log(1+exp((pred_lasso)*(-y)))
  return(list("auc"=auc, "auc_lasso" = auc_lasso, "loss" = myloss, "loss_lasso"=loss_lasso))
}

k=5
save.image(file="functions.RData")