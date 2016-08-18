#----------------------------------------------------------------------------------#
# Package: matfact                                                                 #
# matfact(): The user interface for matfact()                                      #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 1st, 2016                                                              #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

matfact <- function(X, 
                    Y, 
                    U = NULL,
                    V = NULL,
                    rank = NULL, 
                    lambda = NULL,
                    nlambda = NULL,
                    lambda.min.ratio = NULL,
                    rho = NULL,
                    method = "linear",
                    prec = 1e-4,
                    max.ite = 1e3,
                    verbose = TRUE)
{
  if(method!="linear" && method!="logit" && method!="spca"){
    cat(" Wrong \"method\" input. \n \"method\" should be one of \"linear\", \"logit\" and \"spca\".\n", 
        method,"does not exist. \n")
    return(NULL)
  }
  if(is.null(rank)){
    cat(" Provide a value of the rank \n")
    return(NULL)
  }
    
  if(!is.null(lambda)) nlambda = length(lambda)
  if(is.null(lambda)){
    if(is.null(nlambda))
      nlambda = 10
    lambda.max = 0.5
    if(is.null(lambda.min.ratio)){
      lambda.min = 0.2*lambda.max
    }else{
      lambda.min = min(lambda.min.ratio*lambda.max, lambda.max)
    }
    if(lambda.min>=lambda.max) cat("\"lambda.min\" is too small. \n")
    lambda = exp(seq(log(lambda.max), log(lambda.min), length = nlambda))
  }
  if(method=="linear"){
    out = matfact.lin(X = X, Y=Y, U = U, V = V, rank = rank, lambda = lambda, nlambda = nlambda,
                      prec = prec, max.ite = max.ite, verbose = verbose)
  }
  if(method=="logit"){
    out = matfact.logit(X = X, Y=Y, U = U, V = V, rank = rank, lambda = lambda, nlambda = nlambda,
                        prec = prec, max.ite = max.ite, verbose = verbose)
  }
  if(method=="rpca"){
    out = matfact.rpca(Y=Y, U = U, V = V, rank = rank, lambda = lambda, nlambda = nlambda,
                       rho=rho,prec = prec, max.ite = max.ite, verbose = verbose)
  }
  out$method = method
  return(out)
}
