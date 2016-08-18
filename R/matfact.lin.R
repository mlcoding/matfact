#----------------------------------------------------------------------------------#
# Package: matfact                                                                 #
# matfact.lin(): The user interface for matfact.lin()                              #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 1st, 2016                                                              #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

matfact.lin <- function(X, 
                        Y,
                        U = NULL,
                        V = NULL,
                        rank = 1,
                        lambda = NULL,
                        nlambda = NULL,
                        prec = 1e-4,
                        max.ite = 1e4,
                        verbose = TRUE)
{
  begt=Sys.time()
  if(verbose)
    cat("Matrix Linear Regression. \n")
  
  m = dim(X)[1]
  n = dim(X)[2]
  N = dim(X)[3]
  k = rank
  mk = m*k
  nk = n*k
  if(is.null(U)) U = matrix(rnorm(m*k,mean=0,sd=1), m, k)
  if(is.null(V)) V = matrix(rnorm(n*k,mean=0,sd=1), n, k)
  Imk = diag(mk)
  Ink = diag(nk)
  est = list()
  est$U = list()
  est$V = list()
  
  for(ilamb in 1:nlambda){
    ilambda = lambda[ilamb]
    
    ite = 0
    gap = 1
    obj = rep(0,max.ite)
    while(ite<max.ite && gap>prec){
      # update U
      sum_term1 = matrix(0, nrow=mk, ncol = mk)
      sum_term2 = matrix(0, nrow=mk, ncol = 1)
      for (i in 1:N){
        vec.XV = matrix(X[,,i]%*%V, nrow = mk)
        sum_term1 = sum_term1 + vec.XV%*%t(vec.XV)
        sum_term2 = sum_term2 + Y[i]*vec.XV
      }
      vec.U = solve(N*ilambda*Imk + sum_term1, sum_term2)
      U2 = matrix(vec.U, nrow=m, ncol=k)
      
      # update V
      sum_term1 = matrix(0, nrow=nk, ncol = nk)
      sum_term2 = matrix(0, nrow=nk, ncol = 1)
      for (i in 1:N){
        vec.XU = matrix(t(X[,,i])%*%U2, nrow = nk)
        sum_term1 = sum_term1 + vec.XU%*%t(vec.XU)
        sum_term2 = sum_term2 + Y[i]*vec.XU
      }
      vec.V = solve(N*ilambda*Ink + sum_term1, sum_term2)
      V2 = matrix(vec.V, nrow=n, ncol=k)
      
      gap = max(sum(sum((V-V2)^2)),sum(sum((U-U2)^2)))
      V = V2
      U = U2
      ite = ite+1
      
      # obj[ite] = 0
      # for(i in 1:N){
      #   obj[ite] = obj[ite] + (sum(diag(M%*%t(X[,,i])))-Y[i])^2
      # }
      # obj[ite] = obj[ite]/N+ (sum(sum(U^2))+sum(sum(V^2)))/ilambda
      # cat("ite=",ite,",gap=",gap,",dif=",norm(M-U%*%t(V)),"\n",sep="")
    }
    # M.hat = U%*%t(V)
    
    # est = .C("matfact_lin", as.double(X), as.double(Y), as.double(U), as.double(V), 
    #          as.integer(m), as.integer(n), as.double(lanbda), as.integer(nlambda), 
    #          as.integer(rank), as.double(prec), as.double(max.ite), PACKAGE="matfact")
    est$U[[ilamb]] = U
    est$V[[ilamb]] = V
  }
  est$verbose = verbose
  # est$runtime = runt
  class(est) = "matfact"
  return(est)
}


