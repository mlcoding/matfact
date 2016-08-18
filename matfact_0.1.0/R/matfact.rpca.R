#----------------------------------------------------------------------------------#
# Package: matfact                                                                 #
# matfact.rpca(): The user interface for matfact.rpca()                            #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 1st, 2016                                                              #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

matfact.rpca <- function(Y,
                         U = NULL,
                         V = NULL,
                         rank = 1,
                         lambda = NULL,
                         nlambda = NULL,
                         rho = NULL,
                         prec = 1e-4,
                         max.ite = 1e4,
                         verbose = TRUE)
{
  begt=Sys.time()
  if(verbose)
    cat("Robust Principle Component Analysis. \n")
  
  if(is.null(rho))
    rho = 0.2
  
  m = dim(Y)[1]
  n = dim(Y)[2]
  mk = m*k
  nk = n*k
  if(is.null(U)) U = matrix(rnorm(m*k,mean=0,sd=1), m, k)
  if(is.null(V)) V = matrix(rnorm(n*k,mean=0,sd=1), n, k)
  Imk = diag(mk)
  Ink = diag(nk)
  est = list()
  est$U = list()
  est$V = list()
  est$S = list()
  
  for(ilamb in 1:nlambda){
    ilambda = lambda[ilamb]
    
    ite = 0
    gap = 1
    obj = rep(0,max.ite)
    while(ite<max.ite && gap>prec){
      # update U
      U2 = (Y - S)%*%V%*%solve(t(V)%*%V + ilambda*diag(k))
      
      # update U
      V2 = t(Y - S)%*%U2%*%solve(t(U2)%*%U2 + ilambda*diag(k))
      
      # update S
      S.tmp = Y-U2%*%t(V2)
      idx1 = which(abs(S.tmp)<rho)
      S.tmp[idx1] = 0
      S2 = sign(S.tmp)*(abs(S.tmp)-rho)
      
      gap = max(sum(sum((V-V2)^2)),sum(sum((U-U2)^2)),sum(sum(S-S2)^2))
      V = V2
      U = U2
      S = S2
      ite = ite+1
      
      # obj[ite] = sum(sum((Y-U%*%t(V)-S)^2))/2+ilambda*((sum(sum(U^2)))+(sum(sum(V^2))))/2+rho*sum(sum(abs(S)))
      # cat("ite=",ite,",gap=",gap,",dif=",norm(M-U%*%t(V)),"\n",sep="")
    }
    est$U[[ilamb]] = U
    est$V[[ilamb]] = V
    est$S[[ilamb]] = S
  }
  est$verbose = verbose
  # est$runtime = runt
  class(est) = "matfact"
  return(est)
}


