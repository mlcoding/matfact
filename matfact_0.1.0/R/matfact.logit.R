#----------------------------------------------------------------------------------#
# Package: matfact                                                                 #
# matfact.logit(): The user interface for matfact.lopgit()                         #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 1st, 2016                                                              #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

matfact.logit <- function(X, 
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
    cat("Matrix Logtistic Regression. \n")
  
  m = dim(X)[1]
  n = dim(X)[2]
  N = dim(X)[3]
  mk = m*k
  nk = n*k
  if(is.null(U)) U = matrix(rnorm(m*k,mean=0,sd=1), m, k)
  if(is.null(V)) V = matrix(rnorm(n*k,mean=0,sd=1), n, k)
  U2 = U
  V2 = V
  vec_U = matrix(U, nrow=mk, ncol = 1)
  vec_V = matrix(V, nrow=nk, ncol = 1)
  vec_U2 = vec_U
  vec_V2 = vec_V
  Imk = diag(mk)
  Ink = diag(nk)
  est = list()
  est$U = list()
  est$V = list()
  
  for(ilamb in 1:nlambda){
    ilambda = lambda[ilamb]
    
    ite = 0
    gap = 1
    alpha = 0.01
    obj = rep(0,max.ite)
    true.err = rep(0,max.ite)
    class.err = rep(0,max.ite)
    while(ite<max.ite && gap>prec){
      ite = ite+1
      sum_term_U = matrix(0, mk, 1)
      sum_term_V = matrix(0, nk, 1)
      sum_term_UU = matrix(0, mk, mk)
      sum_term_VV = matrix(0, nk, nk)
      sum_term_VU = matrix(0, mk, nk)
      sum_term_UV = matrix(0, nk, mk)
      for(i in 1:N){
        x = X[,,i]
        y = Y[i]
        v_xV = matrix(x%*%V2, nrow=mk, ncol = 1)
        v_xU = matrix(t(X[,,i])%*%U2, nrow=nk, ncol = 1)
        exponent = -y*sum(diag(M%*%t(X[,,i])))
        exp_term = (y * exp(exponent)) / (1 + exp(exponent));
        hess_exp_term = y /(1 + exp(exponent));
        sum_term_U = sum_term_U + exp_term * v_xV;
        sum_term_V = sum_term_V + exp_term * v_xU;
        
        sum_term_UU = sum_term_UU + exp_term * hess_exp_term * v_xV %*% t(v_xV)
        sum_term_VV = sum_term_VV + exp_term * hess_exp_term * v_xU %*% t(v_xU)
        
        sum_term_VU = sum_term_VU + exp_term * (hess_exp_term * v_xV %*% t(v_xU) - kronecker(diag(k),x))
        sum_term_UV = sum_term_UV + exp_term * (hess_exp_term * v_xU %*% t(v_xV) - kronecker(diag(k),t(x)))
      }
      
      grad_U = ilambda*matrix(U2, nrow=mk, ncol = 1) - sum_term_U/N
      grad_V = ilambda*matrix(V2, nrow=nk, ncol = 1) - sum_term_V/N
      
      hess_UU = ilambda*diag(mk) + sum_term_UU/N
      hess_VV = ilambda*diag(nk) + sum_term_VV/N
      hess_UV = sum_term_UV/N
      hess_VU = sum_term_VU/N
      vec_V_old = vec_V
      alpha_opt = alpha/sqrt(ite)
      
      vec_U = vec_U2 -alpha_opt * solve(hess_UU)%*%(hess_VU%*%(vec_V-vec_V2) + grad_U)
      vec_V = vec_V2 - alpha_opt * solve(hess_VV)%*%(hess_UV%*%(vec_U-vec_U2) + grad_V)
      
      U = matrix(vec_U,nrow=m, ncol=k)
      V = matrix(vec_V,nrow=n, ncol=k)
      
      gap = max(sum(sum((V-V2)^2)),sum(sum((U-U2)^2)))
      vec_V2 = vec_V_old
      vec_U2 = vec_U
      U2 = matrix(vec_U2,nrow=m, ncol=k)
      V2 = matrix(vec_V2,nrow=n, ncol=k)
      
      # obj[ite] = 0
      # true.err[ite] = 0
      # class.err[ite] = 0
      # for(i in 1:N){
      #   obj[ite] = obj[ite] + log(1+exp(-Y[i]*sum(diag(M%*%t(X[,,i])))))
      #   true.err[ite] = true.err[ite]+abs((sign(1/(1+exp(-sum(diag(M%*%t(X[,,i])))))-0.5)+1)/2-Y[i])
      #   class.err[ite] = class.err[ite]+abs((sign(1/(1+exp(-sum(diag(U2%*%t(V2)%*%t(X[,,i])))))-0.5)+1)/2-Y[i])
      # }
      # obj[ite] = obj[ite]/N+ (sum(sum(U^2))+sum(sum(V^2)))/ilambda
      # class.err[ite] = class.err[ite]/N
      # true.err[ite] = true.err[ite]/N
      # cat("ite=",ite,",gap=",gap,",dif=",norm(M-U%*%t(V)),",true.err=",true.err[ite],",class.err=",class.err[ite],"\n",sep="")
    }
    est$U[[ilamb]] = U
    est$V[[ilamb]] = V
  }
  est$verbose = verbose
  # est$runtime = runt
  class(est) = "matfact"
  return(est)
}


