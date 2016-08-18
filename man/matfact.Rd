\name{matfact}
\alias{matfact}

\title{
Matrix Factorization Modeling
}

\description{
The function "matfact" implements the user interface.
}

\usage{
matfact(X, Y, U = NULL, V = NULL, rank = NULL, lambda = NULL,
        nlambda = NULL, lambda.min.ratio = NULL, rho = NULL,
        method = "linear", prec = 1e-4, max.ite = 1e3, verbose = TRUE)
}

\arguments{
  \item{X}{A data array, which contains \code{N} sensing matrices with \code{nrow=m} and \code{ncol=n}. }
  \item{Y}{Reponse data. For low rank matrix linear regression \code{method="linear"} low rank matrix logistic regression \code{method="logit"}, \code{Y} is a vector of size \code{N}. For robust principle component analysis \code{method="rpca"}, \code{Y} is a matrix of size with \code{nrow=m} and \code{ncol=n}. }
  \item{U}{First matrix factor component with \code{nrow=m} and \code{ncol=k}. }
  \item{V}{Second matrix factor component with \code{nrow=n} and \code{ncol=k}. }
  \item{rank}{An estimate of the rank of the matrix. }
  \item{lambda}{A seqeunce of regularization parameters.}
  \item{nlambda}{The number of values used in \code{lambda}. Default value is 10.}
  \item{lambda.min.ratio}{The smallest value for \code{lambda}, as a fraction of the uppperbound (\code{MAX}) of the regularization parameter. The program can automatically generate \code{lambda} as a sequence of length = \code{nlambda} starting from \code{MAX} to \code{lambda.min.ratio*MAX} in log scale. The default value is \code{0.2}.}
  \item{rho}{Regularization parameter for the sparse component, only used in the robust principle component analysis. The default value is \code{0.2}.}
  \item{method}{Options for model. Low rank matrix linear regression is applied if \code{method="linear"}, low rank matrix logistic regression is applied \code{method="logit"}, and robust principle component analysis is applied if \code{method="rpca"}. The default value is \code{method="linear"}.}
  \item{prec}{Stopping precision. The default value is 1e-4.}
  \item{max.ite}{The iteration limit. The default value is 1e4.}
  \item{verbose}{Tracing information is disabled if \code{verbose = FALSE}. The default value is \code{TRUE}.}
  }
\details{
For low rank matrix linear regression,  \eqn{y_i = \langle {UV^T}, {X_i} \rangle + \epsilon_i}, where \eqn{\epsilon_i} is the noise in the model for all \eqn{i=1,\ldots,N}, we solve

\deqn{
    \min_{U,V} {f({U,V})} = \frac{1}{N} \sum_{i=1}^{N} ({y_i - \langle {UV^T},{X_i} \rangle})^2 + \frac{\lambda}{2}\|{U}\|^2_F + \frac{\lambda}{2}\\{V}\|^2_F,
    }\cr
    where \eqn{R(\beta)} can be \eqn{\ell_1} norm, MCP, SCAD regularizers.
   
For sparse logistic regression,

\deqn{
    \min_{\beta} {\frac{1}{n}}\sum_{i=1}^n (\log(1+e^{x_i^T \beta}) - y_i x_i^T \beta) + \lambda R(\beta),
    }\cr
    where \eqn{R(\beta)} can be \eqn{\ell_1} norm, MCP, and SCAD regularizers.
    
For sparse column inverse operation,
\deqn{
    \min_{\beta} {\frac{1}{2}} \beta^T S \beta - e^T \beta + \lambda R(\beta),
    }\cr
    where \eqn{R(\beta)} can be \eqn{\ell_1} norm, MCP or SCAD  regularizers.

}

\value{
An object with S3 classes \code{"matfact"} is returned: 
  \item{U,V}{
  Arrays of matrix factor components. Each constains \code{N} estimates. 
  }
  \item{verbose}{The \code{verbose} from the input.}
}

\author{
Xingguo Li, Enayat Ullah and Tuo Zhao \cr
Maintainer: Xingguo Li <xingguo.leo@gmail.com>
}


\seealso{
\code{\link{matfact-package}}.
}

\examples{
################################################################
## Low rank matrix linear regression
m=20
n=30
k = 3
N = (m+n)*k*5

X = array(data = rnorm(m*n*N,mean=0,sd=1), dim = c(m,n,N))
M = matrix(rnorm(m*k,mean=0,sd=1), m, k)\%*\%matrix(rnorm(n*k,mean=0,sd=1), k, n)
Y = matrix(0,nrow=N, ncol=1)
for(i in 1:N){
  Y[i] = sum(diag(M\%*\%t(X[,,i])))
}

out1 = matfact(X, Y, rank = k, method = "linear")

}