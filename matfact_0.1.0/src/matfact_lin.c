# include <stdio.h>
# include "matrixAlgebra.h"
//# include "mat.h"

# define VAL_FREQ 10				// Validation frequency.
# define ERROR_THRESHOLD 0.0001		// Error Threshold to quit Alternating Minimization if error on validation set is below a minimal threshold.
													// Desired low rank

// Calculates the least squared loss
double leastSquaredLoss(matrix *U, matrix *V, tensor *X, matrix *Y)
{
	int i,N,m,n,k;
	N = X->n_samples;
	m = X->n_rows;
	n = X->n_cols;
	k = U->n_cols;

	matrix *x, *x_T, *UVX;
	double y, y_pred;
	double obj = 0;
	matrix *V_T = transpose(V);									// New Memory
	matrix *UV = matrixMultiply(U, V_T);						// New Memory
	freeMatrix(V_T);

	for(i=0; i<N; i++)
	{
		x = getMatrix(X, i);
		y = getM(Y, i, 0);
		x_T = transpose(x);										// New Memory
		UVX = matrixMultiply(UV,x_T);							// New Memory
		y_pred = trace(UVX);
		obj = obj + pow((y - y_pred),2);

		freeMatrix(x_T);
		freeMatrix(UVX);
	}
	freeMatrix(UV);

	obj = obj/(2 * (double) N);
	return obj;
}


void matfact_lin(double *XX_train, double * YY_train, double * UU, double * VV, int * mm, int * nn, int * NN, double *llambda, int *nnlambda, int *rank, double *pprec, int *mmax_ite)
{
	// To generate random numbers.
	time_t tm;								
	srand((unsigned) time(&tm));											

	int i,j,t,m,n,k,N,nlambda,max_ite;
    double prec,lambda;

	t = 1;
	double obj = 0;
    
    m = *mm;																// Number of rows
	n = *nn;																// Number of columns
    k = *rank;
    N = *NN;
    nlambda = *nnlambda;
    lambda = llambda[0];
    max_ite = *mmax_ite;
    prec = *pprec;
    

	int mk = m*k;
	int nk = n*k;
    
    tensor* X_train = emptyTensor();
    matrix* Y_train = emptyMatrix();
    matrix *U = newMatrix(m, k);
    matrix *V = newMatrix(n, k);
	matrix *vec_U = vec(U);
	matrix *vec_V = vec(V); 

	matrix *x;
	double y, obj_i;

	matrix *x_T, *XV, *XV_vec, *XV_vec_T, *XU, *XU_vec, *XU_vec_T;		
	matrix *sum_term1, *sum_term2;						
	matrix *temp_term1, *temp_term2;														// Mostly temporary matrices used later.
				
	randomizeMatrix(U);																		// Elements of U sampled from standard normal.
	randomizeMatrix(V);																		// Elements of V sampled from standard normal.

	matrix* const_term_trainU = scalarMultiply(N * lambda, eye(mk));				
	matrix* const_term_trainV = scalarMultiply(N * lambda, eye(nk));

	obj_i = leastSquaredLoss(U, V, X_train, Y_train);
	obj_i = obj_i + (lambda/2)*(pow(frobNorm(U),2) + pow(frobNorm(V),2));				// Calculated Iniitial Objective
	//printf("Training: Initial Objective = %f\n",obj_i);

	// Alternating Minimization Starts
	while(t<=max_ite)
	 {
		//printf("T = %d\n",t);

		// Compute U
		sum_term1 = newMatrix(mk, mk);
		sum_term2 = newMatrix(mk, 1);
		for(i=0; i<N; i++)
		{
			x = getMatrix(X_train,i); 														// No New Memory
			y = getM(Y_train, i, 0);

			XV = matrixMultiply(x,V);														// New Memory
			XV_vec = vec(XV);																// No New Memory
			XV_vec_T = transpose(XV_vec); 													// New Memory
			temp_term1 = matrixMultiply(XV_vec,XV_vec_T);									// New Memory

			temp_term2 = scalarMultiply(y,XV_vec);											// New Memory

			matrixAddInPlace(sum_term1,temp_term1);	// New Memory
			matrixAddInPlace(sum_term2,temp_term2);	// New Memory

			freeMatrix(XV);
			freeMatrix(XV_vec_T);
			freeMatrix(temp_term1);
			freeMatrix(temp_term2);
       }
       matrixAddInPlace(sum_term1, const_term_trainU);			// New Memory
       inverse(sum_term1);							
       vec_U = matrixMultiply(sum_term1, sum_term2);			// Dont freeMatrix
       freeMatrix(U);
       U = unvec(vec_U, m, k);

       freeMatrix(sum_term1);
       freeMatrix(sum_term2);

   		// Compute V
       sum_term1 = newMatrix(nk, nk);
       sum_term2 = newMatrix(nk, 1);

       for(i=0; i<N; i++)
       {
			x = getMatrix(X_train,i); 								// No New Memory
			y = getM(Y_train, i, 0);

			x_T = transpose(x);										// New Memory
			XU = matrixMultiply(x_T,U);								// New Memory
			XU_vec = vec(XU);										// No New Memory
			XU_vec_T = transpose(XU_vec);							// New Memory

			temp_term1 = matrixMultiply(XU_vec,XU_vec_T);			// New Memory
			temp_term2 = scalarMultiply(y,XU_vec);					// New Memory
			matrixAddInPlace(sum_term1,temp_term1);					// New Memory
			matrixAddInPlace(sum_term2,temp_term2);					// New Memory

			freeMatrix(x_T);
			freeMatrix(XU);
			freeMatrix(XU_vec_T);
			freeMatrix(temp_term1);
			freeMatrix(temp_term2);
		}
		matrixAddInPlace(sum_term1, const_term_trainV);				// New Memory
		inverse(sum_term1);											// New Memory
		vec_V = matrixMultiply(sum_term1, sum_term2);				// No new Memory
		freeMatrix(V);
		V = unvec(vec_V, n, k);

		freeMatrix(sum_term1);
		freeMatrix(sum_term2);
      
      	// Calculate objective/Validate.
		if(t % VAL_FREQ == 0)
		{
			// Calculate Objective on Training set.
			obj = leastSquaredLoss(U, V, X_train, Y_train);
			obj = obj + (lambda/2)*(pow(frobNorm(U),2) + pow(frobNorm(V),2));
			//printf("Training: Objective = %f\n",obj);
			if(obj < prec)
				break;
		}
		t++;
	}

	freeMatrix(const_term_trainU);
	freeMatrix(const_term_trainV);

	// Free memory allocated to Data.
	// freeTensor(X_train);
	// freeMatrix(Y_train);
	// freeTensor(X_val);
	// freeMatrix(Y_val);
	// freeTensor(X_test);
	// freeMatrix(Y_test);

	freeMatrix(U);
	freeMatrix(V);

}
