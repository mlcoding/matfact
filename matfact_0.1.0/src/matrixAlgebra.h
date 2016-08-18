# include <stdlib.h>
# include <time.h>
# include <math.h>

// LU decomoposition of a general matrix
extern void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

// generate inverse of a matrix given its LU decomposition
extern void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);

typedef struct matrix matrix;
typedef struct tensor tensor;

// Matrix Structure defined.	
struct matrix {
	int n_rows;
	int n_cols;
	double *mat;

};

// Tensor Structure defined.	
struct tensor {
	int n_samples;
	int n_rows;
	int n_cols;
	double *mat;
};

// Creates an emoty matrix (no memory allocated for the array)
matrix* emptyMatrix() { 
  matrix* M = malloc(sizeof(matrix));
  return M;
}

// Creates an emoty tensor (no memory allocated for the array)
tensor* emptyTensor() { 
  tensor* T = malloc(sizeof(tensor));
  return T;
}

// Function to create a new matrix (with memory allocation for the array).
matrix* newMatrix(int m, int n) { 
  matrix* M = malloc(sizeof(matrix));
  M->n_rows = m;
  M->n_cols = n;
  M->mat = (double*) calloc(m*n,sizeof(double));
  return M;
}

// Function to create a new Tensor (with memory allocation for the array).
tensor* newTensor(int N, int m, int n) { 
  tensor* T = malloc(sizeof(tensor));
  T->n_samples = N;
  T->n_rows = m;
  T->n_cols = n;
  T->mat = (double*) calloc(N*m*n, sizeof(double));
  return T;
}

// Allocates an (m x n) double memory space for matrix M.
void allocateM(matrix *M)
{
	int m,n;
	m = M->n_rows;
 	n= M->n_cols;
 	M->mat = (double*) calloc(m*n,sizeof(double));
}

// Allocates an (N x m x n) double memory space for matrix M.
void allocateT(tensor *T)
{
	int N,m,n;
	N = T->n_samples;
	m = T->n_rows;
 	n = T->n_cols;
 	T->mat = (double*) calloc(N*m*n,sizeof(double));
}

// Reshapes matrix M into a (m x n) matrix
void reshapeM(matrix *M,int m,int n)
{
	M->n_rows = m;
 	M->n_cols = n;
}

// Reshapes Tensor T into a (N x m x n) tensor
void reshapeT(tensor *T,int N,int m,int n)
{
	T->n_samples = N;
	T->n_rows = m;
 	T->n_cols = n;
}

// Samples from a 1-d standard normal distribution
double sampleNormal() {
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return sampleNormal();
    double c = sqrt(-2 * log(r) / r);
    return u * c;
}


// Sets element M[i,j] = value
void setM(matrix *M, int i, int j, double value)
{
	M->mat[i + j*M->n_rows] = value;
}

// Retrieves element M[i,j]
double getM(matrix *M, int i, int j)
{
	return M->mat[i + j * M->n_rows];
}

// Retrieves element M[n,i,j]
double getT(tensor *T, int n, int i, int j)
{
	return T->mat[n*(T->n_rows * T->n_cols) + i + j*T->n_rows];
}

// Retrives n^th element of tensor T (which is an (m x n) matrix)
matrix* getMatrix(tensor *T, int n)
{
	int n_rows = T->n_rows;
	int n_cols = T->n_cols;
	matrix* M = malloc(sizeof(matrix));
	M->n_rows = n_rows;
	M->n_cols = n_cols;
	M->mat = &(T->mat[(n_rows*n_cols)* n]);
	return M;
}

// Returns a matrix with the array poiting at M's array
matrix* softCopy(matrix *M)	
{
	matrix* copy = emptyMatrix();
	int n_rows = M->n_rows;
	int n_cols = M->n_cols;

	copy->n_rows = n_rows;
	copy->n_cols = n_cols;
	copy->mat = M->mat;

	return copy;
}

// Returns a matrix copying elements of M in new allocated memory
matrix* hardCopy(matrix *M)
{
	int i;
	int n_rows = M->n_rows;
	int n_cols = M->n_cols;
	matrix* copy = newMatrix(n_rows, n_cols);
	int total_elems = n_rows * n_cols;

	copy->n_rows = n_rows;
	copy->n_cols = n_cols;
	for(i = 0; i< total_elems; i++)
		copy->mat[i] = M->mat[i];
	return copy;
}

// Given two matrices M1 and M2. It copies elements of M1 into M2
void internalCopy(matrix *M1, matrix *M2)
{
	int i;
	
	int n_rows = M1->n_rows;
	int n_cols = M1->n_cols;
	int total_elems = n_rows * n_cols;
	for(i = 0; i< total_elems; i++)
		M2->mat[i] = M1->mat[i];
}

// Returns a matrix with elements of M sampled from a standard normal distribution
void randomizeMatrix(matrix* M)
{
	int i,j,n_rows,n_cols;
	n_rows = M->n_rows;
	n_cols = M->n_cols;
	for(i=0;i<n_rows;i++)
	{
		for(j=0;j<n_cols;j++)
			setM(M,i,j,sampleNormal());
	}
}

// calculate the cofactor of element (row,col)
void getMinor(matrix* src, matrix* dest, int row, int col)
{
	int i,j,order;
	order = src->n_rows;
    // indicate which col and row is being copied to dest
    int colCount=0,rowCount=0;
 
    for(i = 0; i < order; i++ )
    {
        if( i != row )
        {
            colCount = 0;
            for(j = 0; j < order; j++ )
            {
                // when j is not the element
                if( j != col )
                {
                    setM(dest,rowCount,colCount, getM(src,i,j));
                    colCount++;
                }
            }
            rowCount++;
        }
    }
}

// Calculates the kronecker product of two matrices A and B
matrix* kron(matrix *A, matrix *B)
{
	int i,j,k,l;
	int n_rows_A = A->n_rows;
	int n_cols_A = A->n_cols;
	int n_rows_B = B->n_rows;
	int n_cols_B = B->n_cols;

	int n_rows = n_rows_A * n_rows_B;
	int n_cols = n_cols_A * n_cols_B;

	matrix *result = newMatrix(n_rows, n_cols);


	for(j = 0; j<n_cols_A; j++)
	{
		for(l = 0; l<n_cols_B; l++)
		{
			for(i = 0; i<n_rows_A; i++)
			{
				for(k = 0; k<n_rows_B; k++)
				{
					setM(result, n_rows_B*i + k, n_cols_B*j+l, getM(B,k,l) * getM(A,i,j)); 
				}
		
			}	
		}
	}
	return result;
}

// frees the memory allocated for matrix M.
void freeMatrix(matrix* M)
{
	free(M->mat);
	free(M);
}

// frees the memory allocated for tensor T.
void freeTensor(tensor* T)
{
	free(T->mat);
	free(T);
}

// Calculate the determinant of matrix M.
double determinant(matrix* mat)
{
	int i,j;
	int order = mat->n_rows;
    // order must be >= 0
    // stop the recursion when matrix is a single element
    if( order == 1 )
        return getM(mat,0,0);
    // the determinant value
    double det = 0;
 
    // allocate the cofactor matrix
    // double **minor = createMatrix(order - 1, order -1);
    matrix* minor= newMatrix(order-1,order-1);
 
    for(i = 0; i < order; i++ )
    {
        // get minor of element (0,i)
        getMinor( mat, minor, 0, i);
        // the recusion is here!
 
        det += (i%2==1?-1.0:1.0) * getM(mat,0,i) * determinant(minor);
        //det += pow( -1.0, i ) * mat[0][i] * CalcDeterminant( minor,order-1 );
    }
 
    // release memory
    freeMatrix(minor);
    return det;
}

// Calcuctes the inverse of matrix M using the lapack library. 
void inverse(matrix* M)
{
	int N  = M->n_rows;
	double *A = M->mat;
    int *IPIV = (int *)malloc((N+1) * sizeof(int));
    int LWORK = N*N;
    double *WORK = (double*)malloc(LWORK * sizeof(double));
    int INFO;

    dgetrf_(&N,&N,A,&N,IPIV,&INFO);
    dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

   free(IPIV);
   free(WORK);
}


// Calculates the inverse naively (using minors)
matrix* inverseUsingMinors(matrix* M)
{
	int i,j;
	int order = M->n_rows;
    // get the determinant of a
    double det = 1.0/determinant(M);
    matrix* inv = newMatrix(order, order);
    matrix* minor = newMatrix(order-1, order-1);
 
    for(j=0;j<order;j++)
    {
        for(i=0;i<order;i++)
        {
            // get the co-factor (matrix) of A(j,i)
            getMinor(M,minor,j,i);
            setM(inv,i,j,det*determinant(minor));
            if( (i+j)%2 == 1)
                setM(inv,i,j,-1*getM(inv,i,j));
        }
    }
    // release memory
    freeMatrix(minor);
    return inv;
}

// Returns an identity matrix of order k
matrix* eye(int k)
{
	int i;
	matrix *I = newMatrix(k,k);
	for(i = 0;i<k;i++)
		setM(I,i,i,1);
	return I;

}

// Vectorizes a matrix M  -- No new memory allocated.
matrix* vec(matrix* M)
{
	int i,j,n_rows,n_cols;
	n_rows = M->n_rows;
	n_cols = M->n_cols;

	matrix *vec_M = emptyMatrix();
	vec_M->n_rows = (n_rows*n_cols);
	vec_M->n_cols = 1;
	vec_M->mat = M->mat;
	return vec_M;
}

// Unvectorized a vectorized matrix M into a (m x n) matrix -- No new memory allocated.
matrix* unvec(matrix* vec_M, int m, int n)
{
	int i,j;
	matrix* M = emptyMatrix();
	M->n_rows = m;
	M->n_cols = n;
	M->mat = vec_M->mat; 
	return M;
}

// Prints the matrix 
void printMatrix(matrix* M)
{
	int i,j,n_rows,n_cols;
	n_rows = M->n_rows;
	n_cols = M->n_cols;

	for(i = 0; i<n_rows; i++)
	{
		for(j = 0; j<n_cols; j++)
		{
			printf("%f ",getM(M, i, j));
		}
		printf("\n");
	}
}

// Prints the tensor.
void printTensor(tensor* T)
{
	int i,j,k,n_samples,n_rows,n_cols;
	n_samples = T->n_samples;
	n_rows = T->n_rows;
	n_cols = T->n_cols;
	for(k =0;k<n_samples;k++)
	{	
		printf("\nTensor[%d]\n",k);
		for(i = 0; i<n_rows; i++)
		{
			for(j = 0; j<n_cols; j++)
			{
				printf("%f ",getT(T,k, i, j));
			}
			printf("\n");
		}
	}
}

// Returns the transpose of matrix M -- New memory allocated
matrix* transpose(matrix* M)
{
	int i,j,n_rows,n_cols;
	n_rows = M->n_rows;
	n_cols = M->n_cols;

	matrix* M_t = newMatrix(n_cols, n_rows);
	for(i=0; i<n_rows; i++)
	{
		for(j=0; j<n_cols; j++)
			setM(M_t,j,i,getM(M,i,j));
	}
	return M_t;
}

// Returns the trace of matrix M 
double trace(matrix *M)
{
	int i;
	int n_rows = M->n_rows;
	double sum = 0;
	for(i=0; i<n_rows; i++)
		sum = sum + getM(M, i, i);
	return sum;
}

// Creates a matrix of size (m x n), every element being 1
matrix* ones(int m, int n)
{
	int i,j;
	matrix *M = newMatrix(m,n);
	for(i = 0;i<m;i++)
	{
		for(j=0;j<n;j++)
			setM(M,i,j,1);
	}
	return M;
}

// Concatenates two matrices wlong their rows
matrix *rowCat(matrix *A, matrix *B)
{
	int i, j, k, n_rows, n_rows_A, n_rows_B, n_cols;
	if(!(A->n_cols == B->n_cols))
	{
		printf("Invalid matrices for row concatenation.\n");
		return 0;
	}
	n_rows_A = A->n_rows;
	n_rows_B = B->n_rows;
	n_rows = n_rows_A + n_rows_B;
	n_cols = A->n_cols;
	matrix *rCat = newMatrix(n_rows, n_cols);
	for(i=0; i<n_cols; i++)
	{
		for(j=0; j<n_rows_A; j++)
			setM(rCat, j, i, getM(A,j,i));
		for(k=0; j<n_rows; j++,k++)
			setM(rCat, j, i, getM(B,k,i));
	}
	return rCat;
}

// Concatenates two matrices wlong their columns
matrix *colCat(matrix *A, matrix *B)
{
	int i, j, n_rows, n_cols, n_cols_A, n_cols_B, total_elems, total_elems_A;
	if(!(A->n_rows == B->n_rows))
	{
		printf("Invalid matrices for column concatenation.\n");
		return 0;
	}
	n_rows = A->n_rows;
	n_cols_A = A->n_cols;
	n_cols_B = B->n_cols;
	n_cols = n_cols_A + n_cols_B;
	total_elems_A = n_rows*n_cols_A;
	total_elems = n_rows*n_cols;

	matrix *cCat = newMatrix(n_rows, n_cols);
	A->n_cols = n_cols;
	for(i=0; i<total_elems_A; i++)
		cCat->mat[i] = A->mat[i];
	for(j = i; j<total_elems; j++)
		cCat->mat[j] = B->mat[j-i];
		
	return cCat;
}

// Returns a matrix which is the sum of two matrices  -- New memory allocated.
matrix* matrixAdd(matrix* M1, matrix* M2)
{
	int i,j,n_rows,n_cols;
	n_rows = M1->n_rows;
	n_cols = M1->n_cols;

	matrix* M_sum = newMatrix(n_rows,n_cols);
	for(i=0; i<n_rows; i++)
	{
		for(j=0; j<n_cols; j++)
			setM(M_sum, i, j, getM(M1, i, j) + getM(M2, i, j));
	}
	return M_sum;
}

// Replaces elemnts of M1 with the sum of M1 and M2. ( M1 = M1 + M2) -- No new memory allocated
void matrixAddInPlace(matrix* M1, matrix* M2)	
{
	int i,j,n_rows,n_cols;
	double sum_term;
	n_rows = M1->n_rows;
	n_cols = M1->n_cols;

	for(i=0; i<n_rows; i++)
	{
		for(j=0; j<n_cols; j++)
		{
			sum_term = getM(M1, i, j) + getM(M2, i, j);
			setM(M1, i, j, sum_term);
		}
	}
}

// Returns a matrix which is the difference of two matrices  -- New memory allocated.
matrix* matrixSubtract(matrix* M1, matrix* M2)
{
	int i,j,n_rows,n_cols;
	n_rows = M1->n_rows;
	n_cols = M1->n_cols;

	matrix* M_diff = newMatrix(n_rows,n_cols);
	for(i=0; i<n_rows; i++)
	{
		for(j=0; j<n_cols; j++)
			setM(M_diff, i, j, getM(M1, i, j) - getM(M2, i, j));
	}
	return M_diff;
}


// Multiples each elements of matrix m with c. -- New memory allocated
matrix* scalarMultiply(double c, matrix* M)
{
	int i,j,n_rows,n_cols;
	n_rows = M->n_rows;
	n_cols = M->n_cols;
	matrix* result = newMatrix(n_rows,n_cols);
	for(i=0; i<n_rows; i++)
	{
		for(j=0; j<n_cols; j++)
			setM(result, i,j, c * getM(M, i, j));
	}
	return result;
}

// Repalces each element of matrix M with its product with c. -- No new memory allocated
void scalarMultiplyInPlace(double c, matrix* M)
{
	int i,j,n_rows,n_cols;
	double value;
	n_rows = M->n_rows;
	n_cols = M->n_cols;
	for(i=0; i<n_rows; i++)
	{
		for(j=0; j<n_cols; j++)
		{
			value = c * getM(M, i, j);
			setM(M, i,j, value);
		}
	}
}

// Returns the product of two matrices :  M1 and M2. --  New memory allocated
matrix* matrixMultiply(matrix* M1, matrix* M2)
{
	int i,j,m,n,k,K;
	double sum;

	m = M1->n_rows;
	K = M1->n_cols;
	n = M2->n_cols;
	matrix* product = newMatrix(m, n);
	for(i=0; i<m; i++)
	{
		for(j=0; j<n; j++)
		{
			sum = 0.0;
			for(k=0; k<K; k++)
				sum = sum + getM(M1,i,k)* getM(M2,k,j);
			setM(product,i,j,sum);
		}
 	}
 	return product;
}

// Calculates the innter product of two vectors M1 and M2.
double innerProduct(matrix* M1, matrix* M2)  
{
	int i;
	double sum = 0;
	int n_rows = M1->n_rows;
	for(i=0; i<n_rows; i++)
		sum = sum + getM(M1,i,0) * getM(M2,i,0);
	return sum;
}

// Returns the frobenius norm of matrix M
double frobNorm(matrix *M)
{
	int i,j, n_rows, n_cols;
	n_rows = M->n_rows;
	n_cols = M->n_cols;
	double norm = 0;
	for (i = 0; i < n_rows; i++)
	{
		for (j = 0; j < n_cols; j++)
			norm = norm + pow(getM(M,i,j),2);
	}
	norm = sqrt(norm);
	return norm;
}

// Returns various L norms of matrix M
double norm(matrix *M, int l)
{
	int i,j, n_rows, n_cols;
	n_rows = M->n_rows;
	n_cols = M->n_cols;
	double norm = 0;
	double maxNorm = 0 ;
	for (j = 0; j < n_cols; j++)
	{
		norm = 0;
		for (i = 0; i < n_rows; i++)
		{
			if(l == 1)
				norm = norm + fabs(getM(M,i,j));
		}
		if(norm>maxNorm)
			maxNorm = norm;
	}
	return maxNorm;
}

