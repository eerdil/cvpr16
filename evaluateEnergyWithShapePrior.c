#include <math.h>
#include "mex.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define XPOS 1
#define	XNEG 2
#define YPOS 4
#define YNEG 8
#define XNHBRS 3
#define YNHBRS 12
#define MY_PI 3.14

// Global parameters
double alpha, dt;
double *Psi, *Image, *NarrowBand;
int sz, sz_i, sz_j; // image sizes
int *bitcodes; // initialized in function initializeBitcodes()
double *F; // Force for curve evolution
// parameters initialized in function computeXYIJCoordinateAndUniverseAndR()
int *ICoord, * JCoord, *universe;
double *XCoord, *YCoord, *R;


// HELPER FUNCTIONS TO FREE THE MEMORY
void freeMatrix(double ** a) {
  free(a[0]);
  free(a) ;
}

void freeMatrixNew (double ** matrix, int numRows) 
{
    int i;
    for(i = 0; i < numRows; i++)  
	{
	    free (matrix[i]);
    }
    free(matrix);
}

void freeTrainingIAndTrainingPhi (double ** trainingI, double ** trainingPhi, int numShapes) {
    int i;
    for (i=0; i<numShapes;i++)  {
	    free (trainingI[i]);
	    free (trainingPhi[i]);
    }
    free(trainingI);
    free(trainingPhi);
}

#define gkernel(A,B) 1/sqrt(2*MY_PI)/(B)*exp(-(A)*(A)/2/(B)/(B))
#define Heaviside(A)  ((A) > 0 ? 1 : 0 )

/* Computes template distance between two shapes */
double computeDistanceTemplate(double phi1[], double phi2[]) 
{
	int i;
	double area;

	area = 0; 
	for(i = 0; i < sz; i++) 
	{
		if(phi1[i] > 0 && phi2[i] < 0 || phi1[i] < 0 && phi2[i] > 0)
			area ++;
	}
	return area;
}

/* Allocates dynamic memory for a matrix with given row and column */
double ** matrix(int row, int col) 
{
	double **S;
	int i, j;
   
	S = (double **) malloc (row * sizeof(double*));
	S[0] = (double *) malloc (row * col * sizeof(double));
	for(i = 1; i < row; i++)
	{
		S[i]=S[i-1]+col;
	}
	for (i=0;i<row;i++)
	{
		for (j=0;j<col;j++)
		{
			S[i][j]=0;
		}
	}

	return S;
}

/* finds kernel size by computing average of all min distances between shapes*/
double shapeKernelSize(double** trainingPhi, int numShapes) 
{
	double sumSq, sum,avg,sigma;
    double **distMtx;
    double ksize;
    int i,j;

    sumSq=0; sum=0;
    distMtx=matrix(numShapes,numShapes);
    for (i=0;i<numShapes;i++) 
	for(j=0;j<numShapes;j++) {
            distMtx[i][j]=computeDistanceTemplate(trainingPhi[i], trainingPhi[j]);
   	    sumSq+=distMtx[i][j]*distMtx[i][j];
	    sum+=distMtx[i][j];
	} 
    avg=sum/numShapes/numShapes;
    sigma=sqrt(sumSq/numShapes/numShapes -avg*avg);
    freeMatrix(distMtx, numShapes);
    return sigma;
}

void scaleLevelSetFunction(double phi[], double factor)  
{
    int i;
    for(i = 0; i < sz ; i++)
	{
		phi[i] *= factor;
	}
}

void Tp(double Psi[], double p[], double filling, double newPsi [], int domain[]) 
{
    double a, b, theta, h, cosTheta, sinTheta;
    int i, ii, jj, ic, jc;
    a = p[0];
    b = p[1];
    theta = p[2];
    h = p[3]; 

    ic = (sz_i + 1) / 2;
    jc = (sz_j + 1) / 2;

    for (i=0;i<sz;i++) 
	{
		cosTheta = cos(theta);
		sinTheta = sin(theta);
        ii = (int) floor((cosTheta * XCoord[i] + sinTheta * YCoord[i]) / h + ic - a + 0.5);
        jj = (int) floor((-sinTheta * XCoord[i] + cosTheta * YCoord[i]) / h + jc - b + 0.5);
        if(ii > 0 && ii <= sz_i && jj > 0 && jj <= sz_j) 
		{
            domain[i] = 1;
            newPsi[i] = Psi[ii - 1 + (jj - 1) * sz_i]; 
		}
		else
		{
			domain[i] = 0; 
			newPsi[i] = filling;
		}
    }
}

void TpSDF(double Psi[], double p[], double filling, double newPsi [], int domain[]) 
{
     Tp(Psi, p, filling, newPsi, domain);
     scaleLevelSetFunction(newPsi, p[3]);
}

/* helper function */
void computeXYIJCoordinateAndUniverseAndR () 
{
	int i, ii, jj;		 
    XCoord = (double *) malloc (sizeof(double) * sz);
    YCoord = (double *) malloc (sizeof(double) * sz);
    R = (double *) malloc (sizeof(double) * sz);
    ICoord = (int *) malloc (sizeof(int) * sz);
    JCoord = (int *) malloc (sizeof(int) * sz);
    universe = (int *) malloc (sizeof(int) * sz);
    
	for(i = 0; i < sz; i++) 
	{
 		ii = i % sz_i + 1;
		jj = (int) floor(i / sz_i) + 1;
        ICoord[i] = ii;
        JCoord[i] = jj;
        XCoord[i] = ii - (sz_i + 1.0) / 2.0;       
        YCoord[i] = jj - (sz_j + 1.0) / 2.0;
		R[i] = sqrt(XCoord[i] * XCoord[i] + YCoord[i] * YCoord[i]);
		universe[i]=1;
    }
}

/* calculates force that comes from shape prior*/
double computeShapeForce (double phi[], double pose[], double ** trainingPhi, int numShapes, double ksize, double shapeF[], int currentClassId, int numShapesInEachClass) 
{
	int i, j;
	double * tildePhi;
	double * tildeForce;
	double weight;
	double dist;
	int * domain, * shapeForceDomain;
	double filling = 0.0;
	double pOfPhi = 0, factor;
	double *shapeForce;

	tildeForce = (double *) malloc(sz * sizeof(double)); 
	tildePhi = (double *) malloc(sz * sizeof(double));
	shapeForce = (double *) malloc(numShapes * sizeof(double));
	domain = (int * ) malloc(sz * sizeof(int));

	for(j = 0; j < sz; j++)
		tildeForce[j] = 0; 
    
	TpSDF(phi, pose, filling, tildePhi, domain);
    
	for (i = 0; i < numShapes; i++) 
	{           
		dist = computeDistanceTemplate(tildePhi, trainingPhi[i]);
		weight = gkernel(dist, ksize);
		pOfPhi += weight;
		shapeForce[i] = pOfPhi;
	}
	for(i = 0; i < numShapes; i++)
	{
		shapeForce[i] = shapeForce[i] / shapeForce[numShapes - 1];
	}
	pOfPhi = 0;
	for(i = currentClassId * numShapesInEachClass; i < (currentClassId + 1) * numShapesInEachClass; i++)
	{
		if(i == 0)
		{
			pOfPhi += shapeForce[i];
		}
		else
		{
			pOfPhi += (shapeForce[i] - shapeForce[i - 1]);
		}
	}
	pOfPhi = pOfPhi / numShapesInEachClass;
	free(tildeForce); free(tildePhi); free(domain); free(shapeForce);
	return pOfPhi;
	
}

/* adds shape force to data force */
void  addShapeForceToDataForce(double shapeF[], double beta, double F[]) 
{
	int i;
	double maxDataF = 0, maxShapeF = 0;
	double internalFactor;

	for(i = 0; i < sz; i++) 
	{
		if (F[i] !=0) 
		{
			if(fabs(F[i]) > maxDataF)
				maxDataF = fabs(F[i]);
			if(fabs(shapeF[i]) > maxShapeF)
				maxShapeF = fabs(shapeF[i]);
		}
	}
	internalFactor = maxShapeF / maxDataF;
	    
	for(i = 0; i < sz; i++)
		F[i] += beta * shapeF[i] / internalFactor;
}

/* forms given vector to matrix with specified rows and columns */
void reshapeMatrixFromVector(double vector[], double **matrix, int numRows, int numColumns)
{
	int i, j, k;
	for(i = 0; i < numRows; i++)
	{
		matrix[i] = (double *) malloc(numColumns * sizeof(double));
		for(j = 0; j < numColumns; j++)
		{
			k = i * numColumns + j;
			matrix[i][j] = vector[k];
		}
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// variable definitions
	double *trainingIMatrix, *trainingPhiMatrix;
	double **trainingI, **trainingPhi;
	double *pose;
	int numberOfClasses, numberOfShapesInEachClass, selectedClassId, numberOfAllTrainingShapes;
	double kernelSize;
	double *shapeF, *energy;

	// Get inputs from MATLAB
	Psi = mxGetPr(prhs[0]); // initial level set
	trainingIMatrix = mxGetPr(prhs[1]); // vector containing all binary training shapes
	trainingPhiMatrix = mxGetPr(prhs[2]); // vector containing level set representations of all training shapes
	numberOfClasses = mxGetScalar(prhs[3]); // number of classes in training set
	numberOfShapesInEachClass = mxGetScalar(prhs[4]); // number of shapes in each class. We assume that each class contains same number of training shapes
	pose = mxGetPr(prhs[5]);
	selectedClassId = mxGetScalar(prhs[6]); // id of the randomly selected class
	energy = mxGetPr(prhs[7]);

	// image sizes
	sz_i = (int) mxGetM(prhs[0]);
	sz_j = (int) mxGetN(prhs[0]);
	sz = sz_i * sz_j;
	
	// Dynamic memory allocations
	shapeF = (double *) malloc (sz * sizeof(double));

	// 2D dynamic memory allocations
	numberOfAllTrainingShapes = numberOfClasses * numberOfShapesInEachClass;
	trainingI = (double **) malloc (numberOfAllTrainingShapes * sizeof( double *));
	trainingPhi = (double **) malloc (numberOfAllTrainingShapes * sizeof( double *));

	reshapeMatrixFromVector(trainingIMatrix, trainingI, numberOfAllTrainingShapes, sz);
	reshapeMatrixFromVector(trainingPhiMatrix, trainingPhi, numberOfAllTrainingShapes, sz);

	computeXYIJCoordinateAndUniverseAndR(); 

	kernelSize = shapeKernelSize(trainingPhi, numberOfAllTrainingShapes);
	energy[0] = computeShapeForce(Psi, pose, trainingPhi, numberOfAllTrainingShapes, kernelSize, shapeF, selectedClassId, numberOfShapesInEachClass);

	free(shapeF);
	freeMatrixNew(trainingI, numberOfAllTrainingShapes);
	freeMatrixNew(trainingPhi, numberOfAllTrainingShapes);
	free(XCoord); 
	free(YCoord); 
	free(ICoord); 
	free(JCoord); 
	free(R); 
	free(universe);
	
}