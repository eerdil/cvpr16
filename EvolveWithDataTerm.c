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

// Global parameters
double alpha, dt;
double *Psi, *Image, *NarrowBand;
double *F; // Force for curve evolution
int sz, sz_i, sz_j; // image sizes
int *bitcodes; // initialized in function initializeBitcodes()

// parameters initialized in function computeXYIJCoordinateAndUniverseAndR()
int *ICoord, * JCoord, *universe;
double *XCoord, *YCoord, *R;


// HELPER FUNCTIONS TO FREE THE MEMORY
void freeMatrix(double ** a) 
{
	free(a[0]);
	free(a);
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

void calcGradient(double A[], double Ax[], double Ay[] ) 
{
    int i, j, k;
    for(k = 0; k < sz; k++) 
	{
		i = ICoord[k]; 
		j = JCoord[k];

		if(i == 1)
			Ax[k] = A[k + 1] - A[k];
		else if(i == sz_i)
			Ax[k] = A[k] - A[k - 1];
		else 
			Ax[k] = (A[k + 1] - A[k - 1]) / 2;

		if(j == 1)
			Ay[k] = A[k + sz_i] - A[k];
		else if(j == sz_j)
			Ay[k] = A[k] - A[k - sz_i];
		else 
			Ay[k] = (A[k + sz_i] - A[k - sz_i]) / 2;
    }
}

/* function to align current shape with the shapes in training set */
#define numPose  4
#define numPoseIteration 24
#define ratioThreshold 0.01

void updatePose(double **trainingI, double Psi[], int numShapes, double pose[]) 
{
	double *I, * tildeI;
	double *Ix, *Iy,  *TpIx, *TpIy;
	double * gradTildeI[numPose]; // this is an array of pointers
	double gradEpose[numPose]; // this is an array of pointers
	double maxR;
	double normalizer[numPose];
	int * domain;
	double sum, diff, num1[numPose], num2, num3[numPose], denom;
	double filling = 0;
	double factor = 1;
	double sinTheta, cosTheta, h;
	double Energy, prevEnergy, ratio = 1.0;
	int i, j, l, ct;
	I = (double *) malloc (sz * sizeof(double)); 
	Ix = (double *) malloc (sz * sizeof(double)); 
	Iy = (double *) malloc (sz * sizeof(double)); 
	TpIx = (double *) malloc (sz * sizeof(double)); 
	TpIy = (double *) malloc (sz * sizeof(double)); 
	tildeI = (double *) malloc (sz * sizeof(double)); // careful: is it int or double? Ans: double
	domain = (int *) malloc(sz * sizeof(int));

	for(i = 0; i < numPose; i++)
	{
		gradTildeI[i] = (double *) malloc (sz * sizeof(double));
	}

	// compute the target binary image I from Psi 
	// note that this Psi and binary image I is fixed but tildeI changes as the pose change
	for(i = 0; i < sz; i++) 
	{
		if(Psi[i] < 0)
			I[i] = 1.0;
		else 
			I[i] = 0.0;
	}
	
	// from I, compute gradient  Ix[], Iy[]  
	calcGradient(I, Ix, Iy);

// iterate the gradient decent
	ct = 0; 
	while(ratio > ratioThreshold) 
	{
		// anything that depends on pose should be inside this loop then apply Tp 
		Tp(Ix,  pose, filling, TpIx, domain);
		Tp(Iy,  pose, filling, TpIy, domain);

		// compute tildeI
		Tp(I,  pose, filling, tildeI, domain);
		sinTheta = sin(pose[2]); 
		cosTheta = cos(pose[2]); 
		h = pose[3];

		// compute the gradient flow
		for (i=0; i<sz; i++)  
		{
			gradTildeI[0][i] =- TpIx[i];
			gradTildeI[1][i] =- TpIy[i];
			gradTildeI[2][i] = (TpIx[i] * (-sinTheta * XCoord[i] + cosTheta * YCoord[i] ) + TpIy[i] * (-cosTheta * XCoord[i] - sinTheta * YCoord[i])) / h;
			gradTildeI[3][i] =- (TpIx[i] * (cosTheta * XCoord[i] + sinTheta * YCoord[i]) + TpIy[i] * (-sinTheta * XCoord[i] + cosTheta * YCoord[i])) / h / h;
		}

		for(l = 0; l < numPose; l++)
			gradEpose[l] = 0;

		Energy = 0;
		
		for(i = 0; i < numShapes; i++) 
		{ 
			for(l = 0; l < numPose; l++)
			{
				num1[l] = 0;
				num3[l] = 0;
			} 
			num2 = 0;
			denom = 0;

			for(j = 0; j < sz; j++) 
			{
				sum = tildeI[j] + trainingI[i][j];
				diff = tildeI[j] - trainingI[i][j];
				num2 += diff * diff;
				denom += sum * sum;
				for(l = 0; l < numPose; l++) 
				{
					num1[l] += diff * gradTildeI[l][j];
					num3[l] += sum * gradTildeI[l][j];
				}
			}
			Energy += num2 / denom;

			for(l = 0; l < numPose; l++)  
			{
				gradEpose[l] += 2 * num1[l] / denom - 2 * num2 * num3[l] / denom / denom;
			}

		}
	
		maxR = 0;
		for(i = 0; i < sz; i++) 
		{
			if(tildeI[i] > 0 && R[i] > maxR )
				maxR=R[i];
		}
		
		normalizer[0] = fabs(gradEpose[0]);
		normalizer[1] = fabs(gradEpose[1]);
		normalizer[2] = maxR * fabs(gradEpose[2]);
		normalizer[3] = maxR * fabs(gradEpose[3]);
		for(l = 0; l < numPose; l++) 
		{
			if(normalizer[l] != 0)
				gradEpose[l] /= normalizer[l];
		}
	       	    
		// update pose by the gradient  	
		for (l=0; l<numPose; l++)
		{
			pose[l] -= factor * gradEpose[l];
		}
        
		if(ct > 0 && Energy > prevEnergy)  
		{
			factor /= 2;
		}

		if(ct >= 1)
			ratio = fabs(Energy - prevEnergy) / prevEnergy; 

		prevEnergy = Energy;
		ct++;
	} // end iteration

	// free variables
	free(I); 
	free(Ix); 
	free(Iy);
	free(TpIx); 
	free(TpIy); 
	free(tildeI);
	free(domain);
	
	for (l=0;l<numPose;l++)
		free(gradTildeI[l]);
}

/* computes the data force for curve evolution which we use Chan-Vese*/
void calculateImageForce() 
{

	int i;
	int area1, area2;
	double sum1, sum2;
	double c1, c2; // mean intensities inside and outside of the curve
	
	sum1 = 0; sum2 = 0;
	area1 = 0; area2 = 0;
    
	for(i = 0; i < sz; i++)
	{
		if(Psi[i] < 0.0)
		{
			sum1 += Image[i];
			area1++;
		}
		else
		{
			sum2 += Image[i];
			area2++;
		}
	}
      
	c1 = sum1 / area1; 
	c2 = sum2 / area2;

	/* compute data force, F, for each pixel i */
	for(i = 0; i < sz; i++) 
	{
		F[i] = -(2 * Image[i] - c1 - c2) * (c1 - c2);    
	}
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
	int i, j, k, ctPos, ctNeg, currentIteration, lastIteration;
	double *trainingIMatrix, *trainingPhiMatrix;
	double **trainingI, **trainingPhi;
	double *poseForEachClassVector, **poseForEachClass;
	double *kappa;
	int numberOfClasses, numberOfShapesInEachClass, numberOfAllTrainingShapes;
	int *domain; // is used to know image domain after alignment
	double **tempTrainingI, **psiAligned;

	// Get inputs from MATLAB
	Image = mxGetPr(prhs[0]); // input image
	Psi = mxGetPr(prhs[1]); // initial level set
	NarrowBand = mxGetPr(prhs[2]); // narrow band curve representation: band = 1, otherwise = 0.
	trainingIMatrix = mxGetPr(prhs[3]); // vector containing all binary training shapes
	trainingPhiMatrix = mxGetPr(prhs[4]); // vector containing level set representations of all training shapes
	poseForEachClassVector = mxGetPr(prhs[5]); // tx, ty, theta, h
	numberOfClasses = mxGetScalar(prhs[6]); // number of classes in training set
	numberOfShapesInEachClass = mxGetScalar(prhs[7]); // number of shapes in each class. We assume that each class contains same number of training shapes
	// dt and alpha are used to determine gradient step size
	dt = mxGetScalar(prhs[8]);
	currentIteration = mxGetScalar(prhs[9]);
	lastIteration = mxGetScalar(prhs[10]);
	kappa = mxGetPr(prhs[11]);

	// image sizes
	sz_i = (int) mxGetM(prhs[0]);
	sz_j = (int) mxGetN(prhs[0]);
	sz = sz_i * sz_j;
	
	// Dynamic memory allocations
	F = (double *) malloc(sz * sizeof(double));
	domain = (int *) malloc(sz * sizeof(int));
	bitcodes = (int *) malloc (sz * sizeof(int));

	// 2D dynamic memory allocations
	numberOfAllTrainingShapes = numberOfClasses * numberOfShapesInEachClass;
	trainingI = (double **) malloc (numberOfAllTrainingShapes * sizeof( double *));
	trainingPhi = (double **) malloc (numberOfAllTrainingShapes * sizeof( double *));
	poseForEachClass = (double **) malloc(numberOfClasses * sizeof(double *));
	tempTrainingI = (double **) malloc (numberOfShapesInEachClass * sizeof(double *));
	psiAligned = (double **) malloc (numberOfClasses * sizeof(double *));

	reshapeMatrixFromVector(trainingIMatrix, trainingI, numberOfAllTrainingShapes, sz);
	reshapeMatrixFromVector(trainingPhiMatrix, trainingPhi, numberOfAllTrainingShapes, sz);
	reshapeMatrixFromVector(poseForEachClassVector, poseForEachClass, numberOfClasses, 4);

	// initializations
	computeXYIJCoordinateAndUniverseAndR(); 

	// Check if the curve has disappeared during curve evolution
	ctPos=0; 
	ctNeg=0;
	for(j = 0; j < sz; j++) 
	{
		if(Psi[j] < 0) 
			ctNeg++;
		else if (Psi[j] > 0)
			ctPos++;
	}
	if (ctPos == 0 || ctNeg == 0) 
	{
		printf ("one region has disappeared; we stop the curve evolution\n");
	}

	calculateImageForce();

	for (i = 0; i < sz; i++)
	{
		if(NarrowBand[i] == 1)
			Psi[i] += dt * F[i] + kappa[i]; 
    }

	if(currentIteration == lastIteration)
	{
		for(j = 0;  j < numberOfClasses; j++)
		{
			for(k = 0; k < numberOfShapesInEachClass; k++)
			{
				tempTrainingI[k] = trainingI[j * numberOfShapesInEachClass + k];
			}
			updatePose(tempTrainingI, Psi, numberOfShapesInEachClass, poseForEachClass[j]);
			psiAligned[j] = (double *) malloc(sz * sizeof(double));
			TpSDF(Psi, poseForEachClass[j], 0.0, psiAligned[j], domain);
			for(k = 0; k < 4; k++)
			{
				poseForEachClassVector[j * 4 + k] = poseForEachClass[j][k];
			}
		}
		freeMatrixNew(psiAligned, numberOfClasses);
	}

	//function call to free memory
	free(F);
	free(domain);
	free(bitcodes);
	freeMatrixNew(trainingI, numberOfAllTrainingShapes);
	freeMatrixNew(trainingPhi, numberOfAllTrainingShapes);
	freeMatrixNew(poseForEachClass, numberOfClasses);
	free(tempTrainingI);
	free(XCoord); 
	free(YCoord); 
	free(ICoord); 
	free(JCoord); 
	free(R); 
	free(universe);
}