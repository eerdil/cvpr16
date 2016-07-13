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
double alpha, beta, dt;
double *Psi, *Image, *NarrowBand;
double *F; // Force for curve evolution
int sz, sz_i, sz_j; // image sizes
int *bitcodes; // initialized in function initializeBitcodes()

// parameters initialized in function computeXYIJCoordinateAndUniverseAndR()
int *ICoord, * JCoord, *universe;
double *XCoord, *YCoord, *R;


//HELPER FUNCTIONS TO FREE THE MEMORY

void freeMatrix(double ** matrix, int rowNumber) 
{
    int i;
    for(i = 0; i < rowNumber; i++)  
	{
	    free(matrix[i]);
    }
    free(matrix);
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
	
	for(i = 0; i < row; i++)
	{
		S[i] = (double *) malloc (col * sizeof(double));
	}

	for(i = 0; i < row; i++)
	{
		for(j = 0; j < col; j++)
		{
			S[i][j] = 0;
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

void inverseTp(double Psi[], double p[], double filling, double newPsi [], int domain[]) 
{
    double a, b, theta, h, cosTheta, sinTheta;
    int i, ii, jj, ic, jc;
    a = p[0];
    b = p[1];
    theta = p[2];
    h = p[3]; 

    ic = (sz_i + 1) / 2;
    jc = (sz_j + 1) / 2;

    for(i = 0; i < sz; i++) 
	{
		cosTheta = cos(theta);
		sinTheta = sin(theta);
        ii = (int) floor((cosTheta * (XCoord[i] + a) - sinTheta * (YCoord[i] + b)) * h + ic + 0.5);
        jj = (int) floor((sinTheta * (XCoord[i] + a) + cosTheta * (YCoord[i] + b)) * h + jc + 0.5);
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

void inverseTpSDF(double Psi[], double p[], double filling, double newPsi [], int domain[]) 
{
     inverseTp(Psi, p, filling, newPsi, domain);
     scaleLevelSetFunction(newPsi, 1.0 / p[3]);
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
		F[i] =- (2 * Image[i] - c1 - c2) * (c1 - c2);    
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

/* Performs random shape and class selection with forward and reverse perturbation probability */
void performRandomSelections(double phi[], double pose[], double ** trainingPhi, int numShapes, double ksize, double randomNumberForClassSelection, int randomNumberForTotalShapes, double randomNumberArray[], int numClasses, int numShapesInEachClass, int selectedShapeIds[], double *pForward, int* currentSelectedClassId, int* currentSelectedShapeId, int *previousSelectedShapeId, int acceptedCount, double *pReverse, int sampleIter, double **allPose, int singlePerturbIter)
{
	int i;
	double * tildePhi; // curve after registration
	double *classWeights, *shapeWeights; // classWeights: stores weights for each class given in Eq 5 in the paper for random class selection. shapeWeights: stores weights of each shape in randomly selected class as given in Eq 6 in the paper.
	double weight; // temporary weight when computing cumulative weights in both classWeights and shapeWeights
	double dist; // temporary distance when computing cumulative weights in both classWeights and shapeWeights
	int * domain; // necessary when applying pose parameters to shape to determine image domain
	double filling = 0.0; // necessary when applying pose parameters to shape to fill regions outside the image domain
	double pOfPhi = 0; // temporary variable when computing cumulative weights in both classWeights and shapeWeights
	int counter = 0;
	int randomlySelectedClassId; // stores the id of randomly selected class.
	bool loop = true;
	double pOfSelectingShapes; // temp variable to compute forward perturbation probability

	/* dynamic memory allocation */
	tildePhi = (double *) malloc(sz * sizeof(double));
	classWeights = (double *) malloc(numClasses * sizeof(double));
	shapeWeights = (double *) malloc(numShapesInEachClass * sizeof(double));
	domain = (int *) malloc(sz * sizeof(int));
    
	/* in the first iteration, select a random class as introduced in Section 4.1 */
	if(sampleIter == 1)
	{
		// for each class, compute cumulative weights of the shapes in that class
		for(i = 0; i < numShapes; i++)
		{
			counter++;

			TpSDF(phi, allPose[i / numShapesInEachClass], filling, tildePhi, domain); // apply pose parameters of the current class to the evolving shape. The pose parameters are computed in the last iteration of the data driven segmentation
			dist = computeDistanceTemplate(tildePhi, trainingPhi[i]); // compute distance between the evolving shape and a shape in the training set
			weight = gkernel(dist, ksize); // gaussian kernel
			//printf("Shape %d, dist = %f, weight = %.10f, ksize = %f \n", i, dist, weight, ksize);
			pOfPhi += weight; // cumulative weight
		
			if(counter == numShapesInEachClass)
			{
				counter = 0;
				classWeights[i / numShapesInEachClass] = pOfPhi; // classWeights array stores the cumulative weights for each class
				
			}
		}

		// normalize class weights
		for(i = 0; i < numClasses; i++)
		{
			classWeights[i] = classWeights[i] / classWeights[numClasses - 1];
			/*
			if(sampleIter == 1 && singlePerturbIter == 1)
			{
				if(i == 0)
				{
					printf("Class %d, weight = %.10f \n", i, classWeights[i]);
				}
				else
				{
					printf("Class %d, weight = %.10f \n", i, (classWeights[i] - classWeights[i - 1]));
				}
			}
			*/
		}

	
		i = 0;
		while(loop)
		{
			if(randomNumberForClassSelection <= classWeights[i])
			{
				randomlySelectedClassId = i;
				loop = false;
			}
			i++;
		}
		currentSelectedClassId[0] = randomlySelectedClassId;

		// store pose parameters of the selected class
		for(i = 0; i < 4; i++)
		{
			pose[i] = allPose[currentSelectedClassId[0]][i];
		}
	}
	/* Random class selectio has been completed */

	/* Random shape selection as introduced in Section 4.2 */
	TpSDF(phi, pose, filling, tildePhi, domain); // apply pose parameters to the evolving shape

	pOfPhi = 0; // stores cumulative shape weights

	// compute cumulative weights of the shapes from the randomly selected class
	counter = 0;
	for(i = currentSelectedClassId[0]*numShapesInEachClass; i < (currentSelectedClassId[0] + 1) * numShapesInEachClass; i++)
	{
		dist = computeDistanceTemplate(tildePhi, trainingPhi[i]);
		weight = gkernel(dist, ksize);
		pOfPhi += weight;
		shapeWeights[counter] = pOfPhi;
		counter++;
	}

	// normalize weights
	for(i = 0; i < numShapesInEachClass; i++)
	{
		shapeWeights[i] = shapeWeights[i] / shapeWeights[numShapesInEachClass - 1];
	}

	// select randomNumberForTotalShapes shapes from the selected class. The probabilities for selecting each shape are stored in randomNumberArray
	pOfSelectingShapes = 1.0;

	for(i = 0; i < randomNumberForTotalShapes; i++)
	{
		loop = true;
		counter = 0;
		while(loop)
		{
			if(randomNumberArray[i] < shapeWeights[counter])
			{
				selectedShapeIds[i] = currentSelectedClassId[0] * numShapesInEachClass + counter;
				currentSelectedShapeId[i] = selectedShapeIds[i];
				//printf("shape Id = %d \n", currentSelectedShapeId[i]);
				dist = computeDistanceTemplate(tildePhi, trainingPhi[selectedShapeIds[i]]);
				pOfSelectingShapes *= gkernel(dist,ksize);
				loop = false;
			}
			counter++;
		}
	}

	pForward[0] = pOfSelectingShapes; // forward perturbation probability

	/* compute reverse perturbation probability. Note that reverse perturbation probability is not computed for the first sampling iteration as introduced in Section 4.3 */
	pReverse[0] = 1.0;
	if(acceptedCount != 0)
	{
		for(i = 0; i < randomNumberForTotalShapes; i++)
		{
			dist = computeDistanceTemplate(tildePhi, trainingPhi[previousSelectedShapeId[i]]);
			pReverse[0] *= gkernel(dist, ksize);
		}
	}

    free(tildePhi); free(classWeights); free(shapeWeights); free(domain);
}

/* calculates force that comes from shape prior*/
void computeShapeForce (double phi[], double pose[], double ** trainingPhi, int numShapes, double ksize, double shapeF[], int randomNumber2, int selectedShapeIds[]) 
{
	int i, j;
	double * tildePhi;
	double * tildeForce;
	double weight;
	double dist;
	int * domain, * shapeForceDomain;
	double filling = 0.0;
	double pOfPhi = 0, factor;

	tildeForce = (double *) malloc(sz * sizeof(double)); 
	tildePhi = (double *) malloc(sz * sizeof(double));
	domain = (int * ) malloc(sz * sizeof(int));

	for(j = 0; j < sz; j++)
		tildeForce[j] = 0; 
    
	TpSDF(phi, pose, filling, tildePhi, domain);
    
	for (i = 0; i < randomNumber2; i++) 
	{           
		dist = computeDistanceTemplate(tildePhi, trainingPhi[selectedShapeIds[i]]);
		weight = gkernel(dist, ksize);
		//printf("weight = %.10f, ksize = %f \n", weight, ksize);
		pOfPhi += weight;
		for(j = 0; j < sz; j++)
		{
			if(domain[j])
				tildeForce[j] += -weight * (1 - 2 * Heaviside(trainingPhi[selectedShapeIds[i]][j])) / randomNumber2;
		}
	}
	
	pOfPhi /= randomNumber2;
	factor = 1 / (pOfPhi * randomNumber2);
	for(j = 0; j < sz; j++)
		tildeForce[j] *=  factor;

	shapeForceDomain = domain;
	inverseTpSDF(tildeForce, pose, filling, shapeF, shapeForceDomain);
	
	free(tildeForce); free(tildePhi); free(domain);
}

/* adds shape force to data force */
void  addShapeForceToDataForce(double shapeF[], double F[]) 
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
		F[i] = beta * F[i] + alpha * shapeF[i] / internalFactor;
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
	int i, j, k, ctPos, ctNeg, samplingIteration;
	double *trainingIMatrix, *trainingPhiMatrix;
	double **trainingI, **trainingPhi;
	double *poseForEachClassVector, **poseForEachClass;
	double *randomNumberArray, *pForward, *pReverse, *pose;
	int numberOfClasses, numberOfShapesInEachClass, numberOfAllTrainingShapes, randomNumberForTotalShapes, acceptedCount;
	int *domain; // is used to know image domain after alignment
	double kernelSize, randomNumberForClassSelection;
	int *selectedShapeIds; // stores ids of randomly selected shapes
	int *currentSelectedClassId, *currentSelectedShapeId, *previousSelectedShapeId;
	double *shapeF;
	int numberOfIterationForSinglePertubation;

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
	alpha = mxGetScalar(prhs[9]);
	samplingIteration = mxGetScalar(prhs[10]);
	randomNumberForClassSelection = mxGetScalar(prhs[11]);
	randomNumberForTotalShapes = mxGetScalar(prhs[12]);
	randomNumberArray = mxGetPr(prhs[13]);
	pForward = mxGetPr(prhs[14]);
	pReverse = mxGetPr(prhs[15]);
	currentSelectedClassId = mxGetPr(prhs[16]);
	currentSelectedShapeId = mxGetPr(prhs[17]);
	previousSelectedShapeId = mxGetPr(prhs[18]);
	acceptedCount = mxGetScalar(prhs[19]);
	pose = mxGetPr(prhs[20]);
	numberOfIterationForSinglePertubation = mxGetScalar(prhs[21]);
	k = mxGetScalar(prhs[22]);
	beta = mxGetScalar(prhs[23]);

	// image sizes
	sz_i = (int) mxGetM(prhs[0]);
	sz_j = (int) mxGetN(prhs[0]);
	sz = sz_i * sz_j;
	
	// Dynamic memory allocations
	F = (double *) malloc(sz * sizeof(double));
	domain = (int *) malloc(sz * sizeof(int));
	bitcodes = (int *) malloc (sz * sizeof(int));
	selectedShapeIds = (int *) malloc(randomNumberForTotalShapes * sizeof(double));
	shapeF = (double *) malloc (sz * sizeof(double));

	// 2D dynamic memory allocations
	numberOfAllTrainingShapes = numberOfClasses * numberOfShapesInEachClass;
	trainingI = (double **) malloc (numberOfAllTrainingShapes * sizeof( double *));
	trainingPhi = (double **) malloc (numberOfAllTrainingShapes * sizeof( double *));
	poseForEachClass = (double **) malloc(numberOfClasses * sizeof(double *));

	reshapeMatrixFromVector(trainingIMatrix, trainingI, numberOfAllTrainingShapes, sz);
	reshapeMatrixFromVector(trainingPhiMatrix, trainingPhi, numberOfAllTrainingShapes, sz);
	reshapeMatrixFromVector(poseForEachClassVector, poseForEachClass, numberOfClasses, 4);
	
	// initializations
	//initializeBitcodes(); 
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

	kernelSize = shapeKernelSize(trainingPhi, numberOfAllTrainingShapes);
	if(k == 1)
	{
		performRandomSelections(Psi, pose, trainingPhi, numberOfAllTrainingShapes, kernelSize, randomNumberForClassSelection, randomNumberForTotalShapes, randomNumberArray, numberOfClasses, numberOfShapesInEachClass, selectedShapeIds, pForward, currentSelectedClassId, currentSelectedShapeId, previousSelectedShapeId, acceptedCount, pReverse, samplingIteration, poseForEachClass, k);
	}

	calculateImageForce();
	computeShapeForce(Psi, pose, trainingPhi, numberOfAllTrainingShapes, kernelSize, shapeF, randomNumberForTotalShapes, currentSelectedShapeId);
	addShapeForceToDataForce(shapeF, F);

	// update level set
	for (i = 0; i < sz; i++)
	{
		if(NarrowBand[i] == 1)
			Psi[i] += dt * F[i]; 
	}

	

	//function call to free memory
	free(F);
	free(domain);
	free(bitcodes);
	free(shapeF);
	free(selectedShapeIds);
	freeMatrix(trainingPhi, numberOfAllTrainingShapes);
	freeMatrix(trainingI, numberOfAllTrainingShapes);
	freeMatrix(poseForEachClass, numberOfClasses);
	free(XCoord); 
	free(YCoord); 
	free(ICoord); 
	free(JCoord); 
	free(R); 
	free(universe);
}