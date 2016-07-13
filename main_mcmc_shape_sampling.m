clear all
close all
clc

imageId = '01';
datasetName = 'aircraft'; % set 'MNIST' or 'aircraft'
load(sprintf('%s/testImages/testImage%s', datasetName, imageId)); % load test image

if(strcmp(datasetName, 'aircraft') == 1)
    % training set with the corresponding imageId in the trainingShapes
    % folder does not include the test image with imageId, i.e., works in
    % leave-one-out fashion.
    load(sprintf('%s/trainingShapes/trainingSet%s', datasetName, imageId)); % load training shapes. Contains (sz_i x sz_j) x #ofTrainingImages matrix, sz_i and sz_j. The training set does not contain the shape in test image.
else
    load(sprintf('%s/trainingShapes/trainingSet', datasetName)); % load training shapes. Contains (sz_i x sz_j) x #ofTrainingImages matrix, sz_i and sz_j. The training set does not contain the shape in test image.
end

isOccludedRegionKnown = false; % set true when the occluded region is known
occludedRegion = zeros(size(testImage));
if(isOccludedRegionKnown)
    % if occluded region is known, define this region by labeling
    % corresponging pixels by 1, otherwise 0. Following line is an example. 
    % You must change the known region for your application. 
    % You can also read such regions from an image file like 
    % occludedRegion = imread('occludedRegion.png');
    occludedRegion(5:10, 5:10) = 1;
end

%% providing initial curve
figure, imagesc(testImage); colormap(gray); axis('off');%display test image to ask initial curve
Psi = initialLevelSet(5, sz_i, sz_j);
hold on,
contour(Psi, [0 0], 'r'); % display initial curve on the figure
hold off;

%% construct level set representation of shapes in training set
trainingIMatrix = AlignedShapeMatrix;
numberOfShapesInTrainingSet = size(AlignedShapeMatrix, 2);
trainingPhiMatrix = zeros(sz_i * sz_j, numberOfShapesInTrainingSet);
for i = 1:numberOfShapesInTrainingSet
    curShape = double(reshape(trainingIMatrix(:, i), [sz_i sz_j]) > 0);
    dummy = generateLevelSet(-2 * curShape + 1);
    trainingPhiMatrix(:, i) = dummy(:);
end

%% curve evolution with data term
numberOfIterations = 10;
numberOfClasses = 1; % 1 for 'aircraft', 10 for 'MNIST'
numberOfShapesInEachClass = 10;

poseForEachClass = zeros(4, numberOfClasses); poseForEachClass(4, :) = 1;
dt = 0.2; % gradient step size

for i = 1:numberOfIterations
    disp(sprintf('iter %d', i));
    narrowBand = createNarrowBand(Psi, 5);
    kappa = curvature(testImage);
    EvolveWithDataTerm(double(testImage), Psi, narrowBand, trainingIMatrix, trainingPhiMatrix,  poseForEachClass, numberOfClasses, ...
        numberOfShapesInEachClass, dt, i, numberOfIterations, kappa);
    
    if(mod(i, 10) == 0)
        imagesc(testImage); axis('off');
        hold on,
        contour(Psi, 'LineWidth', 3, 'LineColor', [1 0 0], 'LevelList', 0);
        pause(1)
    end
end
hold off;

%% MCMC shape sampling
numberOfSamples = 500; % number of samples to be generated
numberOfSamplingIterations = 20; % number of iterations to generate a single sample
numberOfIterationForSinglePertubation = 1; % number of iterations to generate a single curve perturbation. suggested 10 for MNIST and 1 for aircraft.
gamma = 1; % number of shapes in the class that contributes the curve evolution. This can be random as well. This is the gamma parameter in the paper.
dt = 0.2; % gradient step
alpha = 5; % weight of shape force
beta = 1; % weight of data force

display = 1;
%parpool(12);
for i = 1:numberOfSamples
    currentCurve = Psi;
    previousSelectedShapeId = zeros(1);
    acceptedCount = 0;
    pose = zeros(1, 4); pose(4) = 360 / 360;
    
%     if(strcmp(datasetName, 'Aircraft')) % aircraft dataset is already aligned
%         poseForEachClass = zeros(1, 4); poseForEachClass(4) = 360 / 360;
%     end
    for j = 1:numberOfSamplingIterations
        Psi_Candidate = zeros(size(currentCurve));
        Psi_Candidate(1:end, 1:end) = currentCurve(1:end, 1:end);
        
        rng('shuffle');
            
        mhThreshold = rand(); % Metropolis-Hasting threshold
        randomNumberForClassSelection = rand(); % to choose class to be sampled
        
        randomNumberArray = rand(1, gamma); % random number for selecting each shape    
        pForward = zeros(1); % forward transition probability
        pReverse = zeros(1); % reverse transition probability
        
        if(j == 1)
            currentSelectedClassId = int32(zeros(1));
        end
        currentSelectedShapeId = int32(zeros(1, gamma));
        
        for k = 1:numberOfIterationForSinglePertubation
            narrowBand = createNarrowBand(Psi_Candidate, 5);
            mcmcShapeSampling(double(testImage), Psi_Candidate, narrowBand, trainingIMatrix, trainingPhiMatrix, poseForEachClass, numberOfClasses, ...
                numberOfShapesInEachClass, dt, alpha, j, randomNumberForClassSelection, gamma, randomNumberArray, ...
                pForward, pReverse, currentSelectedClassId, currentSelectedShapeId, previousSelectedShapeId, acceptedCount, pose, numberOfIterationForSinglePertubation, k, beta);
        end
        pOfCandidate = zeros(1); % pi of candidate
        pOfCurrent = zeros(1); % pi of current
        
        evaluateEnergyWithShapePrior(Psi_Candidate, trainingIMatrix, trainingPhiMatrix, numberOfClasses, numberOfShapesInEachClass, pose, currentSelectedClassId, pOfCandidate);
        evaluateEnergyWithShapePrior(currentCurve, trainingIMatrix, trainingPhiMatrix, numberOfClasses, numberOfShapesInEachClass, pose, currentSelectedClassId, pOfCurrent);

        minusLogpOfDataCandidate = evaluateEnergyWithDataTerm(testImage, Psi_Candidate, occludedRegion);
        minusLogpOfDataCurrent = evaluateEnergyWithDataTerm(testImage, currentCurve, occludedRegion);
        
        energyCandidate = alpha * -log(pOfCandidate);
        energyCurrent = alpha * -log(pOfCurrent);
        
        piOfCandidate = exp(-energyCandidate);
        piOfCurrent = exp(-energyCurrent);
        
        hastingRatio = (piOfCandidate * pReverse) / (piOfCurrent * pForward);

        if(mhThreshold < hastingRatio || acceptedCount == 0) % accept the sample
            currentCurve = Psi_Candidate;
            previousSelectedShapeId = currentSelectedShapeId;
            acceptedCount = acceptedCount + 1;
        else
            currentCurve = currentCurve; % reject the candidate
        end
        if(display & mod(j, 1) == 0)
            imagesc(testImage); colormap(gray); axis('off'); 
            title(sprintf('Sample %d - Sampling iteration %d / %d - Accepted count %d', i, j, numberOfSamplingIterations, acceptedCount));
            hold on,
            contour(currentCurve, 'LineWidth', 3, 'LineColor', [1 0 0], 'LevelList', 0);
            pause(1);
        end
        
    end
    temp = zeros(size(currentCurve));
    temp(find(currentCurve < 0)) = 1;
    t = uint8(zeros(size(testImage, 1), size(testImage, 2), 3));
    t(:, :, 1) = testImage;
    t(:, :, 2) = testImage;
    t(:, :, 3) = testImage;
    
    I1 = bwmorph(temp, 'remove');
    temp1 = t(:, :, 1);
    temp1(find(I1 == 1)) = 255;
    t(:, :, 1) = temp1;
    
    temp1 = t(:, :, 2);
    temp1(find(I1 == 1)) = 0;
    t(:, :, 2) = temp1;
    
    temp1 = t(:, :, 3);
    temp1(find(I1 == 1)) = 0;
    t(:, :, 3) = temp1;
    imwrite(t, sprintf('%s/Results/%d_sample_%d.png', datasetName, currentSelectedClassId, i));
end


keyboard;


















