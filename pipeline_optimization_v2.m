%% Pipeline Optimization
% William Baumchen
close all; clear; clc

% Verbose Output - [0 for suppression, 1 for iteration]
verboze = 1;

% Number of iterations for Bayesian hyperparameter optimization
iternn = 250;

%% Data Inport

% Import Data
datain = readmatrix("winequality-white.csv");
% Shuffle Data Entries for Splitting Data
% Set random seed for reproducibility
% rng(47)
% rng(44)
rng(42)
datain = datain(randperm(size(datain,1)),:);
% Set Fraction of Entries for Test Set
a = 0.2;
% Split Data
xTest = datain(1:round(a*size(datain,1)),1:11);
yTest = datain(1:round(a*size(datain,1)),12);
xTrain = datain(round(a*size(datain,1))+1:end,1:11);
yTrain = datain(round(a*size(datain,1))+1:end,12);

%% Pipeline Optimization

% Instantiate Optimizable hyperparameters for pipeline
norm = optimizableVariable('normVal',[0,1],'Type','integer');
feat = optimizableVariable('featureNum',[0,11],'Type','integer');
solv = optimizableVariable('solver',[0,2],'Type','integer');
trhp1 = optimizableVariable('minLeaf',[1,max(2,height(xTrain)-1)],'Type','integer');
knhp1 = optimizableVariable('distance',[0,10],'Type','integer');
knhp2 = optimizableVariable('numNeigh',[1,max(2,round(height(xTrain)/2))],'Type','integer');
knhp3 = optimizableVariable('knStandard',[0,1],'Type','integer');
enhp1 = optimizableVariable('Method',{'Bag','AdaBoostM2','RUSBoost'},'Type','categorical');

% Create function
fun = @(x)pipopt(x,xTrain,yTrain);
% Assemble hyperparameter variables
vars = [norm,feat,solv,trhp1,knhp1,knhp2,knhp3,enhp1];
% Run Bayesian Optimization
results = bayesopt(fun,vars,'Verbose',verboze,'MaxObjectiveEvaluations',iternn,'AcquisitionFunctionName','expected-improvement-plus');

%% Get Best Pipeline Model

% Retrieve best model, parameters, and results
[mdlError,Model,xxTest] = pipfinal(results.XAtMinEstimatedObjective,xTrain,yTrain,xTest,yTest);

% Find Baseline
solverr = results.XAtMinEstimatedObjective.(3);
% Choose correct model and find a 'baseline' ml model
if solverr == 0
    baseline = fitctree(xTrain,yTrain);
elseif solverr == 1 
    baseline = fitcensemble(xTrain,yTrain);
elseif solverr == 2
    baseline = fitcknn(xTrain,yTrain);
end
% Evaluate model loss
bassline = crossval(baseline,'KFold',5);
baseLoss = kfoldLoss(bassline);

%% Comparison

% Compare model against a baseline approach
baseError = loss(baseline,xTest,yTest);
mdlLoss = results.MinEstimatedObjective;

% Graph confusion chart of predicted results
figure(2)
confusionchart(yTest,predict(baseline,xTest))
title('Baseline Prediction')
figure(3)
confusionchart(yTest,predict(Model,xxTest))
title('Model Prediction')

% Plot objective trace of iterations
figure(4)
plot(1:iternn,results.ObjectiveTrace)
title('Optimization Evaluation')
xlabel('Iteration Number')
ylabel('Average Cross-Validated Classification Loss')

% Find and sort best models from iteration trace
resx = results.ObjectiveTrace;
resobj = [1:iternn]';
res = [resobj,resx];
bbres = sortrows(res,2);
xTrace = results.XTrace;
for i = 1:iternn
score1(i,:) = xTrace(bbres(i,1),:);
end

%% Save Model Workspace
save('pipelineobs.mat')

%% Autonomous function:
function [Result,Model] = pipopt(x,xTrain,yTrain)
%pipopt is a function that takes in the optimization hyperparameters x and
% the training data xTrain and yTrain as two matrices in order to fit a
% 5-fold cv model of the given type to said data, and to report model and
% cv results

% Normalization
if x.normVal == 1
    xxxTrain = normalize(xTrain);
else
    xxxTrain = xTrain;
end

% PCA
if x.featureNum ~= 0
    [~,scoreTrain] = pca(xxxTrain);
    xxxTrain = scoreTrain(:,1:x.featureNum);
end

if x.solver == 0
    % Optimize tree model
    Model = fitctree(xxxTrain,yTrain,'KFold',5,'MinLeafSize',x.minLeaf);
elseif x.solver == 1
    % Optimize ensemble model
    Model = fitcensemble(xxxTrain,yTrain,'KFold',5,'Method',char(x.Method));

elseif x.solver == 2
    % Optimize knn model

    if x.distance == 0
        distmm = 'cityblock';
    elseif x.distance == 1
        distmm = 'chebychev';
    elseif x.distance == 2
        distmm = 'correlation';
    elseif x.distance == 3
        distmm = 'cosine';
    elseif x.distance == 4
        distmm = 'euclidean';
    elseif x.distance == 5
        distmm = 'hamming';
    elseif x.distance == 6
        distmm = 'jaccard';
    elseif x.distance == 7
        distmm = 'mahalanobis';
    elseif x.distance == 8
        distmm = 'minkowski';
    elseif x.distance == 9
        distmm = 'seuclidean';
    elseif x.distance == 10
        distmm = 'spearman';
    end

    Model = fitcknn(xxxTrain,yTrain,'KFold',5,'Distance',char(distmm),'NumNeighbors',x.numNeigh,'Standardize',x.knStandard);
end

Result = kfoldLoss(Model);
end

function [Losst,Model,xxTest] = pipfinal(x,xTrain,yTrain,xTest,yTest)
%pipfinal is a function that takes in hyperparameters x, along with
%training and testing data matrices xTrain, yTrain, xTest, and yTest, and
%fits a model of the given type. After completing the model, it finds the
%classification error on the Test data, as well as reporting the processed
%xTest data as xxTest. 

% Normalization
if x.normVal == 1
    xxxTrain = normalize(xTrain);
    xxTest = normalize(xTest);
else
    xxxTrain = xTrain;
    xxTest = xTest;
end

% PCA
if x.featureNum ~= 0
    [~,scoreTrain] = pca(xxxTrain);
    xxxTrain = scoreTrain(:,1:x.featureNum);
    [~,scoreTrain] = pca(xxTest);
    xxTest = scoreTrain(:,1:x.featureNum);
end

if x.solver == 0
    % Optimize tree model
    Model = fitctree(xxxTrain,yTrain,'MinLeafSize',x.minLeaf);
elseif x.solver == 1
    % Optimize ensemble model
    Model = fitcensemble(xxxTrain,yTrain,'Method',char(x.Method));

elseif x.solver == 2
    % Optimize knn model

    if x.distance == 0
        distmm = 'cityblock';
    elseif x.distance == 1
        distmm = 'chebychev';
    elseif x.distance == 2
        distmm = 'correlation';
    elseif x.distance == 3
        distmm = 'cosine';
    elseif x.distance == 4
        distmm = 'euclidean';
    elseif x.distance == 5
        distmm = 'hamming';
    elseif x.distance == 6
        distmm = 'jaccard';
    elseif x.distance == 7
        distmm = 'mahalanobis';
    elseif x.distance == 8
        distmm = 'minkowski';
    elseif x.distance == 9
        distmm = 'seuclidean';
    elseif x.distance == 10
        distmm = 'spearman';
    end

    Model = fitcknn(xxxTrain,yTrain,'Distance',char(distmm),'NumNeighbors',x.numNeigh,'Standardize',x.knStandard);
end
Losst = loss(Model,xxTest,yTest);
end