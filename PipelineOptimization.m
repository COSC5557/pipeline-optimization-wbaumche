%% Pipeline Optimization
% William Baumchen
close all; clear; clc

% Verbose Output - [0 for suppression, 1 for iteration]
verboze = 1;

% Number of iterations for Bayesian hyperparameter optimization
iternn = 125;

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
% Split Data
% Set Fraction of Training Entries for Training Test Set
a = 0.1;
% Split Data
xxTest = xTrain(1:round(a*size(xTrain,1)),:);
yyTest = yTrain(1:round(a*size(yTrain,1)),:);
xxTrain = xTrain(round(a*size(xTrain,1))+1:end,:);
yyTrain = yTrain(round(a*size(yTrain,1))+1:end,:);


%% Pipeline Optimization

% Instantiate Optimizable hyperparameters for pipeline
norm = optimizableVariable('normVal',[0,1],'Type','integer');
feat = optimizableVariable('featureNum',[0,11],'Type','integer');
solv = optimizableVariable('solver',[0,2],'Type','integer');
trhp1 = optimizableVariable('minLeaf',[1,max(2,height(xxTrain)-1)],'Type','integer');
knhp1 = optimizableVariable('distance',[0,10],'Type','integer');
knhp2 = optimizableVariable('numNeigh',[1,max(2,round(height(xxTrain)/2))],'Type','integer');
knhp3 = optimizableVariable('knStandard',[0,1],'Type','integer');
enhp1 = optimizableVariable('Method',{'Bag','AdaBoostM2','RUSBoost'},'Type','categorical');

% Create function
fun = @(x)pipopt(x,xxTrain,xxTest,yyTest,yyTrain);
% Assemble hyperparameter variables
vars = [norm,feat,solv,trhp1,knhp1,knhp2,knhp3,enhp1];
% Run Bayesian Optimization
% results = bayesopt(fun,vars,'Verbose',verboze,'MaxObjectiveEvaluations',iternn,'AcquisitionFunctionName','expected-improvement-plus')
results = bayesopt(fun,vars,'Verbose',verboze,'MaxObjectiveEvaluations',iternn,'AcquisitionFunctionName','probability-of-improvement')


%% Get Best Pipeline Model

% Retrieve best model, parameters, and results
[resl,Model] = pipopt(results.XAtMinObjective,xxTrain,xTest,yTest,yyTrain);

% Find Baseline
solverr = results.XAtMinObjective.(3);
% Choose correct model and find a 'baseline' ml model
if solverr == 1
    baseline = fitctree(xxTrain,yyTrain);
elseif solverr == 2
    baseline = fitcensemble(xxTrain,yyTrain);
elseif solverr == 3
    baseline = fitcknn(xxTrain,yyTrain);
end
% Evaluate model loss
bassline = crossval(baseline,'KFold',5);
baseLoss = kfoldLoss(bassline);

%% Comparison

% Compare model against a baseline approach
baseError = loss(baseline,xTest,yTest);
Valerror = mean(kfoldLoss(Model,'Mode','individual'));

figure(1)
confusionchart(yTest,predict(baseline,xTest))
title('Baseline Prediction')
figure(2)
confusionchart(yTest,predict(Model.Trained{1},xTest))
title('Model Prediction')

figure(3)
plot(1:iternn,results.ObjectiveTrace)
title('Optimization Evaluation')
xlabel('Iteration Number')
ylabel('Average Cross-Validated Classification Loss')
resx = results.ObjectiveTrace;
resobj = [1:125]';
res = [resobj,resx];
bbres = sortrows(res,2);
xTrace = results.XTrace;
for i = 1:125
score1(i,:) = xTrace(bbres(i,1),:);
end

%% Save Model Workspace
% save('pipelineobs2.mat')

%% Autonomous function:
function [Result,Model] = pipopt(x,xxTrain,xxTest,yyTest,yyTrain)
% Normalization
if x.normVal == 1
    xxxTrain = normalize(xxTrain);
    xxxTest = normalize(xxTest);
else
    xxxTrain = xxTrain;
    xxxTest = xxTest;
end

% PCA
if x.featureNum ~= 0
    [~,scoreTrain] = pca(xxxTrain);
    xxxTrain = scoreTrain(:,1:x.featureNum);
    [~,scoreTrain] = pca(xxxTest);
    xxxTest = scoreTrain(:,1:x.featureNum);
end

if x.solver == 0
    % Optimize tree model
    Model = fitctree(xxxTrain,yyTrain,'KFold',5,'MinLeafSize',x.minLeaf);
elseif x.solver == 1

    Model = fitcensemble(xxxTrain,yyTrain,'KFold',5,'Method',char(x.Method));

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

    Model = fitcknn(xxxTrain,yyTrain,'KFold',5,'Distance',char(distmm),'NumNeighbors',x.numNeigh,'Standardize',x.knStandard);
end

mses = zeros(5,1);
for i = 1:5
    mses(i) = loss(Model.Trained{i},xxxTest,yyTest);
end

Result = mean(mses);
end