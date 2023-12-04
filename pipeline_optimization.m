%% Pipeline Optimization
% William Baumchen
close all; clear; clc

% Suppress Table Warnings
id = 'MATLAB:table:ModifiedAndSavedVarnames';
warning('off',id)
clear id

% Verbose Output - [0 for suppression, 1 for iteration]
verboze = 0;

% Number of iterations for Bayesian hyperparameter optimization
iternn = 50;

%% Data Inport

% Import Data
datain = readmatrix("winequality-white.csv");
% Shuffle Data Entries for Splitting Data
% Set random seed for reproducibility
rng(47)
% rng(44)
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

% Instantiate results matrix
Results = zeros(72,7);
% Instantiate model storage cell array
MODELS = cell(72,2);
% Instantiate normalization state cell array
res_norm = string(zeros(72,1));
% Instantiate time counter
tickk = 0;
% Instantiate possible normalization states
normval = {'On','Off'};
% Begin loop for normalization
for normall = 1:length(normval)
    % Begin loop for number of features used (pca analysis)
    for featurenum = 11:-1:0
        % Normalize the data - with a mean of zero and standard deviation
        % of 1, if normalization is being used
        normn = normval(normall);
        if strcmp(normn,'On')
            xxxTrain = normalize(xxTrain);
            xxxTest = normalize(xxTest);
        else
            xxxTrain = xxTrain;
            xxxTest = xxTest;
        end

        if featurenum ~= 0
            % Conduct Principle Component Analysis on the training and test
            % data
            [~,scoreTrain] = pca(xxxTrain);
            xxxTrain = scoreTrain(:,1:featurenum);
            [~,scoreTrain] = pca(xxxTest);
            xxxTest = scoreTrain(:,1:featurenum);
        end


        % Increment Counter
        tickk = tickk + 1;
        % Optimize tree model
        bayesianOptionsHPO = struct('MaxObjectiveEvaluations',iternn,'Verbose',verboze,'Repartition',1,'ShowPlots',0);
        [bayesianMdlctree,bayesianResultstree] = fitcauto(xxxTrain,yyTrain,'Learners','tree',"HyperparameterOptimizationOptions",bayesianOptionsHPO,"OptimizeHyperparameters",'all');
        % Save model results
        res_norm(tickk,1) = normn;
        Results(tickk,1) = featurenum;
        MODELS{tickk,1} = bayesianMdlctree;
        MODELS{tickk,2} = bayesianResultstree;
        Results(tickk,2:6) = [bayesianResultstree.MinObjective,bayesianResultstree.MinEstimatedObjective,bayesianResultstree.TotalElapsedTime,featurenum,1];
        Results(tickk,7) = loss(bayesianMdlctree,xxxTest,yyTest);
        % Output step results
        div = ['Step ',num2str(tickk),' of 72 - Learner: tree, Time Elapsed: ',num2str(round(bayesianResultstree.TotalElapsedTime,1)),', Test Loss: ',num2str(Results(tickk,7)),', Val Loss: ',num2str(bayesianResultstree.MinObjective),', Estimated Val: ',num2str(bayesianResultstree.MinEstimatedObjective),', Feature Retention: ',num2str(featurenum),', Normalization Val: ',num2str(normall)];
        disp(div)

        % Increment Counter
        tickk = tickk + 1;
        % Optimize naive bayes model
        bayesianOptionsHPO = struct('MaxObjectiveEvaluations',iternn,'Verbose',verboze,'Repartition',1,'ShowPlots',0);
        [bayesianMdlcnb,bayesianResultsnb] = fitcauto(xxxTrain,yyTrain,'Learners','nb',"HyperparameterOptimizationOptions",bayesianOptionsHPO,"OptimizeHyperparameters",'all');
        % Save model results
        res_norm(tickk,1) = normn;
        Results(tickk,1) = featurenum;
        MODELS{tickk,1} = bayesianMdlcnb;
        MODELS{tickk,2} = bayesianResultsnb;
        Results(tickk,2:6) = [bayesianResultsnb.MinObjective,bayesianResultsnb.MinEstimatedObjective,bayesianResultsnb.TotalElapsedTime,featurenum,2];
        Results(tickk,7) = loss(bayesianMdlcnb,xxxTest,yyTest);
        % Output step results
        div = ['Step ',num2str(tickk),' of 72 - Learner: nb, Time Elapsed: ',num2str(round(bayesianResultstree.TotalElapsedTime,1)),', Test Loss: ',num2str(Results(tickk,7)),', Val Loss: ',num2str(bayesianResultstree.MinObjective),', Estimated Val: ',num2str(bayesianResultstree.MinEstimatedObjective),', Feature Retention: ',num2str(featurenum),', Normalization Val: ',num2str(normall)];
        disp(div)


        % Increment Counter
        tickk = tickk + 1;
        % Optimize knn model
        bayesianOptionsHPO = struct('MaxObjectiveEvaluations',iternn,'Verbose',verboze,'Repartition',1,'ShowPlots',0);
        [bayesianMdlcknn,bayesianResultsknn] = fitcauto(xxxTrain,yyTrain,'Learners','knn',"HyperparameterOptimizationOptions",bayesianOptionsHPO,"OptimizeHyperparameters",'all');
        % Save model results
        res_norm(tickk,1) = normn;
        Results(tickk,1) = featurenum;
        MODELS{tickk,1} = bayesianMdlcknn;
        MODELS{tickk,2} = bayesianResultsknn;
        Results(tickk,2:6) = [bayesianResultsknn.MinObjective,bayesianResultsknn.MinEstimatedObjective,bayesianResultsknn.TotalElapsedTime,featurenum,3];
        Results(tickk,7) = loss(bayesianMdlcknn,xxxTest,yyTest);
        % Output step results
        div = ['Step ',num2str(tickk),' of 72 - Learner: knn, Time Elapsed: ',num2str(round(bayesianResultstree.TotalElapsedTime,1)),', Test Loss: ',num2str(Results(tickk,7)),', Val Loss: ',num2str(bayesianResultstree.MinObjective),', Estimated Val: ',num2str(bayesianResultstree.MinEstimatedObjective),', Feature Retention: ',num2str(featurenum),', Normalization Val: ',num2str(normall)];
        disp(div)

    end
end

%% Get Best Pipeline Model

% Find best-performing model from search over inner loop test evaluation
[Bestrow,Bestcol] = find(Results(:,7)==min(Results(:,7)));
% Retrieve said model and parameters
normn = res_norm(Bestrow);
featurenum=Results(Bestrow,5);
solverr = Results(Bestrow,6);
bayesianMdlc = MODELS{Bestrow,1};
bayesianResults = MODELS{Bestrow,2};

% Scale Testing data according to results
if strcmp(normn,'On')
    x2Test = normalize(xTest);
else
    x2Test = xTest;
end

if featurenum ~=0
    % Do PCA on testing data according to results
    [coeff,scoreTrain,~,~,explained,mu] = pca(x2Test);
    x2Test = scoreTrain(:,1:featurenum);
end

%% Find Baseline

% Choose correct model and find a 'baseline' ml model
if solverr == 1
    baseline = fitctree(xxTrain,yyTrain);
elseif solverr == 2
    baseline = fitcnb(xxTrain,yyTrain);
elseif solverr == 3
    baseline = fitcknn(xxTrain,yyTrain);
end
% Evaluate model loss
bassline = crossval(baseline,'KFold',5);
baseLoss = kfoldLoss(bassline);

%% Comparison

% Compare model against a baseline approach
y2Test = loss(bayesianMdlc,x2Test,yTest);
baseError = loss(baseline,xTest,yTest);

% Plot results for varying feature reduction
figure(1);
plot(11:-1:0,Results(1:3:end/2,2))
hold on
plot(11:-1:0,Results(2:3:end/2,2))
plot(11:-1:0,Results(3:3:end/2,2))
plot(11:-1:0,Results(end/2+1:3:end,2))
plot(11:-1:0,Results(end/2+2:3:end,2))
plot(11:-1:0,Results(end/2+3:3:end,2))
hold off
legend('norm. tree','norm. nb','norm. knn','unnorm. tree','unnorm. nb','unnorm. knn')
title('Validation Loss')
axis([0 11 0.3 0.55])
ylabel('Observed Validation Classification Loss')
xlabel('Features Retained')
axis square

figure(2);
plot(11:-1:0,Results(1:3:end/2,7))
hold on
plot(11:-1:0,Results(2:3:end/2,7))
plot(11:-1:0,Results(3:3:end/2,7))
plot(11:-1:0,Results(end/2+1:3:end,7))
plot(11:-1:0,Results(end/2+2:3:end,7))
plot(11:-1:0,Results(end/2+3:3:end,7))
hold off
legend('norm. tree','norm. nb','norm. knn','unnorm. tree','unnorm. nb','unnorm. knn')
title('Test Accuracy Loss')
axis([0 11 0.3 0.7])
ylabel('Test Accuracy Loss')
xlabel('Features Retained')
axis square

%% Save Model Workspace
% save('pipelineobs.mat')
