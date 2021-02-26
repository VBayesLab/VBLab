% Script to fit the Logistic Regression modell using the MGVB method 

clear  % Clear all variables 
clc % Clean workspace

% Random seed to reproduce results 
rng(2020)

% Load the LabourForce dataset
labour = readData('LabourForce',...     % Dataset name
                  'Type','Matrix',...   % Store data as a 2D array (default)
                  'Intercept', true);   % Add column of intercept (default)

% Compute number of features
n_features = size(labour,2)-1;

% Create a Logistic Regression model object
Mdl = LogisticRegression(n_features,...
                        'Prior',{'Normal',[0,50]});
                          
%% Run Cholesky GVB to approximate the posterior distribution of model 
% using a multivariate normal density
Post_CGVB = MGVB(Mdl,labour,...
                'LearningRate',0.001,... % Learning rate
                'NumSample',100,...      % Number of samples to estimate gradient of lowerbound 
                'MaxPatience',50,...     % For Early stopping
                'MaxIter',2000,...       % Maximum number of iterations
                'GradWeight',0.4,...     % Momentum weight
                'WindowSize',30,...      % Smoothing window for lowerbound
                'SigInitScale',0.04,...  % Std of normal distribution for initializing  
                'StepAdaptive',500,...   % For adaptive learning rate   
                'GradientMax',100,...    % For gradient clipping     
                'LBPlot',true);          % Plot the smoothed lowerbound at the end






