% Script to run the Example 3.4 using a built-in Logistic Regression class of
% the VBLab package

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

% Additional Setting
setting.Prior = [0,50];

% Initialize using MLE estimate (for quickly converging)
X = labour(:,1:end-1);
y = labour(:,end);
theta_init = glmfit(X,y,'binomial','constant','off'); % initialise mu

% Run CGVB
Post_CGVB_manual = CGVB(@grad_h_func_logistic,labour,...
                        'NumParams',n_features,...
                        'Setting',setting,...
                        'LearningRate',0.002,...   % Learning rate
                        'NumSample',50,...         % Number of samples to estimate gradient of lowerbound
                        'MaxPatience',20,...       % For Early stopping
                        'MaxIter',5000,...         % Maximum number of iterations
                        'MeanInit',theta_init ,... % Randomly initialize parameters using 
                        'GradWeight1',0.9,...      % Momentum 1
                        'GradWeight2',0.9,...      % Momentum 2
                        'WindowSize',10,...        % Smoothing window for lowerbound
                        'GradientMax',10,...       % For gradient clipping
                        'LBPlot',true); 
                    
%% Plot variational distributions and lowerbound 
figure
% Extract variation mean and variance
mu_vb     = Post_CGVB_manual.Post.mu;
sigma2_vb = Post_CGVB_manual.Post.sigma2;

% Plot the variational distribution for the first 8 parameters
for i=1:8
    subplot(3,3,i)
    vbayesPlot('Density',{'Normal',[mu_vb(i),sigma2_vb(i)]})
    grid on
    title(['\theta_',num2str(i)])
    set(gca,'FontSize',15)
end

% Plot the smoothed lower bound
subplot(3,3,9)
plot(Post_CGVB_manual.Post.LB_smooth,'LineWidth',2)
grid on
title('Lower bound')
set(gca,'FontSize',15)

%% Define gradient of h function for Logistic regression 
% theta: Dx1 array
% h_func: Scalar
% h_func_grad: Dx1 array
function [h_func_grad,h_func] = grad_h_func_logistic(data,theta,setting)

    % Extract additional settings
    d = length(theta);
    sigma2 = setting.Prior(2);
    
    % Extract data
    X = data(:,1:end-1);
    y = data(:,end);
    
    % Compute log likelihood
    aux = X*theta;
    llh = y.*aux-log(1+exp(aux));
    llh = sum(llh);  
    
    % Compute gradient of log likelihood
    ppi       = 1./(1+exp(-aux));
    llh_grad  = X'*(y-ppi);

    % Compute log prior
    log_prior =-d/2*log(2*pi)-d/2*log(sigma2)-theta'*theta/sigma2/2;
    
    % Compute gradient of log prior
    log_prior_grad = -theta/sigma2;
    
    % Compute h(theta) = log p(y|theta) + log p(theta)
    h_func = llh + log_prior;
    
    % Compute gradient of the h(theta)
    h_func_grad = llh_grad + log_prior_grad;

    % h_func_grad must be a column
    h_func_grad = reshape(h_func_grad,length(h_func_grad),1);

end