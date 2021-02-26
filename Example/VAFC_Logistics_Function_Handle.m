% Script to fit a Logistic Regression model, defined as a function handle,
% using the VAFC method

clear  % Clear all variables 
clc % Clean workspace

% Random seed to reproduce results 
rng(2020)

% Load the GermanCredit dataset
credit = readData('GermanCredit',...    % Dataset name
                  'Type','Matrix',...   % Store data as a 2D array (default)
                  'Intercept', true);   % Add column of intercept (default)

% Compute number of features
n_features = size(credit,2)-1;

% Additional Setting
setting.Prior = [0,50];

% Initialize using MLE estimate
X = credit(:,1:end-1);
y = credit(:,end);

% Run CGVB
Post_VAFC_manual = VAFC(@grad_h_func_logistic,credit,...
                        'NumParams',n_features,...
                        'Setting',setting,...
                        'NumFactor',4, ...       % Number of factors of the loading matrix
                        'NumSample',100,...      % Number of samples to estimate gradient of lowerbound
                        'LearningRate',0.05,...  % Learning rate
                        'MaxPatience',30,...     % For Early stopping
                        'MaxIter',10000,...      % Maximum number of iterations
                        'GradientMax',200,...    % For gradient clipping    
                        'WindowSize',20, ...     % Smoothing window for lowerbound
                        'LBPlot',true);          % Dont plot the lowerbound when finish

%% Plot variational distributions and lowerbound 
figure
% Extract variation mean and variance
mu_vb     = Post_VAFC_manual.Post.mu;
sigma2_vb = Post_VAFC_manual.Post.sigma2;

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
plot(Post_VAFC_manual.Post.LB_smooth,'LineWidth',2)
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