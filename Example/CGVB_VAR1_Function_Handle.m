% Example to fit a VAR(1) model, defined as a function handle, using CGVB method
% We simulate a multivariate time-series 

clear 
clc

rng(2021)

% Setting
m = 2;   % Number of time series
T = 100; % Number of observations

% Generate data
y = randn(2,100);

% Additional setting
setting.Prior = [0,1];    % Parameters (mean,variance) of a normal distribution
setting.y.mu = 0;
setting.idx.c = 1:m;
setting.idx.A = m+1:m+m^2;
setting.num_params = m + m^2; 
setting.Gamma = 0.1*eye(m);

%% Run CGVB with defined model
Post_CGVB_VAR1 = CGVB(@grad_h_func_VAR1,y,...
                      'NumParams',setting.num_params,... % Number of model parameters
                      'Setting',setting,...          % Additional setting to compute gradient of h(theta)
                      'LearningRate',0.002,...       % Learning rate
                      'NumSample',50,...             % Number of samples to estimate gradient of lowerbound
                      'MaxPatience',20,...           % For Early stopping
                      'MaxIter',5000,...             % Maximum number of iterations
                      'GradWeight1',0.9,...          % Momentum 1
                      'GradWeight2',0.9,...          % Momentum 2
                      'WindowSize',10,...            % Smoothing window for lowerbound
                      'GradientMax',10,...           % For gradient clipping
                      'LBPlot',false); 
                    
%% Plot variational distributions and lowerbound 
figure
% Extract variation mean and variance
mu_vb     = Post_CGVB_VAR1.Post.mu;
sigma2_vb = Post_CGVB_VAR1.Post.sigma2;

% Plot the variational distribution of each parameter
for i=1:setting.num_params
    subplot(2,4,i)
    vbayesPlot('Density',{'Normal',[mu_vb(i),sigma2_vb(i)]})
    grid on
    title(['\theta_',num2str(i)])
    set(gca,'FontSize',15)
end

% Plot the smoothed lower bound
subplot(2,4,7)
plot(Post_CGVB_VAR1.Post.LB_smooth,'LineWidth',2)
grid on
title('Lower bound')
set(gca,'FontSize',15)
                    
%% Function to compute the gradient of h(theta) and h(theta). This can be defined in a separated file
% Input: 
%       y: mxT matrix with M number of time series and T lenght of each time series
%       theta: Dx1 array of model parameters
%       setting: struct of additional information to compute gradient h(theta)
% Output:
%       grad_h_theta: Dx1 array of gradient of h(theta)
%       h_theta: h(theta) is scalar
function [grad_h_theta,h_theta] = grad_h_func_VAR1(y,theta,setting)

    % Extract size of data
    [m,T] = size(y);
    
    % Extract model settings
    prior_params = setting.Prior;
    d = setting.num_params;
    idx = setting.idx;
    Gamma = setting.Gamma;
    Gamma_inv = Gamma^(-1);

   % Extract params from theta
    c = theta(idx.c);                               % c is a Dx1 colum
    A = reshape(theta(idx.A),length(c),length(c));  % A is a DxD matrix
    
    % Log prior
    log_prior = Normal.logPdfFnc(theta,prior_params);
    
    % Log likelihood
    log_llh = 0;
    for t=2:T
        log_llh = log_llh -0.5*(y(:,t) - A*y(:,t-1)-c)' * Gamma_inv * (y(:,t) - A*y(:,t-1)-c);
    end  
    log_llh = log_llh - 0.5*m*(T-1)*log(2*pi) - 0.5*(T-1)*log(det(Gamma));

    % h(theta)
    h_theta = log_prior + log_llh;
    
    % Gradient of log prior
    grad_log_prior = Normal.GradlogPdfFnc(theta,prior_params);
    
    % Gradient of log likelihood;
    grad_llh_c = 0;
    grad_llh_A = 0;
    for t=2:T
        grad_llh_c = grad_llh_c + Gamma_inv*(y(:,t) - A*y(:,t-1)-c);
        grad_llh_A = grad_llh_A + kron(y(:,t-1),Gamma_inv*(y(:,t) - A*y(:,t-1)-c));
    end
    
    grad_llh = [grad_llh_c;grad_llh_A(:)];
    
    % Gradient h(theta)
    grad_h_theta = grad_log_prior + grad_llh;
    
    % Make sure grad_h_theta is a column
    grad_h_theta = reshape(grad_h_theta,d,1);
    
    
end    