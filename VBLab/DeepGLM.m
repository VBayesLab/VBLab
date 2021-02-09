classdef DeepGLM
    % DEEPGLM Class to define deepGLM model
    
    properties
        ModelName      % Model name 
        NumParams      % Total number of parameters
        NumWeight      % Number of weights until the last hidden layer
        NumBeta        % Number of weights from the last to the output node
        PriorInput     % Prior specified by users
        Prior          % Prior object
        PriorVal       % Parameters of priors  
        Intercept      % Option to add intercept or not (only for testing)
        AutoDiff       % Option to use autodiff (only for testing)
        Post           % Struct to store training results (maybe not used)
    end
    
    methods
        %% Constructor
        function obj = DeepGLM(n_features,n_units, varargin)
            %DEEPGLM Construct an instance of this class
            %   Detailed explanation goes here
            
            
            % Compute number of parameters
            L = length(n_units); % The number of hidden layers
            p = n_features; % Number of covariates
            W_seq = cell(1,L); % cells to store weight matrices
            index_track = zeros(1,L); % keep track of indices of Wj matrices: index_track(1) is the total elements in W1, index_track(2) is the total elements in W1 & W2,...
            index_track(1) = n_units(1)*(p+1); % size of W1 is m1 x (p+1) with m1 number of units in the 1st hidden layer 
            W1_tilde_index = n_units(1)+1:index_track(1); % index of W1 without biases, as the first column if W1 are biases
            for j = 2:L
                index_track(j) = index_track(j-1)+n_units(j)*(n_units(j-1)+1);
            end
            obj.NumWeight = index_track(L);             % the total number of weights up to (and including) the last layer
            obj.NumBeta = n_units(L)+1;                 % dimension of the weights beta connecting the last layer to the output
            obj.NumParams = obj.NumWeight + obj.NumBeta; % the total number of parameters
            
        end
        
        %% Log likelihood
        % Input: 
        %   - params: the ROW vector of parameters
        % Output: 
        %   - llh: Log likelihood of the model
        function llh = logLik(obj,data,params)
            
        end

        %% Compute gradient of Log likelihood
        % Input: 
        %   - params: Dx1 vector of parameters
        % Output: 
        %   - llh_grad: Log likelihood of the model
        function [llh_grad,llh] = logLikGrad(obj,data,theta)
        end
 
        %% Compute gradient of Log likelihood using AutoDiff
        % Input: 
        %   - params: 1xD vector of parameters
        % Output: 
        %   - llh_grad: Log likelihood of the model
        function [llh_grad,llh] = logLikGradAutoDiff(obj,data,params)
        end
        
        %% Compute log prior of parameters
        % Input: 
        %   - params: the Dx1 vector of parameters
        % Output: 
        %   - llh: Log prior of model parameters       
        function log_prior = logPriors(obj,params)
        end  
        
        %% Compute gradient of log prior of parameters
        % Input: 
        %   - params: 1xD vector of parameters
        % Output: 
        %   - log_prior_grad: Gradient of log prior of model parameters       
        function [log_prior_grad,log_prior] = logPriorsGrad(obj,params)
        end        
        
        %% Log of Jacobian of all paramters
        % Input: 
        %   - params: the ROW vector of parameters
        % Output: 
        %   - llh: Log prior of model parameters  
        function logjac = logJac(obj,params)
            logjac = 0;
        end
        
        %% Log of Jacobian of all paramters
        % Input: 
        %   - params: the ROW vector of parameters
        % Output: 
        %   - llh: Log prior of model parameters  
        function [logJac_grad,logJac] = logJacGrad(obj,params)
            logJac_grad = 0;
            logJac      = 0;
        end
        
        %% Compute loss on a validation data (optional)
        function loss = predictLoss(obj,data,params)
            
        end
        
        %% Function to compute h_theta = log lik + log prior
        function h_func = hFunction(obj,data,theta)
            log_lik = obj.logLik(data,theta);
            log_prior = obj.logPriors(theta);
            log_jac = obj.logJac(params);
            h_func = log_lik + log_prior + log_jac;
        end
        
        %% Function to compute gradient of h_theta = grad log lik + grad log prior
        function [h_func_grad, h_func] = hFunctionGrad(obj,data,theta)
            [llh_grad,llh] = obj.logLikGrad(data,theta);
            [log_prior_grad,log_prior] = obj.logPriorsGrad(theta);
            [logJac_grad,logJac] = obj.logJacGrad(theta);
            h_func = llh + log_prior + logJac;
            h_func_grad = llh_grad + log_prior_grad + logJac_grad;
        end        
        
        %% Transform parameters to from normal to original distribution
        function paramsOriginal = toOriginalParams(obj,params)
            paramsOriginal = obj.Prior.toOriginalParams(params);
        end

        %% Transform parameters to from normal to original distribution
        function paramsNormal = toNormalParams(obj,params)
            paramsNormal = obj.Prior.toNormalParams(params);
        end
        
        %% Initialize parameters
        % layers: vector of doubles, each number specifing the amount of
        % nodes in a layer of the network.
        %  
        % weights: cell array of weight matrices specifing the
        % translation from one layer of the network to the next.
        function weights = initParams(obj,type,layers)
            weights = cell(1,length(layers)-1);
            switch type
                case 'Random'
                    for i = 1:length(layers)-1
                        % Using random weights from -b to b 
                        b = sqrt(6)/(layers(i)+layers(i+1));
                        if i==1
                            weights{i} = rand(layers(i+1),layers(i))*2*b - b;  % Input layer already have bias
                        else
                            weights{i} = rand(layers(i+1),layers(i)+1)*2*b - b;  % 1 bias in input layer
                        end
                    end
                otherwise
                    error(['There is no initialization method called ',type,' in the model object!'])
            end
        end
    end
end

