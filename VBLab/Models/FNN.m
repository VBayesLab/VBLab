classdef FNN < handle & matlab.mixin.CustomDisplay
    %FNN Class of Feedforward Neural Network
    
    properties
        ModelName              % Model name 
        ParamNum               % Number of parameters
        ParamIndex             % Index of the weigths
        ParamDim               % Dimensionality of all weigth matrices
        ParamName              % Parameter names
        Params                 % Struct to store model parameters
        Post                   % Struct to store posterior samples and information about estimation
        Forecast               % Struct to store forecast results
        LogLikelihood          % Handle of the log-likelihood function
        ParamValues            % Current values of all parameters
        NN                     % Hidden-to-hidden structures
        InputDim               % Input dimension
        OutputDim              % Output dimension 
        Prior                  % Prior
        Activation             % Activation function
    end
    
    methods
        
        function obj = FNN(x_dim,y_dim,nn,varargin)
            %FNN Construct an instance of this class
            %   Detailed explanation goes here
            obj.ModelName   = 'FNN';
            obj.InputDim    = x_dim;
            obj.OutputDim   = y_dim;
            obj.NN          = nn;
            obj.Activation  = @utils_relu;
            obj.ParamName   = {'W_in','W','W_out'};
 
            % Priors
            obj.Prior.W_mu  = 0;         
            obj.Prior.W_var = 0.01;
            
            % Compute some training settings
            [obj.ParamNum,...
             obj.ParamDim,...
             obj.ParamIndex] = obj.getParamNum();
        end
        
        function [params_num, param_dim, param_idx] = getParamNum(obj)
            %GETPARAMNUM Calculate the number of parameters of the FNN
            param_dim.W_in  = [obj.InputDim,obj.NN(1)];
            num_params_W_in = param_dim.W_in(1)*param_dim.W_in(2);     % w for input->hidden
            idx_start       = 0;
            idx_stop        = idx_start + num_params_W_in;
            param_idx.W_in  = [idx_start+1,idx_stop];
            idx_start       = idx_stop;  
            
            num_params_W_hidden = 0;
            if (length(obj.NN)>1)
                for i=1:length(obj.NN)-1
                    param_dim.W(i,:) = [obj.NN(i)+1,obj.NN(i+1)];
                    num_params_W_hidden  = num_params_W_hidden + param_dim.W(i,1)*param_dim.W(i,2);  % w for hidden-> hidden
                    idx_stop         = idx_start + param_dim.W(i,1)*param_dim.W(i,2);
                    param_idx.W(i,:) = [idx_start+1,idx_stop];
                    idx_start        = idx_stop;
                end    
            else
                param_idx.W = [];
                param_dim.W = [];
            end
            
            param_dim.W_out  = [obj.NN(end)+1,obj.OutputDim];
            num_params_W_out = param_dim.W_out(1)*param_dim.W_out(2);
            idx_stop         = idx_start + num_params_W_out;
            param_idx.W_out  = [idx_start+1,idx_stop];
            
            params_num = num_params_W_in + num_params_W_hidden + num_params_W_out;
        end
        
        % Compute log prior of weights
        % params is matrix with rows are samples
        function log_prior = logPriors(obj,prior,params)
%             W_vech = params.W_in(:);
%             for i=1:length(params.W)
%                 W_vech = [W_vech;params.W{i}(:)];
%             end
%             W_vech = [W_vech;params.W_out(:)];
%             log_prior = sum(utils_logNormalpdf(W_vech,prior.W_mu,prior.W_var));
            log_prior = sum(utils_logNormalpdf(params,prior.W_mu,prior.W_var),2);
        end
        
        % Compute forward pass of the FNN
        function output = forwardPass(obj,input,params)
            
            % From input -> Hidden
            a = params.W_in'*input;
            a_activated = [1;obj.Activation(a)];

            % Forward pass between hidden layers
            for i=1:length(params.W)
                a = params.W{i}'*a_activated;
                a_activated = [1;obj.Activation(a)]; 
            end

            % From hidden-> output
            a = params.W_out'*a_activated;

            % Normalize the output to obtain unit vector
            output = a/norm(a);
        end
        
        % Ravel the weigth matrices
        function params_vec = paramToVec(obj,params)
            params_vec = params.W_in(:)';
            for i = 1:length(params.W)
                params_vec = [params_vec,params.W{i}(:)'];
            end
            
            params_vec = [params_vec,params.W_out(:)'];
        end
        
        %% Convert vector of params to struct 
        % Input: 1D array
        % Output: Struct
        function params = paramToStruct(obj,params_vec)
            % For weight input-> hidden
            params_idx = obj.ParamIndex;
            params_dim = obj.ParamDim;
            range = params_idx.W_in;
            dim_i = params_dim.W_in;
            
            params.W_in = reshape(params_vec(range(1):range(2)),dim_i);

            % For weigths hidden -> hidden
            if(~isempty(params_idx.W))
                nrow = size(params_idx.W,1);
                for j=1:nrow % Each row is the vech of a weight matrix
                    range = params_idx.W(j,:);
                    dim_i = params_dim.W(j,:);
                    params.W{j} = reshape(params_vec(range(1):range(2)),dim_i);
                end
            else
                params.W = {};
            end

            % For weigths hidden-> output
            range = params_idx.W_out;
            dim_i = params_dim.W_out;
            params.W_out = reshape(params_vec(range(1):range(2)),dim_i);
        end
        
        %% Transform parameters to normal distribution
        % Input: 1D array (row)
        % Output: 1D array (row)        
        function paramsNormal = toNormalParams(obj,params)
            paramsNormal = params;
        end
        
        %% Log of Jacobian of all paramters
        % Input: 2D array -> each particle is a row
        % Output: 1D array (column)
        function log_jac = logJac(obj,params)
            [nrow,~] = size(params);
            log_jac = zeros(nrow,1);
        end
        
        %% Initialize parameters from prior.
        % Input: Number of particle
        % Output: 2D array -> each row is a particle cluster of array of params
        function params_vec = initParams(obj,numVal)
            
            prior     = obj.Prior;
            param_dim = obj.ParamDim;
            param_num = obj.ParamNum;
            
            params_vec = zeros(numVal,param_num);
            for idx=1:numVal
                params.W_in  = normrnd(prior.W_mu,sqrt(prior.W_var),param_dim.W_in(1),param_dim.W_in(2));            
                for i=1:size(param_dim.W,1)
                    params.W{i} = normrnd(prior.W_mu,sqrt(prior.W_var),param_dim.W(i,1),param_dim.W(i,2)); % +1 for intercept   
                end    
                params.W_out = normrnd(prior.W_mu,sqrt(prior.W_var),param_dim.W_out(1),param_dim.W_out(2));
                params_vec(idx,:) = obj.paramToVec(params);
            end
        end
        
    end
end

