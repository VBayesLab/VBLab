classdef SRN
    %SRN Summary of this class goes here
    %   Detailed explanation goes here
    
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
        NumHidden              % Hidden-to-hidden structures
        InputDim               % Input dimension
        OutputDim              % Output dimension 
        Prior                  % Prior
        Activation             % Activation function
        TimeCur
        StateInit
        StateCur
    end
    
    methods
        function obj = SRN(x_dim,y_dim,h_dim,varargin)
            %SRN Construct an instance of this class
            %   Detailed explanation goes here
            obj.ModelName  = 'SRN';
            obj.InputDim   = x_dim;
            obj.OutputDim  = y_dim;
            obj.NumHidden  = h_dim;
            obj.Activation = @tanh;
            obj.ParamName  = {'V','W','b','beta0','beta1'};
            obj.TimeCur    = 1;
            obj.StateCur   = zeros(h_dim,1);

            % Prior setting
            obj.Prior.V_mu     = 0;     obj.Prior.V_var     = 0.01;          % Normal distribution
            obj.Prior.W_mu     = 0;     obj.Prior.W_var     = 0.01;          % Normal distribution
            obj.Prior.b_mu     = 0;     obj.Prior.b_var     = 0.01;          % Normal distribution
            obj.Prior.beta0_mu = 0;     obj.Prior.beta0_var = 0.01;          % Normal distribution
            obj.Prior.beta1_mu = 0;     obj.Prior.beta1_var = 0.01;          % Normal distribution

            % Compute some training settings
            [obj.ParamNum,...
             obj.ParamDim,...
             obj.ParamIndex] = obj.getParamNum();            
        end
        
        function [params_num, param_dim, param_idx] = getParamNum(obj)
            %GETPARAMNUM Calculate the number of parameters of the FNN
            
            % V
            param_dim.V  = [obj.InputDim,obj.NumHidden];
            num_params_V = param_dim.V(1)*param_dim.V(2);
            
            idx_start   = 0;
            idx_stop    = idx_start + num_params_V;
            param_idx.V = [idx_start+1,idx_stop];
            idx_start   = idx_stop; 
            
            % W
            param_dim.W  = [obj.NumHidden,obj.NumHidden];
            num_params_W = param_dim.W(1)*param_dim.W(2);     % w for input->hidden

            idx_stop    = idx_start + num_params_W;
            param_idx.W = [idx_start+1,idx_stop];
            idx_start   = idx_stop;  

            % b
            param_dim.b  = [obj.NumHidden,1];
            num_params_b = param_dim.b(1)*param_dim.b(2);     % w for input->hidden

            idx_stop    = idx_start + num_params_b;
            param_idx.b = [idx_start+1,idx_stop];
            idx_start   = idx_stop;              
            
            % beta0
            param_dim.beta0  = [obj.OutputDim,1];
            num_params_beta0 = param_dim.beta0(1)*param_dim.beta0(2);  % w for input->hidden

            idx_stop    = idx_start + num_params_beta0;
            param_idx.beta0 = [idx_start+1,idx_stop];
            idx_start   = idx_stop; 
            
            % beta1
            param_dim.beta1 = [obj.OutputDim,obj.NumHidden];
            num_params_beta1 = param_dim.beta1(1)*param_dim.beta1(2);     % w for input->hidden

            idx_stop    = idx_start + num_params_beta1;
            param_idx.beta1 = [idx_start+1,idx_stop];
            
            params_num = num_params_V +...
                         num_params_W +...
                         num_params_b + ...
                         num_params_beta0 + ...
                         num_params_beta1;
        end
        
        %% Compute log prior of weights
        function log_prior = logPriors(obj,prior,params)
%             log_prior = sum(utils_logNormalpdf(params.V(:),prior.V_mu,prior.V_var))+...
%                         sum(utils_logNormalpdf(params.W(:),prior.W_mu,prior.W_var))+...
%                         sum(utils_logNormalpdf(params.b,prior.b_mu,prior.b_var))+...
%                         sum(utils_logNormalpdf(params.beta0,prior.beta0_mu,prior.beta0_var))+...
%                         sum(utils_logNormalpdf(params.beta1(:),prior.beta1_mu,prior.beta1_var));
                    
            log_prior = sum(utils_logNormalpdf(params,prior.W_mu,prior.W_var),2);
        end
        
        %% Compute forward pass of the RNN
        function [output,StateNew] = forwardPass(obj,input,params)
            
            % From input -> Hidden
            StateNew  = obj.Activation(params.V'*input + params.W*obj.StateCur + params.b);
            output = params.beta0 + params.beta1*StateNew;

        end
        
        %% Ravel the weigth matrices
        function params_vec = paramToVec(obj,params)
            
            params_vec = [];
            for i=1:length(obj.ParamName)
                params_vec = [params_vec,params.(obj.ParamName{i})(:)'];
            end
            
        end
        
        %% Convert vector of params to struct 
        % Input: 1D array
        % Output: Struct
        function params = paramToStruct(obj,params_vec)
            % For weight input-> hidden
            params_idx = obj.ParamIndex;
            params_dim = obj.ParamDim;
            
            for i=1:length(obj.ParamName)
                range = params_idx.(obj.ParamName{i});
                dim_i = params_dim.(obj.ParamName{i});
                params.(obj.ParamName{i}) = reshape(params_vec(range(1):range(2)),dim_i);
            end
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
                params.V     = normrnd(prior.V_mu,sqrt(prior.V_var),param_dim.V(1),param_dim.V(2));
                params.W     = normrnd(prior.W_mu,sqrt(prior.W_var),param_dim.W(1),param_dim.W(2));
                params.b     = normrnd(prior.b_mu,sqrt(prior.b_var),param_dim.b(1),param_dim.b(2));
                params.beta0 = normrnd(prior.beta0_mu,sqrt(prior.beta0_var),param_dim.beta0(1),param_dim.beta0(2));
                params.beta1 = normrnd(prior.beta1_mu,sqrt(prior.beta1_var),param_dim.beta1(1),param_dim.beta1(2));                
                params_vec(idx,:) = obj.paramToVec(params);
            end
        end
        
    end
end

