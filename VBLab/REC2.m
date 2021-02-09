classdef REC2 < handle & matlab.mixin.CustomDisplay
    %REC2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        ModelName              % Model name 
        NumSeries              % N-variate time series   
        ParamNum               % Number of parameters
        ParamIndex             % Index of the weigths
        ParamDim               % Dimensionality of all weigth matrices
        ParamName              % Parameter names
        Params                 % Struct to store model parameters
        Post                   % Struct to store posterior samples and information about estimation
        Forecast               % Struct to store forecast results
        LogLikelihood          % Handle of the log-likelihood function
        ParamValues            % Current values of all parameters
        CorModel               % The NN to model the correlation
        NNModel
        VarModel
        GARCHModel             % The univariate GARCH model to model conditional variance
        FNNStruct              % Structure of FNN
        RNNStruct              % Structure of RNN
        NumHidden              % Number of hidden states in RNN
        InputDim               % Input dimension
        OutputDim              % Output dimension 
        Prior                  % Prior
        Activation             % Activation function used in the NN component
        CorrMat                % Correlation matrix
        CorrMatInit            % Initial correlation matrix
        LMat                   % Lower triagular matrix
        LMatInit
        CovMat                 % Conditional variance matrix
        LowMat                 % Lt
        LowMatInit             % Lt_0
        VarVec                 % lt
        VarVecInit             % Initial values of lt_0
    end
    
    methods
        function obj = REC2(numSeries,varargin)
            %REC2 Construct an instance of this class
            %   Detailed explanation goes here
            
            obj.ModelName = 'REC2';
            obj.NumSeries = numSeries;
            obj.NNModel = 'SRN';

            if nargin > 1
                %Parse additional options
                paramNames = {'CorrMatInit'     'VarVec'          'GARCHModel'        'CorModel'   ...
                              'FNNStruct'       'RNNStruct'       'NumHidden'         'Activation'     ...
                              'VarVecInit'      'LMatInit'        'NNModel'};
                paramDflts = {obj.CorrMatInit   obj.VarVec        obj.GARCHModel      obj.CorModel  ...
                              obj.FNNStruct     obj.RNNStruct     obj.NumHidden       obj.Activation   ...
                              obj.VarVecInit    obj.LMatInit      obj.NNModel};

                [obj.CorrMatInit,...
                 obj.VarVec,...
                 obj.GARCHModel,...
                 obj.CorModel,...
                 obj.FNNStruct,...
                 obj.RNNStruct,...
                 obj.NumHidden,...
                 obj.Activation,...
                 obj.VarVecInit,...
                 obj.LMatInit,...
                 obj.NNModel] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});
            end
            
            % Prepare the Neural network for the correlation dynamic
            if (strcmp(obj.NNModel,'FNN'))
                % Set default FNN structure
                if (isempty(obj.FNNStruct))
                    obj.FNNStruct = [2,2];
                end
                obj.CorModel = FNN(numSeries+numSeries+1,numSeries,obj.FNNStruct,varargin);
            elseif (strcmp(obj.NNModel,'SRN'))
                if (isempty(obj.NumHidden))
                    obj.NumHidden = 3;
                end
                obj.CorModel = SRN(numSeries+numSeries,numSeries,obj.NumHidden);
            end
            [obj.ParamNum,...
             obj.ParamDim,...
             obj.ParamIndex] = obj.CorModel.getParamNum();
            
            if(~isempty(obj.CorrMatInit))
                obj.LowMatInit = chol(obj.CorrMatInit)';  % Lt_0
                obj.VarVecInit = obj.LowMatInit(2,:)';    % lt_0
            end
        end
        
        function [params_num, param_dim, param_idx] = getParamNum(obj)
            %GETPARAMNUM Calculate the number of parameters of the FNN
            NN_model = obj.CorModel;
            [params_num, param_dim, param_idx] = NN_model.getParamNum();
        end
        
        %% Compute log likelihood
        function [llh,Rt] = logLik(obj,y,params_vec)
            
            % Contruct parameter in structs
            params = obj.paramToStruct(params_vec);
            
            % Unload settings
            NN_model = obj.CorModel;
            [D,T]    = size(y);
            ht       = obj.VarVec;
            Rt_0     = obj.CorrMatInit;
            lt_0     = obj.VarVecInit;
            
            % Pre-allocation
            llh       = 0;
            Rt        = zeros(D,D,T);
            Rt(:,:,1) = Rt_0;
            lt        = zeros(D,T);
            lt(:,1)   = lt_0;
            
            % Hidden state

            for t = 2:T
                % Input for NN
%                 X_t = [1;y(:,t-1)/norm(y(:,t-1));lt(:,t-1)];
%                 X_t = [1;y(:,t-1);lt_temp];
                
%                 eps_t = diag(ht(:,t).^-1)*y(:,t-1);
                X_t = [y(:,t-1)/norm(y(:,t-1));lt(:,t-1)];
                
                % Doing a forward pass
                [output,NN_model.StateCur] = NN_model.forwardPass(X_t,params);

                % Normalize the output to obtain unit vector
                lt(:,t) = output/norm(output);
            
                % Reconstruct correlation matrix Rt
                Lt = [1,0;lt(:,t)'];
                Rt(:,:,t) = Lt*Lt';

                % Reconstruct covariance matrix Ht
                Ht = Rt(:,:,t).*(ht(:,t)*ht(:,t)');

                % likelihood
                llh = llh - 0.5*D*log(2*pi) - 0.5*log(det(Ht)) - 0.5*y(:,t)'*Ht^-1*y(:,t);
            end
        end        
        
        %% Compute gradient of log likelihood
        function [grad_llh,llh] = logLikGrad(obj,y,params)
            
        end
        
        %% Compute log prior of weights
        function log_prior = logPriors(obj,params)
            NN_model  = obj.CorModel;
            log_prior = NN_model.logPriors(NN_model.Prior,params);
        end
        
        %% Compute gradient of log prior of weights
        function log_prior = logPriorsGrad(obj,params)
            NN_model  = obj.CorModel;
            log_prior = NN_model.logPriors(NN_model.Prior,params);
        end
        
        %% Ravel the weigth matrices
        function params_vec = paramToVec(obj,params)
            NN_model = obj.CorModel;
            params_vec = NN_model.paramToVec(params);
        end
        
        %% Convert vector of params to struct  
        function params = paramToStruct(obj,params_vec)
            NN_model = obj.CorModel;
            params   = NN_model.paramToStruct(params_vec);
        end
        
        %% Transform parameters to normal distribution
        function paramsNormal = toNormalParams(obj,params)
            NN_model = obj.CorModel;
            paramsNormal = NN_model.toNormalParams(params);
        end

        %% Transform parameters to from normal to original distribution
        % Input: 
        %   - params: the row vector of parameters
        function paramsNormal = toOriginalParams(obj,params)
            NN_model = obj.CorModel;
            paramsNormal = NN_model.toNormalParams(params);
        end
        
        %% Log of Jacobian of all paramters
        % Input: 
        %   - params: the row vector of parameters
        function logjac = logJac(obj,params)
            NN_model = obj.CorModel;
            logjac = NN_model.logJac(params);
        end

        %% Gradient of log Jacobian
        % Input: 
        %   - params: the row vector of parameters
        function logjac = logJacGrad(obj,params)
            NN_model = obj.CorModel;
            logjac = NN_model.logJac(params);
        end
        
        %% Initialize parameters from prior.
        function params = initParams(obj,NumVal)
            NN_model = obj.CorModel;
            params   = NN_model.initParams(NumVal);
        end
        
    end
end

