classdef Beta < Distribution
    %NORMAL Class to compute quantities related to normal distribution
    
    properties
    end
    
    methods (Static)
        
        %% Random number generator
        % dim is [D,1]
        function random_num = rngFnc(params,dim)
            random_num = random('Beta',params,dim);
        end
        
        %% Log pdf function
        % input: 
        %        x: Dx1 
        %        params = [mean(scalar),variance(scalar)]
        % output:
        %        log of pdf function(scalar)
        function log_pdf = logPdfFnc(x,params)
        end

        %% Gradient of log pdf function
        % input: 
        %        x: Dx1 
        %        params = [mean(scalar),variance(scalar)]
        % output:
        %        gradient of log pdf function: Dx1
        function grad_log_pdf = GradlogPdfFnc(x,params)
        end

        %% Log jacobian
        % input: 
        %        x: Dx1 
        %        params = [mean(scalar),variance(scalar)]
        % output:
        %        log Jacobian of transformation (scalar)
        function log_jac = logJacFnc(x,params)
            log_jac = 0;
        end
        
        %% Gradient of log jacobian
        % input: 
        %        x: Dx1 
        %        params = [mean(scalar),variance(scalar)]
        % output:
        %        gradient of log Jacobian of transformation: Dx1 
        function grad_log_jac = GradlogJacFnc(x,params)
        end
        
        %% Transform parameters to normal distribution
        function params_normal = toNormalParams(params)
            params_normal = untils_logit(params);
        end

        %% Tranform normal parameters to original distribution
        function params_ori = toOriginalParams(params)
            params_ori = utils_sigmoid(params);
        end
    end
end

