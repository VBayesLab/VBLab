classdef Normal < Distribution
    %NORMAL Class to compute quantities related to normal distribution
    
    properties
    end
    
    methods (Static)
        
        %% Random number generator
        function random_num = rngFnc(params,dim)
            random_num = normrnd(params(1),sqrt(params(2)),dim);
        end
        
        %% Log pdf function
        % input: 
        %        x: Dx1 
        %        params = [mean(scalar),variance(scalar)]
        % output:
        %        log of pdf function(scalar)
        function log_pdf = logPdfFnc(x,params)
            mu = params(1);
            sigma2 = params(2);
            d = length(x);
            log_pdf = -d/2*log(2*pi)-d/2*log(sigma2)-(x-mu)'*(x-mu)/sigma2/2;
        end

        %% Gradient of log pdf function
        % input: 
        %        x: Dx1 
        %        params = [mean(scalar),variance(scalar)]
        % output:
        %        gradient of log pdf function: Dx1
        function grad_log_pdf = GradlogPdfFnc(x,params)
            mu = params(1);
            sigma2 = params(2);
            grad_log_pdf = -(x-mu)/sigma2;
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
            grad_log_jac = 0;
        end
        
        %% Transform parameters to normal distribution
        % input: 
        %        x: Dx1 
        function params_normal = toNormalParams(x)
            params_normal = x;
        end

        %% Tranform normal parameters to original distribution
        % input: 
        %        x: Dx1 
        function params_ori = toOriginalParams(x)
            params_ori = x;
        end
        
        %% Plot density given distribution parameters
        % input: 
        %        x: Dx1 
        function plotPdf(params,varargin)
            
            % Extract 
            mu = params(1);
            sigma2 = params(2);
            
            xx = mu-4*sqrt(sigma2):0.001:mu+4*sqrt(sigma2);
            yy = normpdf(xx,mu,sqrt(sigma2));
            plot(xx,yy,'LineWidth',2)
        end
    end
end

