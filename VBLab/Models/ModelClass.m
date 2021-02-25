classdef ModelClass
    %MODELCLASS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        ModelName
        NumParam
    end
    
    methods
        function obj = ModelClass(inputArg1,inputArg2)
            %MODELCLASS Construct an instance of this class
            %   Detailed explanation goes here
            obj.Property1 = inputArg1 + inputArg2;
        end
    end
    
    methods (Abstract)
        llh            = logLikFnc(obj,data,params);
        llh_grad       = logLikGradFnc(obj,data,params);
        log_prior      = logPriorsFnc(obj,params);
        log_prior_grad = logPriorsGradFnc(obj,params);
        logjac         = logJacFnc(obj,params);
        logjac_grad    = logJacGradFnc(obj,params);
    end

end

