classdef Distribution
    %DISTRIBUTION A (Abtract) Superclass to define a probability distribution
    
    properties
    end
    
    methods (Abstract)
        random_num     = rngFnc(obj,params,dim);
        llh            = logPdfFnc(obj,data,params);
        llh_grad       = GradlogPdfFnc(obj,data,params);
        logjac         = logJacFnc(obj,params);
        logjac_grad    = GradlogJacFnc(obj,params);
    end
    
end
