function msg_out = utils_errorMsg(identifier)
%UTILS_ERRORMSG Define custom error/warning messages for exceptions
%   UTILS_ERRORMSG = (IDENTIFIER) extract message for input indentifier
%   
%
%   Copyright 2021 Nguyen (nghia.nguyen@sydney.edu.au)
%
%   https://github.com/VBayesLab/VBLab
%
%   Version: 1.0
%   LAST UPDATE: Feb, 2021

switch identifier
    case 'vbayeslab:TooFewInputs'
        msg_out = 'At least two arguments are specified';
    case 'vbayeslab:InputSizeMismatchX'
        msg_out = 'X and Y must have the same number of observations';
    case 'vbayeslab:InputSizeMismatchY'
        msg_out = 'Y must be a single column vector';
    case 'vbayeslab:ArgumentMustBePair'
        msg_out = 'Optinal arguments must be pairs';
    case 'vbayeslab:ResponseMustBeBinary'
        msg_out = 'Two level categorical variable required';
    case 'vbayeslab:DistributionMustBeBinomial'
        msg_out = 'Binomial distribution option required';
    case 'vbayeslab:MustSpecifyActivationFunction'
        msg_out = 'Activation function type requied';
    case 'vbayeslab:InitVectorMisMatched'
        msg_out = 'The length of the initial values must equal to number of model parameters';
end
end

