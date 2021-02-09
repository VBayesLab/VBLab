function output = utils_sigmoid(z)
%UTILS_SIGMOID Summary of this function goes here
%   Detailed explanation goes here
output = 1.0 ./ (1.0 + exp(-z));
end

