function weights = utils_FNNInitialize(layers)
%NNINITIALIZE Summary of this function goes here
%  layers: vector of doubles, each number specifing the amount of
%  nodes in a layer of the network.
%
%  weights: cell array of weight matrices specifing the
%  translation from one layer of the network to the next.
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

weights = cell(1, length(layers)-1);

for i = 1:length(layers)-1
    % Using random weights from -b to b 
    b = sqrt(6)/(layers(i)+layers(i+1));
    if i==1
        weights{i} = rand(layers(i+1),layers(i))*2*b - b;  % Input layer already have bias
    else
        weights{i} = rand(layers(i+1),layers(i)+1)*2*b - b;  % 1 bias in input layer
    end
end

end

