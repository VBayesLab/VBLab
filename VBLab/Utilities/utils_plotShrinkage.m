function plotShrinkage(ShrinkageCoef,opt)
%PLOTSHRINKAGE Plot shrinkage coefficient of Group Lasso regularization
%
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%   
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

% Do not plot intercept coefficient
% ShrinkageCoef = ShrinkageCoef(2:end,:);

TextTitle = opt.title;
labelX = opt.labelX;
labelY = opt.labelY;
linewidth = opt.linewidth;
color = opt.color;

numCoeff = size(ShrinkageCoef,1);   % Number of shrinkage coefficients
fontsize = 13;

% Define default settings
if(isempty(TextTitle))
    TextTitle = 'Shrinakge Coefficients';
end
if(isempty(labelX))
    labelX = 'Iteration';
end

% Plot
plot(ShrinkageCoef','LineWidth',linewidth);
grid on
title(TextTitle,'FontSize', 20)
xlabel(labelX,'FontSize', 15)
ylabel(labelY,'FontSize', 15)
Ytext = ShrinkageCoef(:,end);  % Y coordination of text, different for coefficients
Xtext = size(ShrinkageCoef,2); % X coordination of text, same for all coefficients 
for i=1:numCoeff
    text(Xtext,Ytext(i),['\gamma_{',num2str(i),'}'],'fontsize',fontsize)
end
end

