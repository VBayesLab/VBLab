function [B_var] = utils_jitChol(B_var)
%keyboard
[R,p]=chol(B_var);
if p>0
    min_eig=min(eig(B_var));
    d=size(B_var,1);
    delta=max(0,-2*min_eig+10^(-5)).*eye(d);
    B_var=B_var+delta;
else
    B_var=B_var;
end









end
% if nargin < 2
%   maxTries = 1000;
% end
% n=size(K,1); % no. of input samples
% e=min(eig(K)); % minimum eigenvalue
% jitter=0; % amount if jitter (noise) added to the diagonal
% L=[];
% 
% for i=1:maxTries
%     try
%         L=chol(K,'lower');
%     catch
%         K(1:(n+1):end)=K(1:(n+1):end)+e;
%         jitter=jitter+e;
%         e=e*10;
%         continue;
%     end
%     break;
% end
% 
% K1 = K;

% if isempty(L) %if nothing was assigned in previous step,
%     K(1:(n+1):end)=K(1:(n+1):end)-jitter;
%     e=1e-10; jitter=0; %reset parameters
%     l=max(diag(K)); K=K/l;
%     for i=1:maxTries
%         try
%             L=chol(K,'lower');
%         catch
%             K(1:(n+1):end)=K(1:(n+1):end)+e;
%             jitter=jitter+e;
%             e=e*10;
%             continue;
%         end
%         L=sqrt(l)*L;
%         break;
%     end
% end
        