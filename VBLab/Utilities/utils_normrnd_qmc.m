function x = utils_normrnd_qmc(S,d)
% generate Sxd matrix of standard normal numbers by RQMC
rqmc = utils_rqmc_rnd(S,d);  
rqmc = rqmc(1:S,:);
x = norminv(rqmc); 
end