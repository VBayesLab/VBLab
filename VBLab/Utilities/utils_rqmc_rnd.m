function f = utils_rqmc_rnd(S,d)
% generate a matrix of RQMC of size S times d
max_sobol = 1111;
r = floor(d/max_sobol);
s = d-r*max_sobol;
if r>=1
    f = utils_gen_Sobol(ceil(log2(S)),max_sobol)'; 
    for i = 2:r
        f = [f,utils_gen_Sobol(ceil(log2(S)),max_sobol)']; 
    end
    f = [f,utils_gen_Sobol(ceil(log2(S)),s)']; 
else
    f = utils_gen_Sobol(ceil(log2(S)),d)'; 
end
    
end
