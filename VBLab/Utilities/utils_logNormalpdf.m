function logNormal = utils_logNormalpdf(theta,mu,sigma2)

logNormal = -0.5*log(2*pi)-0.5*log(sigma2)-0.5*(theta-mu).^2/sigma2;

end

