function [theta] = utils_update_sigma(sigma2,acc,p,i,d)
T = 200;
alpha = -norminv(p/2);
c = ((1-1/d)*sqrt(2*pi)*exp(alpha^2/2)/(2*alpha) + 1/(d*p*(1-p)));
Theta = log(sqrt(abs(sigma2)));
Theta = Theta + c*(acc-p)/max(T, i/d);
theta = (exp(Theta));
theta = theta^2;
