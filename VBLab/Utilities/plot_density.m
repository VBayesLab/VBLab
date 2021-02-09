function plot_density(dist,mu,sigma2)

switch dist 
    case 'normal'
        xx = mu-4*sqrt(sigma2):0.001:mu+4*sqrt(sigma2);
        yy = normpdf(xx,mu,sqrt(sigma2));
        plot(xx,yy,'LineWidth',2)
end
end

