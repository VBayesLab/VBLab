%genertate Sobol Sequence
function X1 = utils_gen_Sobol(m,s)
N = pow2(m); % Number of points;
cmax = 52; % number of digits of generated points

N  = pow2(m);                             % Number of points;
P  = sobolset(s);                         % Get Sobol sequence;
P  = scramble(P,'MatousekAffineOwen');    % Scramble Sobol points;
X1 = net(P,N);

X1 = X1';
