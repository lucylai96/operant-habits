function [numActions] = contingency_generateVR(ratio)
%% to simulate the world, VI schedule
if nargin == 0; ratio = 10; end % 7200 seconds is 2 hours

% number of actions required before reward
% lambda_o is actually dictated by a poisson process 
% lambda_o = 1/ratio * lambda_a; % see if this is empirically true
p = 1/ratio; 
A = binornd(1,p*ones(1,10000)); 
numActions = diff(find(A==1)); 

end

