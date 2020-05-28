function [waitTimes] = contingency_generateVI(int)
%% to simulate the world, VI schedule
if nargin == 0; int = 10; end % 7200 seconds is 2 hours

% outcomes are just a poisson process
p = 1/int; % probability of reward at any given time
I = binornd(1,p*ones(1,50000));

intTimes = find(I==1); % interval times 
waitTimes = diff(intTimes); % wait times between intervals
end