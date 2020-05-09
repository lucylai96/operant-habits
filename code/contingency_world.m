function [outcomes] = contingency_VIworld(sch, action, rate, time)
% to simulate the world, either a VR or a VI schedule
% sch = 0 indicates VR schedule
% sch = 1 indicates VI schedule

%% todo
% VI world emits outcomes irrespective of actions
% VR world emits outcomes based on actions 

%% VR world
if sch == 0
    


%% VI world
elseif sch == 1
    % outcomes are just a poisson process
    outcomes = poisson;
    p = 1/rate; % probability of reward at any given time
    binomrnd(time,p)
    
    
end


end