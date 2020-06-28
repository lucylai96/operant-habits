function [results, sched] = habitSimulate(params, sched)

%% initialize
agent.alpha_w = params(1);
agent.alpha_t = params(2);
agent.alpha_r = params(3);  % only relevant for model 3 and 4
sched.acost = params(4);    % action cost
sched.beta = params(5);   
sched.k = 1;

if sched.deval == 1
    sched.devalTime = sched.timeSteps;                % timestep where outcome gets devalued
    sched.timeSteps = sched.timeSteps+300;   % devaluation test period is 5 mins * 60 secs = 300 timesteps
else
    sched.devalTime = sched.timeSteps;                % timestep where outcome gets devalued
end

results = habitAgent(sched, agent);
end