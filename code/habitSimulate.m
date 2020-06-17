function results = habitSimulate(params, sched)

%% initialize
agent.alpha_w = params(1);
agent.alpha_t = params(2);
agent.alpha_b = params(1);  % only relevant if sched.model = 3 or 4
sched.acost = params(3);    % action cost
%sched.acost = 0.01;    % action cost
sched.beta = params(4);     % starting beta; high beta = low cost. beta should increase for high contingency
%sched.cmax = params(6);     % max complexity
sched.k = 1;

if sched.deval == 1
    sched.devalTime = sched.timeSteps;                % timestep where outcome gets devalued
    sched.timeSteps = sched.timeSteps+300;   % devaluation test period is 5 mins * 60 secs = 300 timesteps
else
    sched.devalTime = sched.timeSteps;                % timestep where outcome gets devalued
end

[O,T] =  habitSchedule(sched);
results = habitAgent(O,T, sched, agent);
end