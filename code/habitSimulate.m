function [results, sched] = habitSimulate(params, sched)

%% initialize
maxiter = 20;
agent.alpha_w = params(1);
agent.alpha_t = params(2);
agent.alpha_r = params(3);  % only relevant for model 3 and 4
agent.alpha_b = params(4);  % only relevant for model 3 and 4
sched.acost = params(5);    % action cost
sched.beta = 1;%;params(5);
sched.cmax = 100;
sched.k = 1;

if sched.deval == 1
    sched.devalTime = sched.timeSteps;                % timestep where outcome gets devalued
    sched.timeSteps = sched.timeSteps+300;   % devaluation test period is 5 mins * 60 secs = 300 timesteps
else
    sched.devalTime = sched.timeSteps;                % timestep where outcome gets devalued
end

for i = 1:maxiter
    results(i) = habitAgent(sched, agent);
end

%results.pi_as = mean(cat(3,results.pi_as),3);
%results.pi_as_std = std(cat(3,results.pi_as),[],3);
win = 100;
if sched.deval == 1
    for i  = 1:maxiter
        results(i).actRate = sum(results(i).a(1:sched.devalTime)==2)/sched.devalTime;
        results(i).outRate = sum(results(i).x(1:sched.devalTime)==2)/sched.devalTime;
        results(i).actRateTest = sum(results(i).a(sched.devalTime:sched.timeSteps)==2)/(sched.timeSteps-sched.devalTime);
        results(i).outRateTest = sum(results(i).x(sched.devalTime:sched.timeSteps)==2)/(sched.timeSteps-sched.devalTime);
        results(i).movOutRate = movmean(results(i).x-1, win,'Endpoints','shrink');
        results(i).movActRate = movmean(results(i).a-1, win,'Endpoints','shrink');
    end
else
    for i  = 1:maxiter
        results(i).actRate = sum(results(i).a(1:sched.devalTime-300)==2)/(sched.devalTime-300);
        results(i).outRate = sum(results(i).x(1:sched.devalTime-300)==2)/(sched.devalTime-300);
        results(i).actRateTest = sum(results(i).a(sched.devalTime-300:sched.timeSteps)==2)/(sched.timeSteps-300-sched.devalTime);
        results(i).outRateTest = sum(results(i).x(sched.devalTime-300:sched.timeSteps)==2)/(sched.timeSteps-300-sched.devalTime);
        results(i).movOutRate = movmean(results(i).x-1, win,'Endpoints','shrink');
        results(i).movActRate = movmean(results(i).a-1, win,'Endpoints','shrink');
    end
    
end
end