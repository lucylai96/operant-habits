function [results, sched] = habitSimulate(params, sched)

%% initialize
maxiter = 20;
agent.alpha_w = params(1);
agent.alpha_t = params(2);
agent.alpha_r = params(3);  % only relevant for model 3 and 4
agent.alpha_b = 0;          % only relevant for model 4
sched.beta = params(4);     % only relevant for model 3 and 4
sched.acost = params(5);    % action cost
% sched.beta = 1;           % starting beta; high beta = low cost. beta should increase for high contingency
sched.cmax = 100;           % max complexity
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

% results.pi_as = mean(cat(3,results.pi_as),3);
% results.pi_as_std = std(cat(3,results.pi_as),[],3);
%p = results(1).pi_as(:,2)';
% x = 1:sched.timeSteps; Y = input(1,:);
% figure; hold on; subplot 211;plot(1:sched.timeSteps,p'); x = ones(1,sched.timeSteps); A = binornd(repmat(x,[100,1]),repmat(p,[100,1])); subplot 212; hold on;shadedErrorBar(1:sched.timeSteps,mean(movmean(A',200)'),std(movmean(A',200)'));plot(movmean(Y,200)); title('actions');prettyplot; % were rewards delivered at the same time?
%figure; hold on;plot(1:sched.timeSteps,p'); x = ones(1,sched.timeSteps); A = binornd(repmat(x,[100,1]),repmat(p,[100,1])); figure; hold on;subplot 311; hold on;shadedErrorBar(1:sched.timeSteps,mean(movmean(A',200)'),std(movmean(A',200)'));plot(movmean(Y,200));plot(movmean(Y,200)); title('actions');prettyplot; subplot 313;hold on; plot(results(1).x,'LineWidth',3);plot(input.data(2,:),'LineWidth',3); title('rewards');prettyplot;subplot 312;hold on; plot(results(1).a,'LineWidth',3);plot(input.data(1,:),'LineWidth',3); title('actions');prettyplot; % were rewards delivered at the same time?

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