function results = habitSimulate(params,input)
% INPUT: params - model parameters
%        input - data
%
% use the data to fix:
% (1) total timesteps
% (2) the reward times / number of actions until reward
% (3) the lever-press and reward times
% each session separately (bc diff rats)

sched = input.sched;

%% initialize
agent.alpha_w = params(1);
agent.alpha_t = params(2);
agent.alpha_b = 1;      % only relevant for model 3 and 4
agent.alpha_r = 0.001;
agent.alpha_e = 0.001;


if sched.model == 4
    sched.beta = 0.1;             % only relevant for model 3 and 4
    agent.acost = params(4);    % action cost
elseif sched.model == 3
    sched.beta = params(3);     % only relevant for model 3 and 4
    agent.acost = params(4);    % action cost
elseif sched.model == 2
    sched.beta = 1;     % only relevant for model 3 and 4
    agent.acost = params(3);    % action cost
end


sched.cmax = 100;           % max complexity
sched.k = 1;
sched.sessnum = 20;
sched.devalsess = [2 10 20];
sched.testWin = 300;
results = habitAgentFit(sched, agent, input.data);

win = 100;
for s  = 1:sched.sessnum
    results(s).actRate = sum(results(s).a(1:results(s).sched.trainEnd)==2)/results(s).sched.trainEnd;
    results(s).outRate = sum(results(s).x(1:results(s).sched.trainEnd)==2)/results(s).sched.trainEnd;
    results(s).avgBeta = mean(results(s).beta(1:results(s).sched.trainEnd));
    results(s).movOutRate = movmean(results(s).x-1, win,'Endpoints','shrink');
    results(s).movActRate = movmean(results(s).a-1, win,'Endpoints','shrink');
    
    if ismember(s,sched.devalsess)
        results(s).postTestRate = sum(results(s).a(results(s).sched.devalEnd:end)==2)/sched.testWin;
    end
end

end