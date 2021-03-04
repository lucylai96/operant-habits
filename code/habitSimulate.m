function results = habitSimulate(params,input)

% unpack
sched = input.sched;
timeSteps = input.timeSteps;

%% initialization

% set learning rates
agent.lrate_w = params(1);
agent.lrate_theta = params(2);
agent.lrate_p = params(3);
agent.beta = params(4);

% fixed
agent.acost = 0.01;    % action cost
agent.lrate_r = 0.001;  % avg reward
agent.lrate_e = 0.001;  % expected cost
agent.cmax = 1;

sched.k = 1;
sched.deval = 0;
sched.testWin = 0;

% states
if (isequal(sched.type,'FR') || isequal(sched.type,'VR'))
    nS = sched.R*20; % number of features
end

if (isequal(sched.type,'FI') || isequal(sched.type,'VI'))
    nS = sched.I*20; % number of features
end

nS = 100;
d = 1:nS;       % number of features
mu_d = 1:nS;    % mean of features
wa = 0.1;       % weber fraction for actions
wt = 0.1;       % weber fraction for time

t = 1:nS;       % time or actions
for i = 1:length(d)
    Adt(i,:) = exp((-1/2)*((t-mu_d(i))/(wa*mu_d(i))).^2);
    Tdt(i,:) = exp((-1/2)*((t-mu_d(i))/(wt*mu_d(i))).^2);
    
    Adt2(i,:)=Adt(i,:)./sum(Adt(i,:));
    Tdt2(i,:)=Tdt(i,:)./sum(Tdt(i,:));
end
Adt = Adt2';
Tdt = Tdt2';

% weights
phi0 = [Adt(:,1); Tdt(:,1)];   % features
phi = phi0;
theta  = zeros(length(phi),2); % policy weights
theta(1,:) = 1;                % bias to not do anything
w = zeros(length(phi),1);      % value weights
rho = 0;                       % avg reward init
ecost = 0;
ps = zeros(size(phi));
p = [0.8 0.2];                                 % p(a) init

sched.k = 1;
sched.setrew = 0;
sched.devalWin = 0;  sched.testWin = 300;
%sched.devalEnd = sched.trainEnd + sched.devalWin;
%timeSteps = sched.timeSteps + sched.devalWin + sched.testWin; % devaluation + extinction test period = 1 hr + 5 mins (3600 + 300 timeSteps)
%deval = zeros(1,timeSteps);                       % a vector indicating when devaluation manipulation "turns on"
%test = zeros(1,timeSteps);                        % a vector indicating when extinction test "turns on"

a = 2;
x = 2;
k = 1;
t = 2;
rt = 1;
lik = 0;

Ps = zeros(size(phi));
Q = zeros(length(phi),2);
while t <= timeSteps(end)
    
    %% define features
    na = sum(a(rt:t-1)==2)+1; % number of actions since last reward
    nt = t-rt+1;           % timesteps since last reward
    
    if nt>nS % a fix for conditions where the time / number of actions exceeds the number of features (rare)
        nt = nS;
    end
    
    if na>nS
        na = nS;
    end
    
    phi0 = phi;
    phi(:,1) = [Adt(:,na); Tdt(:,nt)]; % features composed of indexed by action na and time t since last reward
    
    %% policy
    d = agent.beta*(theta'*phi)' + log(p);
    logpolicy = d - logsumexp(d);
    policy = exp(logpolicy);          % softmax policy
    a(t) = fastrandsample(policy);    % action (1) null (2) lever
     
    % observe reward and new state
    [x(t), sched] = habitWorld(t, a, x, sched);
    if x(t) == 2
        rt = t;
    end
    
    value = theta'*phi;
    habit = log(p);
    
    cost = logpolicy(a(t)) - log(p(a(t)));                  % policy complexity cost
    p = p + agent.lrate_p*(policy - p); p = p./nansum(p);   % marginal policy update
    
    %% policy complexity
    dd = agent.beta*(theta) + log(p);
    pas = exp(dd - logsumexp(dd,2));
    
    ps = ps + phi;
    normps = ps./sum(ps); % normalized state distribution
    
    mi = nansum(normps.*nansum(pas.*log(pas./p),2));  % mutual information
    ecost =  ecost + agent.lrate_e * (cost - ecost);
    
    %%  learning updates
    r = double(x(t)==2)-agent.acost*(a(t)==2);
    rpe = agent.beta*(r - rho) - cost + w'*(phi-phi0);      % reward prediction error
    rho =  rho + agent.lrate_r*(r-rho);           % average reward update
    w = w + agent.lrate_w*rpe*phi;                       % weight update with value gradient
    g = agent.beta*phi*(1 - policy(a(t)));      % policy gradient
    theta(:,a(t)) = theta(:,a(t)) + agent.lrate_theta*rpe*g;      % policy weight update
   
    %% store results
    results.a(t) = a(t);         % action
    results.x(t) = x(t);         % observation
    results.r(t) = r;            % expected reward (avg reward)
    results.avgr(t) = mean(results.r);
    results.rho(t) = rho;        % expected reward (avg reward)
    results.pi(t,:) = policy;    % chosen policy
    results.p(t,:) = p;          % marginal action probability over 5 trials
    results.ecost(t) = ecost;    % expected cost (avg cost)
    results.mi(t) = mi;
    results.na(t) = na;
    results.nt(t) = nt;
    t = t+1;
    
end % while t<=timeSteps loop

results.normps = normps; 
results.theta = theta;                         % policy weights
results.theta = theta./nansum(theta(:));       % policy weights

end

function habitSimulateOld(params,input)
% INPUT: params - model parameters
%        input - data
%
% use the data to fix:
% (1) total timesteps
% (2) the reward times / number of actions until reward
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