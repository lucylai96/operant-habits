function nLL = habitAgentFit(params, input)
% PURPOSE: Reward-complexity in free-operant learning
%
% INPUTS:
%   sched - details about the operant schedule
%   agent (optional) - learning parameters, if simulating from params
%   input (optional) - some data used to seed (if in fitting mode)
%
% OUTPUTS:
%   results - structure
%
% NOTES:
%
% Written by Lucy Lai (May 2020)

% INPUT: params - model parameters
%        input - data
%
% use the data to fix:
% (1) total timesteps
% (2) the reward times / number of actions until reward
% (3) the lever-press and reward times
% each session separately (bc diff rats)

% unpack
sched = input.sched;
timeSteps = input.timeSteps;
a = input.a + 1; a(1) = 2;  % conditioning on data (actions)
x = input.x + 1;  % conditioning on data (rewards)

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

k = 1; 
t = 1;
lik = 0;

Ps = zeros(size(phi));
Q = zeros(length(phi),2);
rt = [1 find(input.x==1)];% reward times;
while t <= timeSteps(end)
    
    %% define features
    na = sum(a(rt(k):t)==2)+1; % number of actions since last reward 
    nt = t-rt(k)+1;           % timesteps since last reward

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
    lik = lik + logpolicy(a(t));
    
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
    theta(:,a(t)) = theta(:,a(t)) + agent.lrate_theta*rpe*g/t;      % policy weight update
    
    if x(t) == 2    % if you just saw reward
        k = k+1;
    end
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

results.theta = theta;                         % policy weights
results.theta = theta./nansum(theta(:));       % policy weights

results.Q = Q./nansum(Q(:));                   % policy weights
results.Ps = Ps./sum(Ps);                      % policy weights
results.Q(isnan(results.Q(:))) = 0;
results.Q(1:100,:) = 0;
results.Q(20,2) = 1;

% bernoulli log liklihood for each second
% sum up nLL for each decisecond bin

% Y is the action obs [1 or 0]
% n is number of obs (1)
% p(i) is probability of action (p_as) (changing from timestep to timestep)

% p = results.pi(:,2)';
% p(p<0.01) = 0.01;
% p(p>0.99) = 0.99;
% Y = input.data.session(s).training.lever_binned;       % action sequence (binned in seconds)
% n = ones(size(Y));
% nLL(s) = sum(Y.*log(p) + (n-Y).*log(1-p));

%nLL = -sum(nLL);
nLL = -lik;
params
end % habitAgent

function plt
[R,V] = blahut_arimoto(normps',results.theta,logspace(log10(0.1),log10(50),50));
figure; hold on; 
plot(R,V,'o-');

figure; hold on; 
plot(results.mi,results.avgr,'ko'); hold on;
plot(results.mi(end),results.avgr(end),'ro','MarkerSize',20)
xlabel('Policy complexity')
ylabel('Average reward')

plot(results.mi,results.rho,'ko'); hold on;
plot(results.mi(end),results.rho(end),'mo','MarkerSize',20)


[R,V] = blahut_arimoto(normps',theta,logspace(log10(0.1),log10(50),50));
figure; hold on; 
plot(R,V,'o-');
xlabel('Policy complexity')
ylabel('Average reward')
plot(results.mi,results.avgr,'ko')


[R,V] = blahut_arimoto(results.Ps',results.Q,logspace(log10(0.1),log10(50),50));
figure; hold on; 
plot(R,V,'o-');
xlabel('Policy complexity')
ylabel('Average reward')


[R,V] = blahut_arimoto(normps',results.Q,logspace(log10(0.1),log10(50),50));
figure; hold on; 
plot(R,V,'o-');
xlabel('Policy complexity')
ylabel('Average reward')

%R_data(b) = results.mi(t);
%V_data(b) = results.rho;
end

