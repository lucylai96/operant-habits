function [nLL,results] = habitAgentFit(params, input)
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
a = input.a + 1; a(1) = 2;     % conditioning on data (actions)
x = input.x + 1;               % conditioning on data (rewards)

%% initialization
p = [0.8 0.2];          % p(a) init

% set learning rates
agent.lrate_w = params(1);
agent.lrate_theta = params(2);
agent.lrate_p = params(3);
if params(4) < 0.01
    agent.lrate_b = params(4);
    agent.beta =  0.01;
else
    agent.beta =  params(4);
    agent.lrate_b = 0;
end

% fixed
agent.acost = 0.01;     % action cost
agent.lrate_r = 0.001;  % avg reward
agent.lrate_e = 0.001;  % exp cost
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
ecost = 0;
rho = 0;                       % avg reward init
ps = zeros(size(phi));

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
  
    
    %%  learning updates
    r = double(x(t)==2)-agent.acost*(a(t)==2);
    Q(:,a(t)) = Q(:,a(t)) + agent.lrate_theta*(phi.*(agent.beta*r-cost)-Q(:,a(t)));
    rpe = agent.beta*(r - rho) - cost + w'*(phi-phi0);        % reward prediction error
    rho =  rho + agent.lrate_r*((agent.beta*r-cost)-rho);     % average reward update
    ecost = ecost + agent.lrate_e*(cost-ecost);               % expected policy cost update
    w = w + agent.lrate_w*rpe*phi;                            % weight update with value gradient
    g = agent.beta*phi*(1 - policy(a(t)));                    % policy gradient
    theta(:,a(t)) = theta(:,a(t)) + agent.lrate_theta*rpe*g;  % policy weight update
    
    agent.beta = agent.beta + agent.lrate_b;

    if x(t) == 2    % if you just saw reward
        k = k+1;
    end
    %% store results
    results.a(t) = a(t);         % action
    results.x(t) = x(t);         % observation
    results.r(t) = r;            % expected reward (avg reward)
    results.avgr(t) = mean(results.r);
    results.rho(t) = rho;        % expected reward (avg reward)
    %results.pi(t,:) = policy;    % chosen policy
    results.p(t,:) = p;          % marginal action probability over 5 trials
    %results.mi(t) = mi;
    results.ecost(t) = ecost;
    results.cost(t) = cost;
    %esults.na(t) = na;
    %results.nt(t) = nt;
    t = t+1;
    
end % while t<=timeSteps loop

results.timeSteps = timeSteps;
nLL = -lik;
params;
end % habitAgent

function plt
beta = linspace(0.1,15,50);
[R,V] = blahut_arimoto(results.normps',results.Q,beta);
figure; hold on; 
plot(R,V,'r-');
plot(results.mi,results.avgr,'ko'); hold on;
plot(results.mi(end),results.avgr(end),'ro','MarkerSize',20)
xlabel('Policy complexity')
ylabel('Average reward')

beta = linspace(0.1,15,50);
[R,V] = blahut_arimoto(results.normps',results.theta./sum(results.theta,2),beta);
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


