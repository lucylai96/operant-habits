function results = habitAgent(sched, agent, input)
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

%% initialization
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

plt = 0;
if plt == 1
    map = habitColors;
    figure; subplot 211; hold on;
    plot(t,Adt(:,1:20),'Color',map(1,:));% visualize features
    axis([0 20 0 1])
    xlabel('# actions since last reward')
    ylabel('feature level')
    subplot 212; hold on;
    plot(Tdt(:,1:20)','Color',map(1,:)); % visualize features
    axis([0 20 0 1])
    xlabel('timesteps since last reward')
    ylabel('feature level')
    
    subprettyplot(2,1)
end

% weights
phi0 = [Adt(:,1); Tdt(:,1)];   % features
phi = phi0;
theta  = zeros(length(phi),2); % policy weights
theta(1,:) = 1;                % bias to not do anything
w = zeros(length(phi),1);      % value weights
num = 1000;                    % marginal action probability over the last num seconds
rho = 0;                       % avg reward init
rho2 = 0;
ecost = 0;
ps = zeros(size(phi));

for s = 1:sched.sessnum
    % at start of every session reset these initial values
    phi0(:,1) = [zeros(length(Adt),1); Tdt(:,1)];   % features
    
    phi = phi0;
    x = 2;                                          % observation array init
    a = 1;                                          % action array init
    k = 1; na = 1; nt = 1; NA.rew(k) = na; NT.rew(k) = nt;
    p = [0.5 0.5];                                 % p(a) init
    
    k = 0; rt = 1; dI = 0; dV = 0; V = 0;
    sched.k = 1;
    
    if exist('input') == 1
        sched.timeSteps = input.timeSteps(s);
    end
    
    if sched.sessnum > 1
        sched.setrew = 1;
        sched.trainEnd = sched.timeSteps;
        sched.devalEnd = sched.trainEnd + sched.devalWin;
    else
        sched.setrew = 0;
    end
    
    if exist('input') == 1 && sched.deval == 0 && ismember(s,sched.devalsess) % if this is the "valued" condition
        sched.devalWin = 0;  sched.testWin = 300;
        sched.devalEnd = sched.trainEnd + sched.devalWin;
        timeSteps = sched.timeSteps + sched.devalWin + sched.testWin; % devaluation + extinction test period = 1 hr + 5 mins (3600 + 300 timeSteps)
        deval = zeros(1,timeSteps);                       % a vector indicating when devaluation manipulation "turns on"
        test = zeros(1,timeSteps);                        % a vector indicating when extinction test "turns on"
        
    else
        if ismember(s,sched.devalsess) && sched.deval == 1
            timeSteps = sched.timeSteps + sched.devalWin + sched.testWin; % devaluation + extinction test period = 1 hr + 5 mins (3600 + 300 timeSteps)
            deval = zeros(1,timeSteps);                       % a vector indicating when devaluation manipulation "turns on"
            deval(:,sched.trainEnd:sched.devalEnd-1) = 1;     % a vector indicating when devaluation manipulation "turns on"
            test = zeros(1,timeSteps);                        % a vector indicating when extinction test "turns on"
            test(:,sched.devalEnd:end) = 1;                   % 0 means test mode
            
        else
            timeSteps = sched.timeSteps;
            deval = zeros(1,sched.timeSteps);       % don't perform satiation
            test = zeros(1,sched.timeSteps);        % a vector indicating when devaluation "turns on"
        end
    end
    
    t = 2;
    
    while t <= timeSteps
        if sched.deval == 1 && (t == sched.devalEnd || t == sched.trainEnd) % reinit after test period begins
            rt = t;
        end
        
        %% define features
        na = sum(a(rt:t-1)==2)+1; % number of actions since last reward (add 1 to represent 1)
        nt = t-rt+1;              % timesteps since last reward (add 1 to represent 1)
        
        if nt>nS % a fix for conditions where the time / number of actions exceeds the number of features (rare)
            nt = nS;
        end
        if na>nS
            na = nS;
        end
        
        phi0 = phi;
        
        if deval(t) == 1  % if in devaluation mode
            phi0(:,1) = [zeros(length(Adt),1); zeros(length(Tdt),1)];
            phi = phi0;
        elseif test(t) == 1 % if in test mode
            phi0(:,1) = [Adt(:,na); Tdt(:,nt)]; % "satiated"
            phi = phi0;
        else
            phi(:,1) = [Adt(:,na); Tdt(:,nt)]; % features composed of indexed by action na and time t since last reward
        end
        if x(t-1)==2
            phi0 = phi;     % features
        end
        
        %% policy
        d = agent.beta*(theta'*phi)' + log(p);
        logpolicy = d - logsumexp(d);
        policy = exp(logpolicy);       % softmax policy
        a(t) = fastrandsample(policy);    % action (1) null (2) lever
        
        value = theta'*phi;
        habit = log(p);
        
        cost = logpolicy(a(t)) - log(p(a(t)));                        % policy complexity cost
        
        if exist('input') == 1 && t<=length(input.session(s).training.lever_binned) % if fitting to data, condition on data
            a(t) = input.session(s).training.lever_binned(t)+1;                     % 1 means nothing, 2 means lever press
            x(t) = input.session(s).training.reward_binned(t)+1;                    % 1 means nothing, 2 means reward
        elseif deval(t)==1 % satiation manipulation
            a(t) = 1;
            x(t) = 2;
        else
            if  exist('input') == 1 && (sched.deval == 0 || test(t)==1) % extinction for the valued case
                x(t) = 1;
            else
                % observe reward and new state
                [x(t), sched] = habitWorld(t, a, x, sched);
            end
        end
        
        % reward counter (clean this up)
        if x(t) == 2    % if you just saw reward
            k = k+1;    % counter for # rewards
            rt = t;     % log the timestep of last reward
            RT(k) = rt; % counter for reward times
        end
        
        %% action history
        if deval(t) == 0 && test(t) == 0
            p = p + agent.lrate_p*(policy - p); p = p./nansum(p);        % marginal update
        end
        
        %% policy complexity
        % should encompass all states and actions (entire policy)
        %pas = zeros(size(theta));
        %pas(1,:) = exp(theta(1,:).*phi(1,:));
        %pas(2:end,:) = exp(theta(2:end,:));
        %pas = pas./sum(pas,2);
        
        dd = agent.beta*(theta) + log(p);
        pas = exp(dd - logsumexp(dd,2));
        
        ps = ps + phi;
        normps = ps./sum(ps); % normalized state distribution
        
        mi = nansum(normps.*nansum(pas.*log(pas./p),2));  % mutual information
        
        if deval(t) == 0 && test(t) == 0 %&& x(t)==2% only update when not in test or deval mode
            ecost0 = ecost;
            ecost =  ecost + agent.lrate_e * (cost - ecost);
        end
        
        %% TD update
        r = ~test(t)*double(x(t)==2)-agent.acost*(a(t)==2);
        rpe = agent.beta*(r - rho) - cost + w'*(phi-phi0);      % reward prediction error
        
        if deval(t) == 0 && test(t) == 0 %&& k>2 && x(t) == 2 % only update when not in satiation mode or in test mode
            
            if ecost > agent.cmax % if I> C
                keyboard
            else
                %agent.beta = agent.beta + agent.lrate_b; % increase beta otherwise
            end
        end
        
        rho =  rho + agent.lrate_r*(r-rho);           % average reward update
        
        if test(t) == 0 % only update if not in test mode
            
            %% value update
            w = w + agent.lrate_w*rpe*phi;                       % weight update with value gradient
            
            %% policy update
            g = agent.beta*phi*(1 - policy(a(t)));      % policy gradient
            theta(:,a(t)) = theta(:,a(t)) + agent.lrate_theta*rpe*g/t;      % policy weight update
            
        end
        
        %% store results
        if exist('input') ~= 1 % if simulation mode
            results(s).a(t) = a(t);         % action
            results(s).x(t) = x(t);         % observation
            results(s).r(t) = r;            % instantaneous reward
            results(s).rho(t) = rho;        % expected reward (avg reward)
            results(s).rpe(t) = rpe;      % RPE
            
            results(s).pi_as(t,:) = policy;  % chosen policy
            %results(s).pas(:,:,t) = pas;     % entire state-feature policy
            results(s).val(t,:) = value;
            results(s).hab(t,:) = habit;
            
            results(s).w(t,:) = w;            % state value weights
            results(s).theta(:,:,t) = theta;  % policy weights
            results(s).ecost(t) = ecost;       % expected cost (avg cost)
            results(s).pa(t,:) = p;            % marginal action probability over 5 trials
            results(s).beta(t) = agent.beta;
            results(s).mi(t) = mi;
            results(s).cost(t) = cost;          % policy cost for the action taken C(p(a|s))
        else % if fitting mode
            results(s).a(t) = a(t);         % action
            results(s).x(t) = x(t);         % observation
            results(s).pi_as(t,:) = policy;  % chosen policy
            results(s).ecost(t) = ecost;      % expected cost (avg cost)
            results(s).beta(t) = beta;
        end
        
        t = t+1;
        
        % limiting at 50 rewards (rarely used)
        if exist('input') == 0
            if (isequal(sched.type,'FR') || isequal(sched.type,'VR')) && sum(x==2)==50 && sched.setrew == 1
                sched.setrew = 0;                       % only do this once, make sure "setrew" is on in the big loop
                timeSteps = t-1;
                if ismember(s,sched.devalsess)          % if sess ends early, start devaluation
                    sched.timeSteps = t;
                    sched.trainEnd = sched.timeSteps;
                    sched.devalEnd = sched.trainEnd + sched.devalWin;
                    
                    timeSteps = t + sched.devalWin + sched.testWin;   % devaluation + extinction test period = 1 hr + 5 mins (3600 + 300 timeSteps)
                    deval = zeros(1,timeSteps);                       % a vector indicating when devaluation manipulation "turns on"
                    deval(:,sched.trainEnd:sched.devalEnd-1) = 1;     % a vector indicating when devaluation manipulation "turns on"
                    test = zeros(1,timeSteps);                        % a vector indicating when extinction test "turns on"
                    test(:,sched.devalEnd:end) = 1;                   % 0 means test mode
                else
                    test = zeros(1,timeSteps);          % a vector indicating when devaluation "turns on"
                end
                
            end % if you've reached 50 rewards
        end
        
    end % while t<=timeSteps loop
    
    
end % for each session

end % habitAgent


