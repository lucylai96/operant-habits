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
beta = sched.beta;                      % beta

% learning rates
if nargin <2 % if simulating without input parameters
    alpha_w = 0.1;          % value learning rate
    alpha_t = 0.1;          % policy learning rate
    alpha_r = 0.1;          % rho learning rate
    alpha_b = 0.1;          % beta learning rate
    acost = 0.1;            % action cost
else % if simulating with input parameters
    alpha_w = agent.alpha_w;          % value learning rate
    alpha_t = agent.alpha_t;          % policy learning rate
    alpha_r = agent.alpha_r;          % rho learning rate
    alpha_b = agent.alpha_b;          % beta learning rate
    acost = agent.acost;              % action cost
end
if (isequal(sched.type,'FR') || isequal(sched.type,'VR'))
    nS = sched.R*20; % number of features
end
if (isequal(sched.type,'FI') || isequal(sched.type,'VI'))
    nS = sched.I*20; % number of features
end

nS = 200;
d = 1:nS;       % number of features
mu_d = 1:nS;    % mean of features
wa = 0.1;       % weber fraction for actions
wt = 0.1;      % weber fraction for time
t = 1:nS;       % time or actions
% c = linspace(0,1,nS);
% mu_c = linspace(0,1,nS);
% wc = 0.01;
% Cdt = zeros(nS, size(linspace(0,1,nS),2)); % d dimensions
for i = 1:length(d)
    Adt(i,:) = exp((-1/2)*((t-mu_d(i))/(wa*mu_d(i))).^2);
    Tdt(i,:) = exp((-1/2)*((t-mu_d(i))/(wt*mu_d(i))).^2);
    % Cdt(i,:) = exp((-1/2)*((c-mu_c(i))/(wc)).^2);
end


% figure; subplot 311; hold on;
% plot(Adt'); % visualize features
% subplot 312; hold on;
% plot(Tdt'); % visualize features
% subplot 313; hold on;
% plot(Cdt'); % visualize features


% weights
phi = [Adt(:,1); Tdt(:,1)];     % features
theta  = zeros(length(phi),2);  % policy weights
theta(:,1) = 0.4;              % bias to not do anything
theta(:,2) = -0.4;              % bias to not do anything
w = zeros(length(phi),1);       % value weights
num = 10;                       % marginal action probability over the last num seconds

for s = 1:sched.sessnum
    % at start of every session reset these initial values
    phi = [Adt(:,1); Tdt(:,1)]; % features
    ps = zeros(length(phi),1);
    x = [2];         % observation array init
    a = [2];         % action array init
    k = 1; na = 1; nt = 1; NA.rew(k) = na; NT.rew(k) = nt;
    rho = 0;         % avg reward init
    ecost = 0;
    paa = [0.5 0.5]; % p(a) init
    pa = [0.5 0.5]; % p(a) init
    k = 0; rt = 1; dI = 0; dV = 0; V = 0;
    sched.k = 1;
    %sched.setrew = 1;
    deval = ones(1,sched.timeSteps);        % a vector indicating when devaluation "turns on"
    deval(:,sched.devalTime:end) = 0;       % 0 means devaluation mode
    
    if ismember(s,sched.devalsess) && sched.deval == 1
        timeSteps = sched.timeSteps+300;  % devaluation test period = 5 mins (300 timeSteps)
        deval = ones(1,timeSteps);        % a vector indicating when devaluation "turns on"
        deval(:,sched.devalTime:end) = 0;       % 0 means devaluation mode
    else
        timeSteps = sched.timeSteps;
    end
    t = 2;
    while t <= timeSteps
        %% define features
        na = sum(a(rt:t-1)==2); % number of actions since last reward
        nt = t-rt;              % timesteps since last reward
        
        if nt>nS % a fix for conditions where the time / number of actions exceeds the number of features (rare)
            nt = nS;
        end
        if na>nS
            na = nS;
        end
        
        if x(t-1)==2 % if you just saw reward, reinit state features (end of "episode")
            phi0 = [Adt(:,1); Tdt(:,1)]; % features
        else
            phi0 = phi;
        end
        phi = [Adt(:,na); Tdt(:,nt)]; % features composed of indexed by action na and time t since last reward
        phi(isnan(phi)) = 0;
        
        %% action history
        if t>num
            pa(1) = sum(a(t-num:t-1)==1)/num; pa(2) = sum(a(t-num:t-1)==2)/num; % trailing marginal action distribution
        else
            pa(1) = sum(a==1)/t; pa(2) = sum(a==2)/t;
        end
        pa = pa+0.01; pa = pa./sum(pa);
        
        %% action selection
        act = theta'*phi+log(paa)';  % action probabilities
        pi_as = [1/(1+exp(-(act(1)-act(2)))) 1-(1/(1+exp(-(act(1)-act(2)))))];
        %pi_as = exp((theta'*phi)' + log(pa));
        
        %pi_as(pi_as>100) = 100;
        %pi_as(pi_as<-100) = -100;
        %if sum(pi_as) == 0
        %    pi_as = [0.5 0.5];
        %end
        %pi_as = pi_as./sum(pi_as);
        pi_as = pi_as+0.01; pi_as = pi_as./sum(pi_as);
        
        
        if exist('input') == 1 && t<size(input(:,:,s),2)  % if fitting to data, condition on data
            a(t) = input(1,t,s)+1; % 1 means nothing, 2 means lever press
            x(t) = input(2,t,s)+1; % 1 means nothing, 2 means reward
        else
            % policy sampled stochastically
            if rand < pi_as(1)
                a(t) = 1;      % null
            else
                a(t) = 2;      % press lever
            end
            
            % observe reward and new state
            [x(t), sched] = habitWorld(t, a, x, sched);
            
        end
        
        % reward counter
        if x(t) == 2   % if you just saw reward
            k = k+1;    % counter for # rewards
            rt = t;     % log the timestep of last reward
            RT(k) = rt; % counter for reward times
        end
        
        %% policy complexity
        if k > 1 % after 2 rewards, start using marginal, or else too much initial bias
            paa(1) = sum(a==1)/t; paa(2) = sum(a==2)/t;                    % total marginal action distribution
            paa = paa+0.01; paa = paa./sum(paa);
        end
        
        if t>2
            ps = (sum(results(s).ps,2)./sum(results(s).ps(:)));  % probability of being in a state
        end
        
        % should encompass all states and actions (entire policy)
        pas = exp(theta+log(paa));                                     % entire state dep policy
        pas = pas./sum(pas,2);
        
        mi = nansum(ps.*nansum(pas.*log(pas./paa),2));                 % mutual information
        
        cost0 = log(pi_as./paa);
        cost = cost0(a(t));                                            % policy cost for that state-action pair
        
        %% TD update
        switch sched.model % different models with different costs
            case 1 % no action cost, no complexity cost
                r = deval(t)*double(x(t)==2);                % reward (deval is a flag for whether we are in deval mode)
                rho =  rho + alpha_r *(r - rho);             % average reward update
                rpe = r - rho + w'*(phi-phi0);               % TD error
            case 2 % yes action cost, no complexity cost
                r = deval(t)*double(x(t)==2)-acost*(a(t)==2);
                rho =  rho + alpha_r *(r - rho);
                rpe = r - rho + w'*(phi-phi0);
            case 3  % yes action cost, yes complexity cost (fixed)
                r = deval(t)*double(x(t)==2)-acost*(a(t)==2);
                rho =  rho + alpha_r *(r - rho);
                rpe = r - rho + w'*(phi-phi0)-(1/beta)*cost;
            case 4  % yes action cost, yes complexity cost (adaptive)
                r = deval(t)*double(x(t)==2)-acost*(a(t)==2);
                ecost0 = ecost;
                ecost =  ecost + alpha_r * (cost - ecost);
                rho0 = rho;
                rho =  rho + alpha_r * (r - rho);
                
                %if k > 2 && x(t) == 2 && deval(t)==1
                % dI = mean(results(sess).mi(t-num)) - mean(results(sess).mi(t-2*num:t-num)); % change in policy complexity
                %dI = mean(results(s).ecost(RT(k-1):RT(k)-1)) - mean(results(s).ecost(RT(k-2):RT(k-1)));
                
                % every trial, but doesn't work
                % dI = ecost-ecost0;
                % beta = beta + alpha_b*(dI/rho);
                
                % only at reward points -- fluctuates a lot but correct trend
                %dI =
                %V = mean([results(s).r(RT(k-1):t-1) r]);
                %beta = beta + alpha_b*(dI/V);
                
                % only at reward points
                dI = ecost - ecost0;
                %dI = mean([results.cost(RT(k-1)+1:t-1) cost]) - mean(results.cost(RT(k-2)+1:RT(k-1))); % diff in cost over last two reward intervals
                %dV = mean([results.r(RT(k-1)+1:t-1) r]) - mean(results.r(RT(k-2)+1:RT(k-1))); % diff in cost over last two reward intervals
                %dI = ecost - results.ecost(RT(k-1));
                %V = mean([results(s).r(RT(k-1)+1:t-1) r]); % vs rho
                %dV = results(s).rho-
                beta = beta + alpha_b*dI; % problem bc rho is almost always 0, absorb it into the learning rate
                
                % every timestep?
                %dI = ecost-ecost0;
                %dV = rho-rho0;
                %beta = beta + alpha_b*(beta - dI/dV);
                
                % bounds on beta
                if beta < 0
                    %disp('!')
                    beta = 0.1;
                elseif beta > sched.cmax
                    %disp('!')
                    beta = sched.cmax;
                end
                %end
                
                
                rpe = r - rho + w'*(phi-phi0)-(1/beta)*cost;
        end
        
        if x(t) == 2
            RPE.rew(t) = rpe;
            RPE.nonrew(t) = NaN;
            NA.rew(t) = na;
            NT.rew(t) = nt;
            NA.nonrew(t) = NaN;
            NT.nonrew(t) = NaN;
            
            
        else
            RPE.rew(t) = NaN;
            RPE.nonrew(t) = rpe;
            NA.rew(t) = NaN;
            NT.rew(t) = NaN;
            NA.nonrew(t) = na;
            NT.nonrew(t) = nt;
        end
        VS(t) = w'*(phi-phi0);
        
        if isnan(rpe) || isinf(rpe)
            keyboard
        end
        
        %% value update
        w0 = w;
        w = w + alpha_w*rpe*phi0;                                 % weight update with value gradient
        
        if sum(isnan(w))>0
            keyboard
        end
        
        %% policy update
        theta0 = theta;
        theta(:,a(t)) = theta(:,a(t)) + alpha_t*rpe*phi0;         % policy weight update
        
        %% store results
        results(s).a(t) = a(t);         % action
        results(s).x(t) = x(t);         % observation
        results(s).r(t) = r;            % instantaneous reward
        results(s).rho(t) = rho;        % expected reward (avg reward)
        results(s).rpe(t,:) = rpe;      % RPE
        results(s).pi_as(t,:) = pi_as;  % chosen policy
        
        results(s).w(t,:) = w0;            % state value weights
        results(s).theta(:,:,t) = theta0;  % policy weights
        
        results(s).cost(t) = cost;       % policy cost for the action taken C(p(a|s))
        results(s).ecost(t) = ecost;     % expected cost (avg cost)
        results(s).ps(:,t) = phi;        % state feature visits
        results(s).pa(t,:) = pa;         % marginal action probability over 5 trials
        results(s).paa(t,:) = paa;       % total marginal action probability (entire history)
        results(s).mi(t) = mi;
        results(s).beta(t) = beta;
        results(s).dI(t) = dI;
        %results(s).dV(t) = dV;
        %results(s).V(t) = V;
        
        t = t+1;
        if (isequal(sched.type,'FR') || isequal(sched.type,'VR')) && sum(x==2)==50 && sched.setrew == 1
            sched.setrew = 0; % only do this once, make sure "setrew" is on in the big loop
            timeSteps = t-1;
            if ismember(s,sched.devalsess)          % if sess ends early, start devaluation
                timeSteps = timeSteps+300;          % devaluation test period = 5 mins (300 timeSteps)
                deval = ones(1,timeSteps);          % a vector indicating when devaluation "turns on"
                deval(:,t:end) = 0;                 % 0 means devaluation mode
            else
                deval = ones(1,timeSteps);          % a vector indicating when devaluation "turns on"
            end
            
        end
        
        
    end
    results(s).deval = deval;
    results(s).NA = NA;
    results(s).NT = NT;
    results(s).RPE = RPE;
    results(s).VS = VS;
    %figure; subplot 121; plot(results(sess).dI); hold on; plot(results(sess).V); plot(results.beta)
    %subplot 122; plot(results(sess).dI); hold on; plot(results(sess).dV); plot(results.beta1)
end

% check expected cost
%figure; hold on; plot(results.cost);plot(results.ecost,'k');
%figure; plot(results.dI,'k'); hold on; plot(results.rho,'r');
%figure; plot(results.beta)

end


