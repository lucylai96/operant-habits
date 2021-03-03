function results = habitAgentFit_v0(sched, agent, input)
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
    alpha_e = 0.001;        % expected cost learning rate
    alpha_b = 0.1;          % beta learning rate
    acost = 0.1;            % action cost
else % if simulating with input parameters
    alpha_w = agent.alpha_w;          % value learning rate
    alpha_t = agent.alpha_t;          % policy learning rate
    alpha_r = agent.alpha_r;          % rho learning rate
    alpha_e = agent.alpha_e;          % expected cost learning rate
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
%wa = 1;
%wt = 2;
t = 1:nS;       % time or actions
for i = 1:length(d)
    Adt(i,:) = exp((-1/2)*((t-mu_d(i))/(wa*mu_d(i))).^2); %(1/(sqrt(2*pi*wa*mu_d(i)^2)))*
    Tdt(i,:) = exp((-1/2)*((t-mu_d(i))/(wt*mu_d(i))).^2); %(1/(sqrt(2*pi*wt*mu_d(i)^2)))*
    
    Adt2(i,:)=Adt(i,:)./sum(Adt(i,:));
    Tdt2(i,:)=Tdt(i,:)./sum(Tdt(i,:));
    
end
Adt = Adt2';
Tdt = Tdt2';

% weights
phi0 = [log(0.5);Adt(:,1); Tdt(:,1);0];     % features
phi = phi0;
theta  = zeros(length(phi),2); % policy weights
theta(1,:) = 1;                % bias to not do anything

w = zeros(length(phi),1);      % value weights
num = 1000;                    % marginal action probability over the last num seconds
rho0 = 0;
rho = 0;                       % avg reward init
rho2 = 0;
ecost = 0;

for s = 1:sched.sessnum
    % at start of every session reset these initial values
    rho_c = 0;
    phi0(:,1) = [log(0.5); zeros(length(Adt),1); Tdt(:,1); 0];     % features
    phi0(:,2) = [log(0.5); zeros(length(Adt),1); Tdt(:,1); 0];     % features
    
    phi = phi0;
    x = 2;                                          % observation array init
    a = 1;                                          % action array init
    k = 1; na = 1; nt = 1; NA.rew(k) = na; NT.rew(k) = nt;
    paa = [0.5 0.5]; % p(a) init
    pa = [0.5 0.5];  % p(a) init
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
            %keyboard
        end
        %% action history
        if deval(t) == 0 && test(t) == 0
            if t>num
                pa(1) = sum(a(t-num:t-1)==1)/num; pa(2) = sum(a(t-num:t-1)==2)/num; % trailing marginal action distribution
            else
                pa(1) = sum(a==1)/t; pa(2) = sum(a==2)/t;
            end
            pa = pa+0.01; pa = pa./sum(pa);
        else
            why
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
            phi0(:,1) = [0; zeros(length(Adt),1); zeros(length(Tdt),1); 1];
            phi0(:,2) = [0; zeros(length(Adt),1); zeros(length(Tdt),1); 1];
            phi = phi0;
        elseif test(t) == 1 % if in test mode
            phi0(:,1) = [log(pa(1)); Adt(:,na); Tdt(:,nt); 1]; % "satiated"
            phi0(:,2) = [log(pa(2)); Adt(:,na); Tdt(:,nt); 1];
            phi = phi0;
        else
            phi(:,1) = [log(pa(1)); Adt(:,na); Tdt(:,nt); 0]; % features composed of indexed by action na and time t since last reward
            phi(:,2) = [log(pa(2)); Adt(:,na); Tdt(:,nt); 0]; % features composed of indexed by action na and time t since last reward
        end
        if x(t-1)==2
            phi0 = phi;     % features
        end
        
        %% action selection
        act = [theta(:,1)'*phi(:,1) theta(:,2)'*phi(:,2)];  % action probabilities
        %act = (beta*(theta'*phi)+log(pa)');  % action probabilities
        
        value = [theta(2:end,1)'*phi(2:end,1) theta(2:end,2)'*phi(2:end,2)];
        habit = [theta(1,1)'*phi(1,1) theta(1,2)'*phi(1,2)];
        pi_as = [1./(1+exp(-(act(1)-act(2)))) 1-(1./(1+exp(-(act(1)-act(2)))))+0.01];
        pi_as = pi_as +0.01; pi_as = pi_as./sum(pi_as);
        
        if exist('input') == 1 && t<=length(input.session(s).training.lever_binned)  % if fitting to data, condition on data
            a(t) = input.session(s).training.lever_binned(t)+1;                     % 1 means nothing, 2 means lever press
            x(t) = input.session(s).training.reward_binned(t)+1;                    % 1 means nothing, 2 means reward
        elseif deval(t)==1 % satiation manipulation
            a(t) = 1;
            x(t) = 2;
        else
            % policy sampled stochastically
            if rand < pi_as(1)
                a(t) = 1;      % null
            else
                a(t) = 2;      % press lever
            end
            
            if  exist('input') == 1 && (sched.deval == 0 || test(t)==1) % extinction for the valued case
                x(t) = 1;
            else
                % observe reward and new state
                [x(t), sched] = habitWorld(t, a, x, sched);
            end
            
        end
        
        % reward counter
        if x(t) == 2    % if you just saw reward
            k = k+1;    % counter for # rewards
            rt = t;     % log the timestep of last reward
            RT(k) = rt; % counter for reward times
        end
        
        %% policy complexity
        %if k > 1 % after 2 rewards, start using marginal, or else too much initial bias
        %   paa(1) = sum(a==1)/t; paa(2) = sum(a==2)/t;                    % total marginal action distribution
        %   paa = paa+0.01; paa = paa./sum(paa);
        %end
        
        %if t>2
        %    ps = (sum(results(s).ps,2)./sum(results(s).ps(:)));  % probability of being in a state
        %end
        
        % should encompass all states and actions (entire policy)
        pas = zeros(size(theta));
        pas(1,:) = exp(theta(1,:).*phi(1,:));                                          % entire state dep policy
        pas(2:end,:) = exp(theta(2:end,:));                                          % entire state dep policy
        pas = pas./sum(pas,2);
        
        %mi = nansum(ps.*nansum(pas.*log(pas./paa),2));                 % mutual information
        
        cost0 = log(pi_as./pa);
        cost = cost0(a(t));
        
        if deval(t) == 0 && test(t) == 0 %&& x(t)==2% only update when not in test or deval mode
            ecost0 = ecost;
            ecost =  ecost + alpha_e * (cost - ecost);
        end
        
        %% TD update
        switch sched.model % different models with different costs
            case 1 % no action cost, no complexity cost
                rna = ~test(t)*double(x(t)==2); % reward without action cost
                r = ~test(t)*double(x(t)==2);                % reward (deval is a flag for whether we are in deval mode)
                rpe = r - rho + w'*(phi-phi0);               % TD error
            case 2 % yes action cost, no complexity cost
                rna = ~test(t)*double(x(t)==2); % reward without action cost
                r = ~test(t)*double(x(t)==2)-acost*(a(t)==2);
                rpe = r - rho + w'*(phi(:,a(t))-phi0(:,a(t)));
            case 3  % yes action cost, yes complexity cost (fixed)
                rna = ~test(t)*double(x(t)==2); % reward without action cost
                r = ~test(t)*double(x(t)==2)-acost*(a(t)==2);
                rpe = r - rho + w'*(phi(:,a(t))-phi0(:,a(t)))-(1/beta)*ecost;
                
            case 4  % yes action cost, yes complexity cost (adaptive)
                rna = ~test(t)*double(x(t)==2); % reward without action cost
                r = ~test(t)*double(x(t)==2)-acost*(a(t)==2);
                rpe = r - rho + w'*(phi(:,a(t))-phi0(:,a(t)))-(1/beta)*ecost;
                %rpe = r - rho + w'*(phi-phi0);
                
                if deval(t) == 0 && test(t) == 0 && k>2 && x(t) == 2 % only update when not in satiation mode or in test mode
                    % every trial, but doesn't work
                    % dI = ecost-ecost0;
                    % beta = beta + alpha_b*(dI/rho);
                    
                    % only at reward points -- fluctuates a lot but correct trend
                    %dI = mean([results.ecost(RT(k-1)+1:t-1) ecost]) - mean(results.ecost(RT(k-2)+1:RT(k-1)));
                    %V = mean([results(s).r(RT(k-1):t-1) r]);
                    %beta = beta + alpha_b*(dI/V);
                    
                    % only at reward points -- fluctuates a lot but correct trend
                    %dI = ecost - results.ecost(RT(k-1));
                    %dV = rho - results.rho(RT(k-1));
                    %dIdV = dI/dV;
                    %beta = beta + alpha_b*(dI/dV - beta);
                    
                    %dI = mean([results(s).cost(RT(k-1)+1:t-1) cost]) - mean(results(s).cost(RT(k-2)+1:RT(k-1))); % diff in cost over last two reward intervals
                    dI = ecost - results.ecost(RT(k-1));
                    
                    %dV = mean([results(s).r(RT(k-1)+1:t-1) r]) - mean(results(s).r(RT(k-2)+1:RT(k-1))); % diff in reward over last two reward intervals
                    %V = mean([results(s).r(RT(k-1)+1:t-1) r]);
                    
                    %dI = ecost - ecost0;
                    %dV = rho - rho0;
                    beta = beta + alpha_b*(dI/rho);
                    
                    %dV = rho - rho0;
                    %if abs(dV)>0.001
                    %    beta = beta + alpha_b*(dI/dV - beta);
                    %end
                    %if rho > -0.01 && rho < 0.01
                    %    rho = 0.005;
                    %    rho_c = rho_c + 1;
                    %end
                    
                    %if dV> -0.001 && dV < 0.001
                    %    dV = 0.001;
                    %end
                    %dI = ecost - results.ecost(RT(k-1));
                    %beta = beta + alpha_b*(dI/rho);
                    
                    
                    % only at reward points (problem is that dV is often too small)
                    %dI = mean([results.cost(RT(k-1)+1:t-1) cost]) - mean(results.cost(RT(k-2)+1:RT(k-1))); % diff in cost over last two reward intervals
                    %V = mean([results.r(RT(k-1)+1:t-1) r]); % diff in reward over last two reward intervals
                    %beta = beta + alpha_b*(dI/V);
                    
                    % bounds on beta
                    if beta < 0
                        %disp('!')
                        beta = 0.01;
                    elseif beta > sched.cmax
                        %disp('!')
                        beta = sched.cmax;
                    end
                end
                
                
                
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
        VS(t) = w'*(phi(:,a(t))-phi0(:,a(t)));
        
        if isnan(rpe) || isinf(rpe)
            keyboard
        end
        
        %% avg reward update
        %if x(t) == 2 % only update if not in test mode
        rho0 = rho;
        %rho =  rho + alpha_r*(r-rho);           % average reward update
        rho =  rho + alpha_r*(rna-rho);           % average reward update
        
        %end
        
        if test(t) == 0 % only update if not in test mode
            
            %% value update
            w0 = w;
            w = w + alpha_w*rpe*phi0(:,a(t));                                 % weight update with value gradient
            
            if sum(isnan(w))>0
                keyboard
            end
            
            %% policy update
            
            theta0 = theta;
            theta(:,a(t)) = theta(:,a(t)) + alpha_t*rpe*phi0(:,a(t));         % policy weight update
            
        end
        %% store results
        results(s).a(t) = a(t);         % action
        results(s).x(t) = x(t);         % observation
        %results(s).r(t) = r;            % instantaneous reward
        results(s).rho(t) = rho;        % expected reward (avg reward)
        %results(s).rho2(t) = rho2;     % expected reward (avg reward)
        %results(s).rpe(t,:) = rpe;      % RPE
        results(s).pi_as(t,:) = pi_as;  % chosen policy
        %         results(s).pas(:,:,t) = pas;    % entire state-feature policy
        %         results(s).val(t,:) = value;
        %         results(s).hab(t,:) = habit;
        %         results(s).w(t,:) = w0;            % state value weights
        %         results(s).theta(:,:,t) = theta0;  % policy weights
        %
        %
        %         results(s).cost(t) = cost;       % policy cost for the action taken C(p(a|s))
        %         results(s).cost0(t) = sum(cost0);       % policy cost for the action taken C(p(a|s))
        results(s).ecost(t) = ecost;      % expected cost (avg cost)
        %results(s).phi(:,t) = phi;        % state feature visits
        %results(s).pa(t,:) = pa;          % marginal action probability over 5 trials
        %results(s).paa(t,:) = paa;       % total marginal action probability (entire history)
        %results(s).mi(t) = mi;
        results(s).beta(t) = beta;
        %results(s).dIdV(t) = dIdV;
        %results(s).dV(t) = dV;
        % results(s).V(t) = V;
        
        t = t+1;
        
        
    end
    results(s).sched = sched;
end

end


