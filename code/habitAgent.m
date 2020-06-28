function results = habitAgent(sched, agent)
% PURPOSE: Free-operant learning
% AUTHOR: Lucy Lai
%
% USAGE: results = habitTD(x,O,T)
%
% INPUTS:
%   x - observations (1 = nothing, 2 = reward)
%   a - last action (1 = nothing, 2 = press lever)
%   O - [S' x A x O] observation distribution: O(i,j,m) = P(x'=m|s'=i, a=j)
%   probability of observing x in state s' after taking action a
%   T - [S x S' x A] transition distribution: T(i,j,k) = P(s'=j|s=i, a=k)
%   probability of transitioning to state s' from state s after taking action a
%
% OUTPUTS:
%   results - structure with the following fields:
%        .w - weight vector at each time point (prior to updating)
%        .b - belief state at each time point (prior to updating)
%        .rpe - TD error (after updating)


%% initialization
% states
beta = sched.beta;    % beta
acost = sched.acost;  % action cost
deval = ones(1,sched.timeSteps);        % a vector of when devaluation turns on
deval(:,sched.devalTime:end) = 0; % 0 means devaluation model

% learning rates
if nargin <2
    alpha_w = 0.1;          % value learning rate
    alpha_t = 0.1;          % policy learning rate
    alpha_r = 0.1;          % rho learning rate
    alpha_b = 0.1;            % beta learning rate
    wa = 0.15;     % weber fraction for actions
    wt = 0.2;    % weber fraction for time
    
else
    alpha_w = agent.alpha_w;          % value learning rate
    alpha_t = agent.alpha_t;          % policy learning rate
    %alpha_b = agent.alpha_b;          % beta learning rate
    alpha_r = agent.alpha_r;          % rho learning rate

    
end
rho = 0;                % avg reward

nS = sched.R*20;
d = 1:nS;    % number of features
mu_d = 1:nS; % mean of features
wa = 0.1;     % weber fraction for actions
wt = 0.15;    % weber fraction for time
t = 1:nS;    % time or actions
c = linspace(0,1,nS);
mu_c = linspace(0,1,nS);
wc = 0.01; % contingency is NOT scalar
Cdt = zeros(nS, size(linspace(0,1,nS),2)); % d dimensions
for i = 1:length(d)
    Adt(i,:) = exp((-1/2)*((t-mu_d(i))/(wa*mu_d(i))).^2);
    Tdt(i,:) = exp((-1/2)*((t-mu_d(i))/(wt*mu_d(i))).^2);
    Cdt(i,:) = exp((-1/2)*((c-mu_c(i))/(wc)).^2);
end


% figure; subplot 311; hold on;
% plot(Adt'); % visualize features
% subplot 312; hold on;
% plot(Tdt'); % visualize features
% subplot 313; hold on;
% plot(Cdt'); % visualize features


% initial values
num = 30;   % marginal action probability over the last num timesteps (1 minute)
%phi = [1; Adt(:,1); Tdt(:,1); Cdt(:,end)]; % features
%imagesc([Adt; Tdt; Cdt])
phi = [1; Adt(:,1); Tdt(:,1)]; % features
%phi = [1; Cdt(:,end)]; % features
x = 2;
a = 2;
k = 0;
dI=0;
% weights
theta  = zeros(length(phi),2);  % policy weights
w = zeros(length(phi),1);       % value weights

win = 60; % calculate contingency over this window, 30 seconds
ps = zeros(length(phi)-1,1);
beta1 = 1;
for t = 2:sched.timeSteps
    %% define features
    if x(t-1) == 2 % if you just saw reward
        k = k+1;  % counter
        rt = t-1;  % log the timestep of last reward
        RT(k) = rt;
    end
    
    na = sum(a(rt:t-1)==2); % number of actions since last reward
    nt = t-rt; % timesteps since last reward
    
    if nt>nS
        nt = nS;
    end
    
    if na>nS || nt>nS
        keyboard
    end
    
    %% contingency
    if t>win
        %outTimes = find(x(1:t-1)==2); % 30 seconds
        %     if length(outTimes)<3
        %         ct = 1; % contingency is 0
        %     else
        %HO = entropy(diff(outTimes));
        %actTimes = find(a(1:t-1)==2); % 30 seconds
        
        rr = sum((a(t-win:t-1)==2)+(x(t-win:t-1)==2)==2); % total number of reinforced actions made
        pR(t) = rr/sum(a(t-win:t-1)==2); % total number of actions made in some time window
        
        c = sum((a(t-win:t-1)==1 + x(t-win:t-1)==2)==2); % total number of outcomes when there wasnt an action
        
        % now if you only want contingency in a certain time window...
        
        %         for i = 2:length(outTimes)
        %             at = actTimes((actTimes>outTimes(i-1))+(actTimes<=outTimes(i))==2); % only the relevant actions
        %             ct(i) = length(at)/(length(at)+1); % probability that a response is reinforced (non-reinforced actions/reinforced actions)
        %             %HOA(i) = entropy(outTimes(i)-at);
        %         end
        %end
    else
        rr = sum((a(1:t-1)==2)+(x(1:t-1)==2)==2); % total number of reinforced actions made
        pR(t) = rr/sum(a(1:t-1)==2); % total number of actions made in some time window
        
        %c = sum((a(t-win:t-1)==1 + x(t-win:t-1)==2)==2); % total number of outcomes when there wasnt an action
        
    end
    
    if isnan(pR(t))
        pR(t) = 0;
    end
    %
    %     if t>100
    %         outTimes = find(x(t-win:t-1)==2);
    %         if isempty(outTimes) || length(outTimes) <= 5 % if you haven't seen any rewards, contingency is automatically 0
    %             C = 0;
    %         end
    %
    %         if length(outTimes) > 5
    %             k = 1;
    %             ho = entropy(diff(find(x(t-win:t-1)==2)));
    %             for nt = t-win:t
    %                 if  k < length(outTimes) && nt <= outTimes(k)
    %                     if a(nt) == 2
    %                         ato(nt) = outTimes(k)-nt;% calculate action-to-next outcome
    %                         %numAct(k) = numAct(k)+1;% how many actions before next outcome
    %                     end
    %                 elseif k < length(outTimes) %if current time is not less than the time of next reward
    %                     k = k+1; %go to next reward time
    %                     if a(nt) == 2
    %                         ato(nt) = outTimes(k)-nt;% calculate time-to-next outcome, action to outcome
    %                     end
    %
    %                 end
    %             end
    %         end
    %
    %         %C = contingency(a,x,t,win);
    %         %C(c,:) = mutInfo(a(t-c:t-1),x(t-c:t-1))/entropy(x(t-c:t-1)); % contingency
    %
    %     end
    %     C(isnan(C)) = 0;
    
    [~,ct] = min(abs(pR(t)-mu_c));
    
    phi0 = phi;
    %phi = [1; Adt(:,na); Tdt(:,nt); Cdt(:,ct)]; % features composed of indexed by action na and time t since last reward, 100 features to sum over at time nt
    %phi = [1; Cdt(:,ct)]; % features composed of indexed by action na and time t since last reward, 100 features to sum over at time nt
    phi = [1; Adt(:,na); Tdt(:,nt)]; % features composed of indexed by action na and time t since last reward, 100 features to sum over at time nt
    
    phi(isnan(phi)) = 0;
    
    %% action history
    if t>num
        pa(1) = sum(a(t-num:t-1)==1)/num; pa(2) = sum(a(t-num:t-1)==2)/num;             % trailing marginal action distribution
    else
        pa(1) = sum(a==1)/t; pa(2) = sum(a==2)/t;
    end
    pa = pa+0.01; pa = pa./sum(pa);
    
    %% take an action according to policy
    pi_as = exp(beta.*(theta'*phi)'+log(pa));
    %pi_as(pi_as>100) = 100;
    %pi_as(pi_as<-100) = -100;
    if sum(pi_as) ==0
        pi_as = [0.5 0.5];
    end
    pi_as = pi_as./sum(pi_as);
    
    % policy sampled stochastically
    if rand < pi_as(1)
        a(t) = 1;      % null
    else
        a(t) = 2;      % press lever
    end
    
    
    %% observe reward and new state
    [x(t), sched] = habitWorld(t, a, x, sched);
    
    %% policy complexity
    % should encompass all states and actions (entire policy)
    paa(1) = sum(a==1)/t; paa(2) = sum(a==2)/t;             % total marginal action distribution
    paa = paa+0.01; paa = paa./sum(paa);
    pas = exp(theta); % entire state dep policy
    pas = pas./sum(pas,2);
    pas = pas(2:end,:);
    
    ps = ps+phi(2:end,:); % probability of being in a state
    ps = ps./sum(ps);
    mi = sum(ps.*sum(pas.*log(pas./paa),2));                   % expected cost
    
    cost1 = log(pi_as./pa(a(t)));
    costt = cost1(a(t));                               % policy cost for that state-action pair
    
    %% TD update
    %phis0 = phis;      % old expected features
    %phis = phi*pi_as'; % expected features
    switch sched.model % different models with different costs
        case 1                                % no action cost, no complexity cost
            r = deval(t)*double(x(t)==2);         % reward (deval is a flag for whether we are in deval mode)
            rpe = r - rho + w'*(phi-phi0);                  % TD error
            rho =  rho + alpha_r *(r - rho);             % average reward update
        case 2                                % yes action cost, no complexity cost
            r = deval(t)*double(x(t)==2)-acost*(a(t)==2);   % yes action cost, no complexity cost
            rpe = r - rho + w'*(phi-phi0);
            rho =  rho + alpha_r *(r - rho);             % average reward update
        case 3                                % yes action cost, yes complexity cost (fixed)
            r = deval(t)*double(x(t)==2)-acost*(a(t)==2);
            rpe = r - rho + w'*(phi-phi0)-(1/beta)*costt;
            rho =  rho + alpha_r *(r - rho);             % average reward update
        case 4                                % yes action cost, yes complexity cost (adaptive)
            r = deval(t)*double(x(t)==2)-acost*(a(t)==2);
            % average reward update
            if t>30
                if x(t-1) == 2 && k > 2 %&& t<sched.devalTime% only update beta every time you see reward
                % beta = x(t-1)
                %dI = costt - mean(results.cost(t-30:t-1)); % change in policy complexity
                %dV = r - results.r(t-1);
                %dI = mean(results.cost(RT(k-1):RT(k))) - mean(results.cost(RT(k-2):RT(k-1))); % change in policy complexity
                dI = mean(results.mi(RT(k-1):RT(k))) - mean(results.mi(RT(k-2):RT(k-1))); % change in policy complexity
                %dV = results.rho(RT(k))- results.rho(RT(k-1));
                %dI = results.mi(RT(k))- results.mi(RT(k-1));
                %beta = dI/dV;
                %beta = beta + alpha_b*(dI/rho);
                beta = beta + alpha_b*(dI/rho);
                %dV = mean(results.rho(RT(k-1):RT(k)))-mean(results.rho(RT(k-2):RT(k-1))); % change in avg reward
                %beta = dI/dV;
                if beta<1
                    disp('!')
                    beta = 1;
                end
                if beta > sched.cmax  %|| isnan(beta)
                    beta = sched.cmax;
                end
                %                 if beta>1
                %                     (sched.cmax-cost)/rho); % adaptive beta, beta high = low cost to pay, beta low = high cost to pay
                %                 else
                %                     beta = 1;       % specify a floor
                %                 end
            %end
            %if t>sched.devalTime
            %    beta = mean(results.beta(t-num:t-1));
                end
            end
            
            rpe = r - rho + w'*(phi-phi0)-(1/beta)*costt;
            rho =  rho + alpha_r *(r - rho);
    end
    
    if isnan(rpe) || isinf(rpe)
        keyboard
    end
    
    
    %% value update
    w0 = w;
    w = w + alpha_w*rpe*phi0;                 % weight update
    if sum(isnan(w))>0
        keyboard
    end
    
    
    %% policy update
    theta0 = theta;                            % reward
    pgrad = phi;%(:,a(t)) - phis;                % policy gradient
    theta(:,a(t)) = theta(:,a(t)) + alpha_t*rpe*pgrad;         % policy weight update
    
    %theta(:,a(t)) = theta(:,a(t)) + alpha_t*rpe*b0;         % weight update (check policy gradient..)
    
    %% store results
    %  results.b(t,:) = b0;
    results.w(t,:) = w0;
    results.theta(:,:,t) = theta0;
    results.r(t) = r; % instantaneous reward
    results.rho(t) = rho; % avg reward
    results.rpe(t,:) = rpe; % RPE
    results.v(t) = w'*phi; % estimated value
    results.cost(t,:) = costt;
    results.cost1(t,:) = sum(cost1);
    results.mi(t) = mi;
    results.beta(t) = beta;
    results.beta1(t) = beta1;
    results.rt(k) = rt;
    results.dI(t) = dI;
    results.cont(t) = pR(t); % contingency
    
    results.pa(t) = pi_as(2);
    results.a(t) = a(t);
    results.x(t) = x(t);
    
end
end


