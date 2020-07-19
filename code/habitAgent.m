function results = habitAgent(sched, agent, input)
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
    alpha_b = 1;            % beta learning rate
    wa = 0.15;     % weber fraction for actions
    wt = 0.2;    % weber fraction for time
    
else
    alpha_w = agent.alpha_w;          % value learning rate
    alpha_t = agent.alpha_t;          % policy learning rate
    alpha_b = agent.alpha_b;          % beta learning rate
    alpha_r = agent.alpha_r;          % rho learning rate
    
end
rho = 0;                % avg reward

nS = sched.R*2;
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
num = sched.R;   % marginal action probability over the last num seconds
%phi = [1; Adt(:,1); Tdt(:,1); Cdt(:,end)]; % features
%imagesc([Adt; Tdt; Cdt])
phi = [1; Adt(:,1); Tdt(:,1)]; % features
%phi = [1; Cdt(:,end)]; % features
x = [2];
a = [2];
paa = [0.5 0.5];
k = 0;
dI = 0;
dV = 0;
% weights
theta  = zeros(length(phi),2);  % policy weights
w = zeros(length(phi),1);       % value weights
Q = zeros(length(phi),2);  % state-action values

win = 60; % calculate contingency over this window, 30 seconds
ps = zeros(length(phi)-1,1);
paa = [1 1];

for t = 2:sched.timeSteps
    %% define features
    if x(t-1) == 2 % if you just saw reward
        k = k+1;  % counter
        rt = t-1;  % log the timestep of last reward
        RT(k) = rt;
    end
    
    na = sum(a(rt:t-1)==2); % number of actions since last reward
    nt = t-rt; % timesteps since last reward
    
    if nt>nS || na>nS
        nt = nS;
        na = nS;
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
    pi_as = exp(beta.*(theta'*phi)' + log(paa));
    pi_as(pi_as>100) = 100;
    pi_as(pi_as<-100) = -100;
    if sum(pi_as) == 0
        pi_as = [0.5 0.5];
    end
    pi_as = pi_as./sum(pi_as);
    pi_as = pi_as+0.01; pi_as = pi_as./sum(pi_as);
    
    
    if exist('input')==1 && t < 200 % seed the fitter with data
        a(t) = input(t)+1; % 0 means nothing, 1 means lever press
    else
        % policy sampled stochastically
        if rand < pi_as(1)
            a(t) = 1;      % null
        else
            a(t) = 2;      % press lever
        end
    end
    % policy deterministic
    %     if pi_as(1) > pi_as(2)
    %         a(t) = 1;      % null
    %     else
    %         a(t) = 2;      % press lever
    %     end
    paa(1) = sum(a==1)/t; paa(2) = sum(a==2)/t;             % total marginal action distribution
    paa = paa+0.01; paa = paa./sum(paa);
    
    %% observe reward and new state
    [x(t), sched] = habitWorld(t, a, x, sched);
    
    %% policy complexity
    %     betatheta = beta.*(theta'*phi)';
    
    if t>2
        ps = (sum(results.ps,2)./sum(results.ps(:)));         % probability of being in a state
    end
    %
    %     % should encompass all states and actions (entire policy)
    pas = exp(beta.*(theta(2:end,:))+log(paa));                    % entire state dep policy
    pas = pas./sum(pas,2);
    %pas = pas(2:end,:);
    %
    mi = nansum(ps.*nansum(pas.*log(pas./paa),2));     % mutual information
    %mi = sum(pi_as.*log(pi_as./pa));            % mutual information
    %pas = sum(pas)./sum(pas(:));
    %mi = sum(pas.*log(pas./paa));            % mutual information
    %     if mi >= log(2)
    %         keyboard
    %     end
    
    cost0 = log(pi_as./paa);
    cost = cost0(a(t));                                 % policy cost for that state-action pair
    
    
    
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
            if t>num
                if x(t-1) == 2 && k > 2 %&& t<sched.devalTime % only update beta every time you see reward
                    %dV = r - results.r(t-1);
                    %dI = mean(results.cost(RT(k-1):RT(k))) - mean(results.cost(RT(k-2):RT(k-1))); % change in policy complexity
                    dI = mean(results.mi(t-num:t-1)) - mean(results.mi(1:num)); % change in policy complexity
                    
                end
            end
        case 3                                % yes action cost, yes complexity cost (fixed)
            r = deval(t)*double(x(t)==2)-acost*(a(t)==2);
            rpe = r - rho + w'*(phi-phi0)-(1/beta)*cost;
            rho =  rho + alpha_r *(r - rho);             % average reward update
        case 4                                % yes action cost, yes complexity cost (adaptive)
            r = deval(t)*double(x(t)==2)-acost*(a(t)==2);
            % average reward update
            if t>2*num
                %if x(t-1) == 2 %&& k > 2 %&& t<sched.devalTime % only update beta every time you see reward
                %dV = r - results.r(t-1);
                %dI = mean(results.cost(RT(k-1):RT(k))) - mean(results.cost(RT(k-2):RT(k-1))); % change in policy complexity
                dI = mean(results.mi(t-num)) - mean(results.mi(t-2*num:t-num)); % change in policy complexity
                dV = mean(results.rho(t-num)) - mean(results.rho(t-2*num:t-num));
                %beta = beta + alpha_b*(dI/rho);
                beta = beta + alpha_b*dI;%
                %dV = mean(results.rho(RT(k-1):RT(k)))-mean(results.rho(RT(k-2):RT(k-1))); % change in avg reward
                %beta = dI/dV;
                if beta<0
                    disp('!')
                    beta = 0.1;
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
            
            %rpe = r - rho + w'*(phi-phi0)-(1/beta)*cost;
            rpe = r - rho + w'*(phi-phi0)-(1/beta)*cost; %adjustable beta but not incorperated into RPE
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
    
    Q(:,a(t)) = Q(:,a(t))+phi.*rho;
    %% store results
    %  results.b(t,:) = b0;
    results.w(t,:) = w0;
    results.theta(:,:,t) = theta0;
    results.r(t) = r; % instantaneous reward
    results.rho(t) = rho; % avg reward
    results.rpe(t,:) = rpe; % RPE
    results.v(t) = w'*phi; % estimated value
    results.cost(t,:) = cost;
    results.cost0(t,:) = sum(cost0);
    results.ps(:,t) = phi(2:end,:);
    results.pa(t,:) = pa;
    results.paa(t,:) = paa;
    results.mi(t) = mi;
    results.beta(t) = beta;
    %results.betatheta(t,:) = betatheta(a(t));
    results.rt(k) = rt;
    results.dI(t) = dI;
    results.dV(t) = dV;
    results.cont(t) = pR(t); % contingency
    
    results.pi_as(t,:) = pi_as;
    results.Q(:,:,t) = Q;
    results.a(t) = a(t);
    results.x(t) = x(t);
    
end
end


