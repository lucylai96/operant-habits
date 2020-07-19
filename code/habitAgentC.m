function results = habitAgentCont(sched, agent, input)
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
beta = sched.beta;    % beta
acost = sched.acost;  % action cost
deval = ones(1,sched.timeSteps);        % a vector indicating when devaluation "turns on"
deval(:,sched.devalTime:end) = 0;       % 0 means devaluation mode

% learning rates
if nargin <2 % if simulating without input parameters
    alpha_w = 0.1;          % value learning rate
    alpha_t = 0.1;          % policy learning rate
    alpha_r = 0.1;          % rho learning rate
    alpha_b = 1;            % beta learning rate
    wa = 0.15;              % weber fraction for actions
    wt = 0.2;               % weber fraction for time
    
else % if simulating with input parameters
    alpha_w = agent.alpha_w;          % value learning rate
    alpha_t = agent.alpha_t;          % policy learning rate
    alpha_b = agent.alpha_b;          % beta learning rate
    alpha_r = agent.alpha_r;          % rho learning rate
    
end

nS = sched.R*2; % number of features
d = 1:nS;       % number of features
mu_d = 1:nS;    % mean of features
wa = 0.1;       % weber fraction for actions
wt = 0.15;      % weber fraction for time
t = 1:nS;       % time or actions
%c = linspace(0,1,nS);
%mu_c = linspace(0,1,nS);
%wc = 0.01; 
%Cdt = zeros(nS, size(linspace(0,1,nS),2)); % d dimensions
for i = 1:length(d)
    Adt(i,:) = exp((-1/2)*((t-mu_d(i))/(wa*mu_d(i))).^2);
    Tdt(i,:) = exp((-1/2)*((t-mu_d(i))/(wt*mu_d(i))).^2);
    %Cdt(i,:) = exp((-1/2)*((c-mu_c(i))/(wc)).^2);
end


% figure; subplot 311; hold on;
% plot(Adt'); % visualize features
% subplot 312; hold on;
% plot(Tdt'); % visualize features
% subplot 313; hold on;
% plot(Cdt'); % visualize features


% initial values
num = sched.R;   % marginal action probability over the last num seconds
phi = [1; Adt(:,1); Tdt(:,1)]; % features
ps = zeros(length(phi)-1,1);
paa = [1 1];

x = [2]; % observation array init
a = [2]; % action array init
rho = 0;         % avg reward init
paa = [0.5 0.5]; % p(a) init
k = 0;
dI = 0;
dV = 0;

% weights
theta  = zeros(length(phi),2);  % policy weights
w = zeros(length(phi),1);       % value weights

win = 60; % calculate contingency over this window, 30 seconds

for t = 2:sched.timeSteps
    %% define features
    if x(t-1) == 2  % if you just saw reward
        k = k+1;    % counter for # rewards
        rt = t-1;   % log the timestep of last reward
        RT(k) = rt; % counter for reward times
    end
    
    na = sum(a(rt:t-1)==2); % number of actions since last reward
    nt = t-rt;              % timesteps since last reward
    
    if nt>nS || na>nS % a fix for conditions where the time / number of actions exceeds the number of features (rare)
        nt = nS;
        na = nS;
    end
    
    phi0 = phi;
    phi = [1; Adt(:,na); Tdt(:,nt)]; % features composed of indexed by action na and time t since last reward
    phi(isnan(phi)) = 0;
    
    %% contingency (ignore this)
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
    
    % [~,ct] = min(abs(pR(t)-mu_c));
    
    
    %% action history
    if t>num
        pa(1) = sum(a(t-num:t-1)==1)/num; pa(2) = sum(a(t-num:t-1)==2)/num;             % trailing marginal action distribution
    else
        pa(1) = sum(a==1)/t; pa(2) = sum(a==2)/t;
    end
    pa = pa+0.01; pa = pa./sum(pa);
    
    %% action selection
    pi_as = exp(beta.*(theta'*phi)' + log(paa));
    pi_as(pi_as>100) = 100;
    pi_as(pi_as<-100) = -100;
    if sum(pi_as) == 0
        pi_as = [0.5 0.5];
    end
    pi_as = pi_as./sum(pi_as);
    pi_as = pi_as+0.01; pi_as = pi_as./sum(pi_as);
    
    
    if exist('input')==1 && t < 100 % when fitting, seed with data (first 100 seconds)
        a(t) = input(t)+1; % 0 means nothing, 1 means lever press
    else
        % policy sampled stochastically
        if rand < pi_as(1)
            a(t) = 1;      % null
        else
            a(t) = 2;      % press lever
        end
    end
    
    paa(1) = sum(a==1)/t; paa(2) = sum(a==2)/t;             % total marginal action distribution
    paa = paa+0.01; paa = paa./sum(paa);
    
    %% observe reward and new state
    [x(t), sched] = habitWorld(t, a, x, sched);
    
    %% policy complexity
    if t>2
        ps = (sum(results.ps,2)./sum(results.ps(:)));              % probability of being in a state
    end
    
    % should encompass all states and actions (entire policy)
    pas = exp(beta.*(theta(2:end,:))+log(paa));                    % entire state dep policy
    pas = pas./sum(pas,2);
 
    mi = nansum(ps.*nansum(pas.*log(pas./paa),2));      % mutual information
 
    cost0 = log(pi_as./paa);
    cost = cost0(a(t));                                 % policy cost for that state-action pair
    
   
    %% TD update
    switch sched.model % different models with different costs
        case 1 % no action cost, no complexity cost
            r = deval(t)*double(x(t)==2);                % reward (deval is a flag for whether we are in deval mode)
            rpe = r - rho + w'*(phi-phi0);               % TD error
            rho =  rho + alpha_r *(r - rho);             % average reward update
        case 2 % yes action cost, no complexity cost
            r = deval(t)*double(x(t)==2)-acost*(a(t)==2);   
            rpe = r - rho + w'*(phi-phi0);
            rho =  rho + alpha_r *(r - rho);         
        case 3  % yes action cost, yes complexity cost (fixed)
            r = deval(t)*double(x(t)==2)-acost*(a(t)==2);
            rpe = r - rho + w'*(phi-phi0)-(1/beta)*cost;
            rho =  rho + alpha_r *(r - rho);            
        case 4  % yes action cost, yes complexity cost (adaptive)
            r = deval(t)*double(x(t)==2)-acost*(a(t)==2);
            % average reward update
            if t>2*num    
                dI = mean(results.mi(t-num)) - mean(results.mi(t-2*num:t-num)); % change in policy complexity
                beta = beta + alpha_b*dI;
                
                % dV = mean(results.rho(t-num)) - mean(results.rho(t-2*num:t-num));
                % beta = dI/dV;
                
                % bounds on beta
                if beta<0
                    disp('!')
                    beta = 0.1;
                end
                
                if beta > sched.cmax  %|| isnan(beta)
                    beta = sched.cmax;
                end
                
            end
            
            rpe = r - rho + w'*(phi-phi0)-(1/beta)*cost;
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
    theta0 = theta;                              
    theta(:,a(t)) = theta(:,a(t)) + alpha_t*rpe*phi;         % policy weight update
  
    %% store results
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
    results.rt(k) = rt;
    results.dI(t) = dI;
    results.dV(t) = dV;
    results.cont(t) = pR(t); % contingency
    
    results.pi_as(t,:) = pi_as;
    results.a(t) = a(t);
    results.x(t) = x(t);
    
end
end


