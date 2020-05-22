function results = habitAgent(O,T,sched,timeSteps)
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
S = size(T,1);        % number of states
b = ones(S,1)/S;      % belief state
beta = sched.beta;    % beta
acost = sched.acost;  % action cost
deval = ones(1,timeSteps);        % a vector of when devaluation turns on
deval(:,sched.devalTime:end) = 0; % 0 means devaluation model

% weights
theta  = zeros(S,2);            % policy weights
w = zeros(S,1);                 % value weights

% learning rates
alpha_w = 0.2;          % value learning rate
alpha_t = 0.2;          % policy learning rate
alpha_b = 0.2;            % beta learning rate

x = 2;
a = 2;

for t = 2:timeSteps 
    
    %% take an action according to policy
    pi_as = exp(theta'*b);
    pi_as = pi_as';
    pi_as = pi_as./sum(pi_as);
    
    %if t>3000
    %    keyboard
    %end
    
    %[~,a(t)] = max(p_a); % deterministic
    
    % policy sampled stochastically
    if rand(1)<pi_as(1)
        a(t) = 1;
    else
       a(t) = 2;
    end
    
    %% observe reward and new state
    [x(t), sched] = habitWorld(t, a, x, sched);
    
    
    %% belief state calculation
    b0 = b;                                             % old posterior, used later
    b = ((T(:,:,a(t))'*b0).*O(:,a(t),x(t)));   % b = b'*(T.*squeeze(O(:,:,x(t))));
    
    if sum(b) <0.01
        keyboard
    end
  
    b = b./sum(b); % normalize
    
    %% policy complexity
    % should encompass all states and actions (entire policy)
    pii = exp(theta.*b); % entire state dep policy
    pii = pii./sum(pii,2);
    
    pa(1) = sum(a==1)/t; pa(2) = sum(a==2)/t;             % marginal action distribution
    pa = pa+0.01; pa = pa./sum(pa);
    mi = sum(sum(pii.*log(pii./pa),2));                   % expected cost
    if isnan(mi)
        keyboard
    end
    %cost = mi;
    cost = log(pi_as./pa);
    cost = cost(a(t));                               % policy cost for that state-action pair
    
    %MI = sum(pi_as.*cost);                                    % calculate mutual information 
    
    %% TD update
    switch sched.model % different models with different costs
        case 1                                % no action cost, no complexity cost
        r = deval(t)*double(x(t)==2);         % reward (deval is a flag for whether we are in deval mode) 
        rpe = r + w'*(b-b0);                  % TD error
        case 2                                % yes action cost, no complexity cost
        r = deval(t)*double(x(t)==2)-acost;   % yes action cost, no complexity cost       
        rpe = r + w'*(b-b0);           
        case 3                                % yes action cost, yes complexity cost (fixed)
        r = deval(t)*double(x(t)==2)-acost;         
        rpe = r + w'*(b-b0)-(1/beta).*cost;        
        case 4                                % yes action cost, yes complexity cost (adaptive)
        r = deval(t)*double(x(t)==2)-acost;        
        rpe = r + w'*(b-b0)-(1/beta).*cost;  
        %rv = results.r(t-100:end)/100;
        beta = beta + alpha_b*r*(sched.cmax-mi); % adaptive beta, beta high = low cost to pay, beta low = high cost to pay
    end
    
    %% value update
    w0 = w;
    w = w + alpha_w*rpe*b0;         % weight update
    
    %% policy update
    theta0 = theta;                         % reward
    theta(:,a(t)) = theta(:,a(t)) + alpha_t*rpe*b0;         % weight update (check policy gradient..)
     
    %% store results
    results.b(t,:) = b0;
    results.w(t,:) = w0;
    results.theta(:,:,t) = theta0;
    results.rpe(t,:) = rpe; % RPE
    results.v(t) = w'*b0; % estimated value
    results.cost(t,:) = cost;
    results.beta(t) = beta;
    results.mi(t) = mi;
    
    results.a(t) = a(t);
    results.x(t) = x(t);
    %results.state(t) = lastS; % state you came from
    
end

%% plots
% if plt == 1
%     figure; hold on;
%     % idx_change =find(change==1);
%
%     % states over time
%     subplot 511
%     plot(results.l_state,'ro-');
%     title('state')
%     ylabel('state #')
%     % line([idx_change' idx_change']',[repmat([0 26],10,1)]','Color','k')
%     %xlabel('timesteps (a.u. ~100ms each)')
%
%     % actions over time
%     subplot 512
%     plot(results.action,'go-');
%     title('action chosen')
%     ylabel(' action (1=wait, 2=tap)')
%     %xlabel('timesteps (a.u. ~100ms each)')
%
%     subplot 513
%     imagesc(results.w');
%     title('state value weights')
%     ylabel('state #')
%     %  line([idx_change' idx_change']',[repmat([0 26],10,1)]','Color','r')
%     set(gca,'YDir','normal')
%     %xlabel('timesteps (a.u. ~100ms each)')
%
%     subplot 514
%     imagesc(results.b');
%     title('inferred belief state')
%     ylabel('state #')
%     % line([idx_change' idx_change']',[repmat([0 26],10,1)]','Color','r')
%     set(gca,'YDir','normal')
%     %xlabel('timesteps (a.u. ~100ms each)')
%
%     subplot 515
%     imagesc(squeeze(results.theta(:,2,:)));
%     ylabel('state #')
%     xlabel('timesteps (a.u. ~100ms each)')
%     set(gca,'YDir','normal')
%     % line([idx_change' idx_change']',[repmat([0 26],10,1)]','Color','r')
%     title('policy weights')
%
%     suptitle(strcat('total correct trials: ',num2str(sum(x>1))));
%
% end

% figure; hold on;
%
% subplot 121
% imagesc(T(:,:,1)); % for "null"
% title('a = null')
% axis square
% set(gca,'YDir','normal')
%
% subplot 122
% imagesc(T(:,:,2)); % for "tone"
% axis square
% set(gca,'YDir','normal')
% title('a = tap')
% suptitle('transition matrices')

end


