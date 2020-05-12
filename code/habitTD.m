function results = habitTD(O,T, sched)
% PURPOSE: Free-operant learning in VR and VI environments 
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
S = size(T,1);      % number of states
A = size(O,1);      % number of actions
b = ones(S,1)/S;    % belief state
timeSteps = 2000; % total number timesteps

% weights
theta  = zeros(S,2);            % policy weights
w = zeros(S,1);                 % value weights

% learning rates
alpha_w = 0.2;          % value learning rate
alpha_t = 0.2;          % policy learning rate
alpha_b = 0.2;          % beta learning rate


lastS = S;

x = 2;
a = 2;

for t = 2:timeSteps 
    
    %% take an action according to policy
    p_a = exp(0.3*theta'*b);
    p_a = p_a';
    p_a = p_a./sum(p_a);
    
    %if t>3000
    %    keyboard
    %end
    % deterministic
    %[~,a(t)] = max(p_a);
    % policy sampled stochastically
    if rand(1)<p_a(1)
        a(t) = 1;
    else
       a(t) = 2;
    end
    
    %% observe reward and new state
    [x(t)] = habitWorld(t, a, x, sched);
    
    
    %% belief state calculation
    % b = b+G_blur(idx,:)'; % currently blurring belief by absolute time, may not be super plausible, maybe the reset happens when they tap
    
    b0 = b;                                             % old posterior, used later
    b = ((T(:,:,a(t))'*b0).*squeeze(O(:,a(t),x(t))));   % b = b'*(T.*squeeze(O(:,:,x(t))));
    %b = b';
    b = b./sum(b); % normalize
    
    %% TD update
    w0 = w;
    r = double(x(t)==2)-0.01;            % instant reward
    rpe = r + w'*(b-b0);            % TD error
    % ** add average reward, add complexity cost **
    w = w + alpha_w*rpe*b0;         % weight update
    
    
    %% policy update
    theta0 = theta;                         % reward
    theta(:,a(t)) = theta(:,a(t)) + alpha_t*rpe*b0;         % weight update (check policy gradient..)

    %% store results
    results.b(t,:) = b0;
    results.w(t,:) = w0;
    results.theta(:,:,t) = theta0;
    results.rpe(t,:) = rpe; % average reward RPE
    results.v(t) = w'*b0; % estimated value
    
    results.a(t) = a(t);
    results.x(t) = x(t);
    %results.state(t) = lastS; % state you came from
    
end
why
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


