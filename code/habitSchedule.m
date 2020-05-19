function [O,T] = habitSchedule(sched)
% takes in state and action, outputs observation x
% for 4 different schedules: FI, FR, VI, VR

% INPUT:  sched.type -  FR, FI, VR, VI
%         sched.R - ratio arming parameter
%         sched.I - interval arming parameter

% OUTPUT: O - observation matrix
%         T - transition matrix

fig = 0;
nA = 2;    % number of actions {1: nothing, 2: lever press}
nO = 2;    % number of observations {1: nothing, 2: reward}

switch sched.type
    %% FR
    case 'FR'
        nS = sched.R;   % number of states
        
        T = zeros(nS, nS, nA);    % S x S x A
        O = zeros(nS, nA, nO);    % S x A x O
        
        % transition matrix
        % a = wait
        for i = 1:nS
            T(i,i,1) = 1;
        end
        
        % a = tap
        for i = 1:nS-1
            T(i,i+1,2) = 1;
        end
        T(nS,1,2) = 1;
        
        % observation matrix
        O(:,:,1) = 1; % see nothing
        O(1,2,1) = 0; % see nothing
        O(1,2,2) = 1; % see reward (only see if transition into state 1)
        
        %% FI
    case 'FI'
        nS = sched.I;   % number of states
        
        T = zeros(nS, nS, nA);    % S x S x A
        O = zeros(nS, nA, nO);    % S x A x O
        
        % transition matrix
        % a = wait
        for i = 1:nS-1
            T(i,i+1,1) = 1;
        end
        T(nS,nS,1) = 1;
        
        % a = tap
        for i = 1:nS-1
            T(i,i+1,2) = 1;
        end
        T(nS,1,2) = 1;
        
        % observation matrix
        O(:,:,1) = 1; % see nothing
        O(1,2,1) = 0; % see nothing
        O(1,2,2) = 1; % see reward (only see if transition into state 1)
        
        
        %% VR
    case 'VR'
        nS = sched.R;   % number of states
        
        T = zeros(nS, nS, nA);    % S x S x A
        O = zeros(nS, nA, nO);    % S x A x O
        
        % transition matrix
        % a = wait
        for i = 1:nS
            T(i,i,1) = 1;
        end
        
        % a = tap
        for i = 1:nS-1
            T(i,i+1,2) = 0.5; % tap either moves closer to reward
            T(i,1,2) = 0.5;   % or gets reward
        end
        T(nS,nS,2) = 0.5;
        T(nS,1,2) = 0.5;
        
        % observation matrix
        O(:,:,1) = 1; % see nothing
        O(1,2,1) = 0; % see nothing
        O(1,2,2) = 1; % see reward (only see if transition into state 1)
        
        
        %% VI
    case 'VI'
        nS = sched.I;   % number of states
        
        T = zeros(nS, nS, nA);    % S x S x A
        O = zeros(nS, nA, nO);    % S x A x O
        
        % transition matrix
        % a = wait
        for i = 1:nS-1
            T(i,i+1,1) = 1;
        end
        T(nS,nS,1) = 1;
        
         % a = tap
        for i = 1:nS-1
            T(i,i+1,2) = 0.5; % tap either moves closer to reward
            T(i,1,2) = 0.5;   % or gets reward
        end
        T(nS,nS,2) = 0.5;
        T(nS,1,2) = 0.5;
        
        % observation matrix
        O(:,:,1) = 1; % see nothing
        O(1,2,1) = 0; % see nothing
        O(1,2,2) = 1; % see reward (only see if transition into state 1)
        
end


if fig
    figure; hold on;
    subplot 221
    imagesc(T(:,:,1)); % for "null"
    title('a = null')
    axis square; prettyplot
    ylabel('transition matrix')
    set(gca,'YDir','normal')
    subplot 222
    imagesc(T(:,:,2)); % for "reward"
    axis square; prettyplot
    set(gca,'YDir','normal')
    title('a = tap')
    
    
    subplot 223
    imagesc(O(:,:,1)); % for "null"
    title('x = null')
    set(gca,'YDir','normal')
    axis square
    ylabel('observation matrix')
    prettyplot
    subplot 224
    imagesc(O(:,:,2)); % for "reward"
    set(gca,'YDir','normal')
    title('x = reward')
    axis square
    prettyplot
   
    
end
end