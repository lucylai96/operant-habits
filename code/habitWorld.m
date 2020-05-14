function [x] = habitWorld(t, A, X, sched)
% takes in state and action, outputs observation x
% for 4 different schedules: FI, FR, VI, VR

% INPUT:  sched.type -  FR, FI, VR, VI
%         sched.R - ratio arming parameter
%         sched.I - interval arming parameter
%         X - history of observations
%         t - current time
%         A - [1 x T] action history vector

% OUTPUT: x - observation: {1: nothing, 2: reward}

switch sched.type
    %% FR
    case 'FR'
        R = sched.R;
        % after R actions, I get a reward
        if sum(A(find(X(1:t-1)==2,1,'last'):t)-1) > R
            x = 2; % observe reward
        else
            x = 1;
        end
        
        %% FI
    case 'FI'
        
        I = sched.I;
        if length(X(find(X(1:t-1)==2,1,'last'):t-1)) >= I && A(t) == 2 % if you waited more than the interval
            x = 2; % observe reward
        else
            x = 1;
        end
        %% VR
        
    case 'VR'
        
        p = 1/sched.R; % if I just took an action, I have a p probability of observing a reward
        if A(t) == 2
            x = binornd(1,p)+1; % observe reward or not?
        else
            x = 1;
        end
        %% VI
    case 'VI'
        p = 1/sched.I; % every second, I have a p probability of observing a reward
        
        % require: intTimes?
        %         if length(X(find(X(1:t)==2,1,'last'):t)) > intTimes(k) && A(t) == 2
        %             x = 2; % observe reward
        %             k = k+1;
        %         end
        if A(t) == 2 && binornd(1,p) == 1 % if i lever press AND reward is delivered
            x = 2; % observe reward or not?
        else
            x = 1;
        end
end
end