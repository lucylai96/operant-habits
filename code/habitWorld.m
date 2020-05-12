function [x] = habitWorld(t, A, X, sched)
% takes in state and action, outputs observation x
% for 4 different schedules: FI, FR, VI, VR

%% FR
% X is history of observations
% R is ratio requirement
% t is current time
% A is actions

if sched == 1
    R = 10;
    % after R actions, I get a reward
    if sum(A(find(X(1:t-1)==2,1,'last'):t)-1) > R
        x = 2; % observe reward
    else
        x = 1;
    end
    
    %[~, nextS] = max(T(lastS,:,a));
    %[~, x] = max(O(nextS,a,:));
end

%% FI
% X is history of observations
% R is ratio requirement
% t is current time
% A is actions
if sched == 2
    %% to simulate the world, FI schedule
    I = 10;
    % if you waited more than the interval
    if length(X(find(X(1:t-1)==2,1,'last'):t-1)) >= I && A(t) == 2
        x = 2; % observe reward
        
    else
        x = 1;
    end
    
end

end