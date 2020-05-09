function [O] = contingency_VRworld(A, O, t, p)
%function [O,k] = contingency_VRworld(A, O, t, k, numActions)
%% to simulate the world, VR schedule

% if I just took an action, I have a p probability of observing a reward 
if A(t) == 1
    O(t) = binornd(1,p); % observe reward or not? 
    
end

% action
%if sum(A(find(O(1:t)==1,1,'last'):t)) > numActions(k) && A(t) == 1 % if the last action was less than the number of needed actions then  
%   O(t) = 1; % observe reward
%   k = k+1;
%end 

end