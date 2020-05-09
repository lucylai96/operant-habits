function [O] = contingency_FRworld(A, O, t, R)
%% to simulate the world, FR schedule

% after R actions, I get a reward
if sum(A(find(O(1:t)==1,1,'last'):t)) > R
    O(t) = 1; % observe reward
end


end