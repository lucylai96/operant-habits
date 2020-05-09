function [O,k] = contingency_FIworld(A, O, t, k, int)
%% to simulate the world, FI schedule

% if you waited more than the interval
if length(O(find(O(1:t)==1,1,'last'):t)) > int && A(t) == 1 %   
   O(t) = 1; % observe reward
   k = k+1;
end 

end