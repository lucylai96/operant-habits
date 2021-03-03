function [O,k] = contingency_VIworld(A, O, t, k, intTimes)
%% to simulate the world, VI schedule

% if i tap, will i collect reward?
if length(O(find(O(1:t)==1,1,'last'):t)) > intTimes(k) && A(t) == 1 %   
   O(t) = 1; % observe reward
   k = k+1;
end 


end