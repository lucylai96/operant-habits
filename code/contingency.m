function C = contingency(a,o,t,win)
% a = [0 1] vector of actions
% o = [0 1] vector of outcomes
% t = current time
% win = how many trials back

a = a-1;
o = o-1;
k = 1;

outTimes = find(o==1);
new_t = t-win;
for nt = new_t:win
    if  k<length(outTimes)&&nt<=outTimes(k)
        if A(nt) == 1
            ATO(nt) = outTimes(k)-nt;% calculate action-to-next outcome
            %numAct(k) = numAct(k)+1;% how many actions before next outcome
        end
        if R(nt) == 1
            RTO(nt) = FI.outTimes(k)-nt;% calculate time-to-next outcome, rand To Outcome
        end
    elseif k<length(FI.outTimes) %if current time is not less than the time of next reward
        
        k = k+1; %go to next reward time
        if A(nt) == 1
            ATO(nt) = FI.outTimes(k)-nt;% calculate time-to-next outcome, action to outcome
        end
        
        if R(nt) == 1
            RTO(nt) = FI.outTimes(k)-nt;% calculate time-to-next outcome, rand To Outcome
        end
        
        
    end
end

ho = entropy(outTimes);
hoa = entropy(ATO);
C = (ho-hoa)/ho;
end