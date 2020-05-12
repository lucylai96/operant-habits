function habitDriver
%% PURPOSE: Main function
% AUTHOR:
% NOTES:
%   T(i,j,k)  = P(s'=j|s=i,a=k) is the probability of transitioning from sub-state i->j after taking action k
%   O(i,j,k) = P(x=k|s=i,a=j) is the probability of observing x in state s after taking action a

addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools/');                  % various statistical tools

%% testing a learned reward and transition matrix
sched = 1;
fig = 0
%% FR5
%if sched ==1
    R = 10;      % ratio parameters
    
    nS = R;   % number of states
    nA = 2;    % number of actions {1: nothing, 2: lever press}
    nO = 2;    % number of observations {1: nothing, 2: reward}
    
    T = zeros(nS, nS, nA);    % S x S x A
    O = zeros(nS, nA, nO);    % S x A x O
    
    % a = wait
    T(1,1,1) = 1;
    T(2,2,1) = 1;
    T(3,3,1) = 1;
    T(4,4,1) = 1;
    T(5,5,1) = 1;
    T(6,6,1) = 1;
    T(7,7,1) = 1;
    T(8,8,1) = 1;
    T(9,9,1) = 1;
    T(10,10,1) = 1;
    % a = tap
    T(1,2,2) = 1;
    T(2,3,2) = 1;
    T(3,4,2) = 1;
    T(4,5,2) = 1;
    T(5,6,2) = 1;
    T(6,7,2) = 1;
    T(7,8,2) = 1;
    T(8,9,2) = 1;
    T(9,10,2) = 1;
    T(10,1,2) = 1;
    
    O(:,:,1) = 1; % o = nothing
    O(1,2,1) = 0; % 1 = nothing
    O(1,2,2) = 1; % 2 = reward (i see it in state 1)
    
    
    if fig
        figure; hold on;
        subplot 121
        imagesc(O(:,:,1)); % for "null"
        title('x = null')
        set(gca,'YDir','normal')
        axis square
        prettyplot
        subplot 122
        imagesc(O(:,:,2)); % for "reward"
        set(gca,'YDir','normal')
        title('x = reward')
        axis square
        prettyplot
        
        
        figure; hold on;
        subplot 121
        imagesc(T(:,:,1)); % for "null"
        title('a = null')
        axis square; prettyplot
        set(gca,'YDir','normal')
        subplot 122
        imagesc(T(:,:,2)); % for "reward"
        axis square; prettyplot
        set(gca,'YDir','normal')
        title('a = tap')
        
    end
    
    FR = habitTD(O,T, sched);
%end
%% FI5
%if sched ==2
sched = 2;
    I = 10;      % ratio parameters
    
    nS = I;   % number of states
    nA = 2;    % number of actions {1: nothing, 2: lever press}
    nO = 2;    % number of observations {1: nothing, 2: reward}
    
    T = zeros(nS, nS, nA);    % S x S x A
    O = zeros(nS, nA, nO);    % S x A x O
    
    % a = wait
    T(1,2,1) = 1;
    T(2,3,1) = 1;
    T(3,4,1) = 1;
    T(4,5,1) = 1;
    T(5,6,1) = 1;
    T(6,7,1) = 1;
    T(7,8,1) = 1;
    T(8,9,1) = 1;
    T(9,10,1) = 1;
    T(10,10,1) = 1;
    
    % a = tap
    T(1,2,2) = 1;
    T(2,3,2) = 1;
    T(3,4,2) = 1;
    T(4,5,2) = 1;
    T(5,6,2) = 1;
    T(6,7,2) = 1;
    T(7,8,2) = 1;
    T(8,9,2) = 1;
    T(9,10,2) = 1;
    T(10,1,2) = 1;
    
    O(:,:,1) = 1; % 1 = nothing
    O(1,2,1) = 0; % 1 = nothing
    O(1,2,2) = 1; % 2 = reward (i see it in state 1)
    
    
    if fig
        figure; hold on;
        subplot 121
        imagesc(O(:,:,1)); % for "null"
        title('x = null')
        set(gca,'YDir','normal')
        axis square
        prettyplot
        subplot 122
        imagesc(O(:,:,2)); % for "reward"
        set(gca,'YDir','normal')
        title('x = reward')
        axis square
        prettyplot
        
        
        figure; hold on;
        subplot 121
        imagesc(T(:,:,1)); % for "null"
        title('a = null')
        axis square; prettyplot
        set(gca,'YDir','normal')
        subplot 122
        imagesc(T(:,:,2)); % for "reward"
        axis square; prettyplot
        set(gca,'YDir','normal')
        title('a = tap')
        
    end
    
    FI = habitTD(O,T, sched);
%end


%% figures
num = 200;
figure;hold on;
FRcs = cumsum(FR.a(end-num:end)-1);
plot(FRcs,'b','LineWidth',2); 
xplot = find(FR.x(end-num:end)==2);
line([xplot' xplot'+10]',[FRcs(xplot)' FRcs(xplot)']','LineWidth',2,'Color','k')

FIcs = cumsum(FI.a(end-num:end)-1);
plot(FIcs,'r','LineWidth' ,2); 
xplot = find(FI.x(end-num:end)==2);
line([xplot' xplot'+10]',[FIcs(xplot)' FIcs(xplot)']','LineWidth',2,'Color','k')

%legend('FR','FR reward','FI','FI reward')
%legend('boxoff')
xlabel('time')
ylabel('cumulative # actions')
prettyplot

figure;hold on;
FR_pa = exp(FR.theta(:,:,end));
FR_pa = FR_pa./sum(FR_pa,2);
FI_pa = exp(FI.theta(:,:,end));
FI_pa = FI_pa./sum(FI_pa,2);
imagesc([FR_pa(:,2) FI_pa(:,2)]) 

imagesc(FR.value) 
prettyplot
xlabel('schedule')
ylabel('states')

why

end

%         O(:,1,2) = update.O(:,1,2) + a_obs.*update.s.*(double(update.x==2) - update.O(:,1,2)); % rew matrix for a = wait
%         O(:,2,2) = update.O(:,2,2) + a_obs.*update.s.*(double(update.x==2) - update.O(:,2,2)); % rew matrix for a = tap
%         O(:,1,1) = update.O(:,1,1) + a_obs.*update.s.*(double(update.x==1) - update.O(:,1,1)); % rew matrix for a = wait
%         O(:,2,1) = update.O(:,2,1) + a_obs.*update.s.*(double(update.x==1) - update.O(:,2,1)); % rew matrix for a = tap
%        wait - nothing (1,1)
% wait - reward (1,2) never happens
% tap - nothing (2,1)
% tap - reward (2,2)