function habitDriver
%% PURPOSE: Main function
% AUTHOR:
% NOTES:
%   T(i,j,k)  = P(s'=j|s=i,a=k) is the probability of transitioning from sub-state i->j after taking action k
%   O(i,j,k) = P(x=k|s=i,a=j) is the probability of observing x in state s after taking action a

close all
map = brewermap(9,'Reds');
colormap(map)
set(0, 'DefaultAxesColorOrder', map) % first three rows

addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools/');                  % various statistical tools

%% testing a learned reward and transition matrix

fig = 0;
timeSteps = 2000;

nS = 10;   % number of states
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

T(:,:,1) = T(:,:,1)./sum(T(:,:,1),2);

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
T(10,10,2) = 1;
T(1:10,1,2) = 1;

T(:,:,2) = T(:,:,2)./sum(T(:,:,2),2);

O(:,:,1) = 1; % o = nothing
O(1,2,1) = 0; % 1 = nothing
O(1,2,2) = 1; % 2 = reward (i see it in state 1)


if fig
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
    suptitle('transition matrix')
    
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
    suptitle('observation matrix')
    
end

%% FR
sched.type = 'FR';
sched.R = 10;      % ratio parameters
FR = habitAgent(O,T, sched, timeSteps);

%% FI
sched.type = 'FI';
sched.I = 10;      % ratio parameters
FI = habitAgent(O,T, sched, timeSteps);

%% VR
sched.type = 'VR';
sched.R = 5;      % ratio parameters
VR = habitAgent(O,T, sched, timeSteps);

%% VI
sched.type = 'VI';
sched.I = 5;      % ratio parameters
VI = habitAgent(O,T, sched, timeSteps);


%% figures
num = 100; % last how many # trials to look at
figure;hold on;
FRcs = cumsum(FR.a(end-num:end)-1);
plot(FRcs,'b','LineWidth',2);
xplot = find(FR.x(end-num:end)==2);
line([xplot' xplot'+10]',[FRcs(xplot)' FRcs(xplot)']','LineWidth',2,'Color','k')

FIcs = cumsum(FI.a(end-num:end)-1);
plot(FIcs,'r','LineWidth' ,2);
xplot = find(FI.x(end-num:end)==2);
line([xplot' xplot'+10]',[FIcs(xplot)' FIcs(xplot)']','LineWidth',2,'Color','k')

VRcs = cumsum(VR.a(end-num:end)-1);
plot(VRcs,'g','LineWidth',2);
xplot = find(VR.x(end-num:end)==2);
line([xplot' xplot'+10]',[VRcs(xplot)' VRcs(xplot)']','LineWidth',2,'Color','k')

VIcs = cumsum(VI.a(end-num:end)-1);
plot(VIcs,'m','LineWidth',2);
xplot = find(VI.x(end-num:end)==2);
line([xplot' xplot'+10]',[VIcs(xplot)' VIcs(xplot)']','LineWidth',2,'Color','k')

xlabel('time')
ylabel('cumulative # actions')
prettyplot

FR_pa = exp(FR.theta(:,:,end));
FR_pa = FR_pa./sum(FR_pa,2);
FI_pa = exp(FI.theta(:,:,end));
FI_pa = FI_pa./sum(FI_pa,2);
VR_pa = exp(VR.theta(:,:,end));
VR_pa = VR_pa./sum(VR_pa,2);
VI_pa = exp(VI.theta(:,:,end));
VI_pa = VI_pa./sum(VI_pa,2);
figure;hold on;
imagesc([FR_pa(:,2) FI_pa(:,2) VR_pa(:,2) VI_pa(:,2)])



%% analysis
VR.lo = sum(VR.x==2)/timeSteps; % outcome rate
VR.la = sum(VR.a==2)/timeSteps; % action rate

% can do moving averages of action rates


%% beta analysis

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