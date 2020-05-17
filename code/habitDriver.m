function habitDriver
%% PURPOSE: Main function
% Written by Lucy Lai (May 2020)
% NOTES:
%   T(i,j,k)  = P(s'=j|s=i,a=k) is the probability of transitioning from sub-state i->j after taking action k
%   O(i,j,k) = P(x=k|s=i,a=j) is the probability of observing x in state s after taking action a
%   assumptions: learned reward and transition matrix

close all
map = brewermap(9,'Reds');
colormap(map)
set(0, 'DefaultAxesColorOrder', map) % first three rows

addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools/');                  % various statistical tools


timeSteps = 2000;

%% FR
sched.type = 'FR';
sched.R = 10;      % ratio parameters
[0,T] =  habitSchedule(sched);
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
sched.times = contingency_generateVI(sched.I); % pre-generated wait times
sched.k = 1;      % ratio parameters
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

% verify that the schedules are actually delivering rewards as expected
% (plot outcome delivery times for high action rates)

%% make lesioned agents that don't have diff learning capabilities 

%% some rudimentry data analysis
% how does action rate change over learning 

%% calculate beta
% moving averages of information between states and actions over the
% average reward
% 
% states are the currently occupied state (need to tally the state that the
% animal is in 
% 
% take real data: for VR schedule, have array of when animal generated
% action in each second time bin, then label each array entry with a
% "state," so we know what state it was in at any time
% 
% look at moving averages of action rates

% al


%% put in different agents with



%% beta analysis



end

%         O(:,1,2) = update.O(:,1,2) + a_obs.*update.s.*(double(update.x==2) - update.O(:,1,2)); % rew matrix for a = wait
%         O(:,2,2) = update.O(:,2,2) + a_obs.*update.s.*(double(update.x==2) - update.O(:,2,2)); % rew matrix for a = tap
%         O(:,1,1) = update.O(:,1,1) + a_obs.*update.s.*(double(update.x==1) - update.O(:,1,1)); % rew matrix for a = wait
%         O(:,2,1) = update.O(:,2,1) + a_obs.*update.s.*(double(update.x==1) - update.O(:,2,1)); % rew matrix for a = tap
%        wait - nothing (1,1)
% wait - reward (1,2) never happens
% tap - nothing (2,1)
% tap - reward (2,2)