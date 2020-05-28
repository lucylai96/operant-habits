function habitAnalysis
% PURPOSE: reproducing major habit findings
% Written by Lucy Lai

% close all
clear all

addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools'); % various plotting tools
map = brewermap(4,'*RdBu');
temp = map(3,:);
map(3,:) = map(4,:);
map(4,:) = temp;                     % swap colors so darker ones are fixed
set(0, 'DefaultAxesColorOrder', map) % first three rows

%% some default parameters
sched.model = 4;     % which lesioned model to run
sched.acost = 0.05;   % action cost
sched.beta = 100;     % starting beta; high beta = low cost. beta should increase for high contingency
sched.cmax = 0.5;     % max complexity (low v high)

%% undertraining vs overtraining
% more habit in overtraining

%% contingency degredation (give free rewards)


%% findings to replicate

%% REED 2001
sched.R = 8;
sched.I = 20;
% In Experiment 1, the time between outcomes obtained on a VR8 (1-16);
% schedule became the intervals for a yoked VI schedule.

% for VR only
sched.actions = contingency_generateVR(sched.R);   % pre-generated number of actions before reward
sched.actions(sched.actions > 16) = [];

sched.k = 1;
sched.timeSteps = 2200;
sched.devalTime = sched.timeSteps;                 % timestep where outcome gets devalued
sched.type = 'VR';

[O,T] =  habitSchedule(sched);
VR = habitAgent(O,T, sched);

outTimes = find(VR.x==2);                     % find time between outcomes

sched.times = [diff(outTimes) diff(outTimes)];
sched.type = 'VI';

[O,T] =  habitSchedule(sched);
VIyoked = habitAgent(O,T, sched);

% Response rates were higher on the VR than on the VI schedule.
figure; hold on; subplot 121; hold on;
actRate(1) = sum(VR.a-1)/sched.timeSteps; % action rate for VR
actRate(2) = sum(VIyoked.a-1)/sched.timeSteps; % action rate for VI yoked
bar(1,actRate(1)*60,'FaceColor',map(2,:));
bar(2,actRate(2)*60,'FaceColor',map(4,:));
ylabel('responses per min')
xlabel('schedule')
set(gca,'xtick',[1 2],'xticklabel',{'ratio','interval'})
title('Reed(2001) Expt 1: VI yoked to VR')
prettyplot


% In Experiment 2, the number of responses required per outcome on a
% VR schedule were matched to that on a master VI 20-s (1-40s) schedule.

% for VI only
sched.times = contingency_generateVI(sched.I);     % pre-generated wait times before reward
sched.times(sched.times>40) = [];
sched.type = 'VI';

[O,T] =  habitSchedule(sched);
VI = habitAgent(O,T, sched);

outTimes = find(VI.x==2);                     % find time between outcomes
k = 1;
for i = 1:length(outTimes)-1
    numAct(k) = sum(VI.a(outTimes(i):outTimes(i+1))-1); % sum up the number of actions between outTimes
    k = k+1;
end

sched.actions = [numAct numAct];
sched.type = 'VR';

[O,T] =  habitSchedule(sched);
VRyoked = habitAgent(O,T, sched);

% Response rates were higher on the VR than on the VI schedule.

subplot 122; hold on;
actRate(1) = sum(VR.a-1)/sched.timeSteps; % action rate for VR
actRate(2) = sum(VIyoked.a-1)/sched.timeSteps; % action rate for VI yoked
bar(1,actRate(1)*60,'FaceColor',map(2,:));
bar(2,actRate(2)*60,'FaceColor',map(4,:));
ylabel('responses per min')
xlabel('schedule')
set(gca,'xtick',[1 2],'xticklabel',{'ratio','interval'})
title('Reed(2001) Expt 2: VR yoked to VI')
prettyplot




% Both ratings of causal effectiveness and response rates were higher in the VR schedule.
% (Analyze contingency)



why


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

end