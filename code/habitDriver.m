function habitDriver
%% PURPOSE: Main function
% Written by Lucy Lai (May 2020)
% NOTES:
%   T(i,j,k)  = P(s'=j|s=i,a=k) is the probability of transitioning from sub-state i->j after taking action k
%   O(i,j,k) = P(x=k|s=i,a=j) is the probability of observing x in state s after taking action a
%   assumptions: learned reward and transition matrix

close all
clear all
map = brewermap(9,'Reds');
set(0, 'DefaultAxesColorOrder', map) % first three rows

addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools/');                  % various statistical tools


timeSteps = 3000;
type = {'FR', 'VR', 'FI', 'VI'};
sched.R = 10; 
sched.I = 10; 
sched.acost = 0.1;   % action cost
sched.cmax = 10;     % max complexity cost
sched.beta = 100;    % starting beta
sched.model = 3;     % which lesioned model to run
sched.devalTime = 1500;
% for VI only
sched.times = contingency_generateVI(sched.I); % pre-generated wait times
sched.k = 1;   

for i = 1:length(type)
    sched.type = type{i};
    [O,T] =  habitSchedule(sched);
    results(i) = habitAgent(O,T, sched, timeSteps);
end 

%% cumulative sum plots 
num = 100; % last how many # trials to look at
figure; hold on;
map = brewermap(4,'*RdBu');
colormap(map)
%set(0, 'DefaultAxesColorOrder', map) % first three rows

for i = 1:length(type)
    results(i).cs = cumsum(results(i).a(end-num:end)-1);
    h(i,:) = plot(results(i).cs,'Color',map(i,:),'LineWidth',2);
    xplot = find(results(i).x(end-num:end)==2);
    line([xplot' xplot'+10]',[results(i).cs(xplot)' results(i).cs(xplot)']','LineWidth',2,'Color','k');
end 

legend(h,type);
legend('boxoff')
xlabel('time')
ylabel('cumulative # actions')
prettyplot(20)

%% analysis 

% probability of action (learned policy weights)
for i = 1:length(type)
    results(i).Pa = exp(results(i).theta(:,:,end));
    results(i).Pa = results(i).Pa./sum(results(i).Pa,2);
end 

figure;
colormap(flipud(gray))
imagesc([results(1).Pa(:,2) results(2).Pa(:,2) results(3).Pa(:,2) results(4).Pa(:,2)])
set(gca,'xtick',[1:length(type)],'xticklabel',type)
ylabel('state')
prettyplot(20)

% outcome and action rates
for i = 1:length(type)
    results(i).outRate = sum(results(i).x==2)/timeSteps;
    results(i).actRate = sum(results(i).a==2)/timeSteps;
    
    % moving average
    win = 100; % # seconds moving window
    results(i).movOutRate = movsum(results(i).x-1, win)/win;
    results(i).movActRate = movsum(results(i).a-1, win)/win;
   
end 

figure; hold on; 
plot(results(1).movOutRate,'k')
figure; hold on; 
plot(results(1).movActRate,'k')

figure; hold on;
map = brewermap(4,'*RdBu');
colormap(map)
for i = 1:length(type)
    bar(i,results(i).outRate);
    bar(i+4,results(i).actRate)
end 


%% findings to replication
% In Experiment 1, the time between outcomes obtained on a variable ratio (VR) schedule became the intervals for a yoked variable interval (VI) schedule. Response rates were higher on the VR than on the VI schedule. In Experiment 2, the number of responses required per outcome on a VR schedule were matched to that on a master VI 20-s schedule. Both ratings of causal effectiveness and response rates were higher in the VR schedule. 
% store action rates 

figure; hold on;



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