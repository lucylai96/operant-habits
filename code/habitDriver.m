function habitDriver
%% PURPOSE: Main function
% Written by Lucy Lai (May 2020)
% NOTES:
%   T(i,j,k)  = P(s'=j|s=i,a=k) is the probability of transitioning from sub-state i->j after taking action k
%   O(i,j,k) = P(x=k|s=i,a=j) is the probability of observing x in state s after taking action a
%   assumptions: learned reward and transition matrix

%close all
clear all

addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools');                  % various statistical tools
map = brewermap(9,'Reds');
%% conditions
%arm param: [2 5 10 20]
%a cost: 0 or 0.1
% cmax always the same % only model 4
%beta = [10 20 30 40 50] % only for model 3 
%deval = [0 1](do some with and without deval
% model

%%

timeSteps = 5000;
type = {'FR', 'VR', 'FI', 'VI'};
sched.R = 10;
sched.I = 10;
sched.acost = 0.1;   % action cost
sched.cmax = 10;     % max complexity cost
sched.beta = 50;    % starting beta; high beta = low cost. beta should increase for high contingency
sched.model = 3;     % which lesioned model to run {1:}
sched.devalTime = timeSteps/2;   % timestep where outcome gets devalued
%sched.devalTime = timeSteps;     % timestep where outcome gets devalued
% for VI only
sched.times = contingency_generateVI(sched.I); % pre-generated wait times
sched.k = 1;

for i = 1:length(type)
    sched.type = type{i};
    [O,T] =  habitSchedule(sched);
    results(i) = habitAgent(O,T, sched, timeSteps);
end

%% cumulative sum plots
num = 200; % last how many # trials to look at
figure; hold on;
map = brewermap(4,'*RdBu');
set(0, 'DefaultAxesColorOrder', map) % first three rows
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

%% probability of action (learned policy weights)
for i = 1:length(type)
    results(i).Pa = exp(results(i).theta);
    results(i).Pa = results(i).Pa./sum(results(i).Pa,2);
end

figure;
colormap(flipud(gray))
imagesc([results(1).Pa(:,2,end) results(2).Pa(:,2,end) results(3).Pa(:,2,end) results(4).Pa(:,2,end)])
set(gca,'xtick',[1:length(type)],'xticklabel',type)
ylabel('state')
prettyplot(20)

%% outcome and action rates over time
for i = 1:length(type)
    results(i).outRate = sum(results(i).x==2)/timeSteps;
    results(i).actRate = sum(results(i).a==2)/timeSteps;
    
    % moving average
    win = 100; % # seconds moving window
    results(i).movOutRate = movsum(results(i).x-1, win,'Endpoints','discard')/win;
    results(i).movActRate = movsum(results(i).a-1, win,'Endpoints','discard')/win;
    
end

figure; hold on;
subplot(2,4,1:3); hold on;
for i = 1:length(type)
    plot(results(i).movOutRate,'LineWidth',1.5)
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
prettyplot
ylabel('outcome rate')
legend([type 'devaluation'],'FontSize',15); legend('boxoff')

subplot(2,4,5:7); hold on;
for i = 1:length(type)
    plot(results(i).movActRate,'LineWidth',1.5)
end
xlabel('time (s)')
ylabel('action rate')
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
prettyplot

subplot(2,4,4); hold on;
for i = 1:length(type)
    bar(i,results(i).outRate);
end
legend(type); legend('boxoff')
set(gca,'xtick',[1:4],'xticklabel',type)
prettyplot
subplot(2,4,8); hold on;
for i = 1:length(type)
    bar(i,results(i).actRate);
end
set(gca,'xtick',[1:4],'xticklabel',type)
prettyplot

%% beta 

figure; hold on;
subplot 411; hold on;
for i = 1:length(type)
    plot(results(i).beta,'LineWidth',1.5)
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel(' \beta')
legend([type 'devaluation'],'FontSize',15); legend('boxoff')

subplot 412; hold on;
for i = 1:length(type)
    plot(sum(results(i).cost,2),'LineWidth',1.5)
end

ylabel('cost')

subplot 413; hold on;
for i = 1:length(type)
    plot((1./results(i).beta).* results(i).cost','LineWidth',1.5)
end
ylabel('1/ \beta * cost')

subplot 414; hold on;
for i = 1:length(type)
    plot(results(i).mi,'LineWidth',1.5)
end
ylabel('mutual information')
subprettyplot(4,1)

%% action rate before and after devaluation

%preDevalRate = sum(results(i).a(1:sched.devalTime)==2)/sched.devalTime;
%postDevalRate = sum(results(i).a(sched.devalTime+1:end)==2)/(timeSteps - sched.devalTime);
if sched.devalTime ~= timeSteps
    figure; hold on;
    for i = 1:length(type)
        %barwitherr([sem(results(i).movActRate(1:sched.devalTime),2) sem(results(i).movActRate(sched.devalTime+1:end),2)], [mean(results(i).movActRate(1:sched.devalTime)) mean(results(i).movActRate(sched.devalTime+1:end))]);
        bar(i,[mean(results(i).movActRate(1:sched.devalTime)) mean(results(i).movActRate(sched.devalTime+1:end))],'FaceColor',map(i,:));
    end
    ylabel('action (press) rate')
    set(gca,'xtick',[1:4],'xticklabel',type)
    prettyplot
end


why

%% undertraining vs overtraining

% general purpose driver:
% need to be able to input # of trials
% what type of schedule
% action cost
% beta, max complecity
% what kind of model: whether beta is fixed or not
% devaluation time
% contingency degredation (give free rewards)


sched.R = 10;
sched.I = 10;
sched.acost = 0.1;   % action cost
sched.cmax = 10;     % max complexity cost
sched.beta = 100;    % starting beta; high beta = low cost. beta should increase for high contingency
sched.model = 3;     % which lesioned model to run {1:}
sched.devalTime = timeSteps/2;   % timestep where outcome gets devalued

%% findings to replication
% In Experiment 1, the time between outcomes obtained on a variable ratio (VR) schedule became the intervals for a yoked variable interval (VI) schedule. Response rates were higher on the VR than on the VI schedule. In Experiment 2, the number of responses required per outcome on a VR schedule were matched to that on a master VI 20-s schedule. Both ratings of causal effectiveness and response rates were higher in the VR schedule.
% store action rates

%figure; hold on;



%% analysis

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