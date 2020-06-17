function [results]=habitDriver(params,type)
% PURPOSE: for simulating model behavior across all 4 types by specifying
% the schedule parameter
%
% NOTES:
%
% INPUT: conditions:
%        arm = [2 5 10 20]
%        model = [1 2 3 4]
%        timeSteps
%        ac = [0 0.01 0.1]
%        cmax = [0.2 0.5 0.8] % only model 4
%        beta = [10 20 30 40 50] % only for model 3
%        deval = [0 1]
%
% Written by Lucy Lai (May 2020)

% close all
% params: [arm model timeSteps acost beta cmax deval]
if nargin <1
    params = [20 2 8000 0.05 100 0.2 0];
    type = {'FR','VR','FI','VI'};
end

%% unpack parameters
arm = params(1);
model = params(2);
timeSteps = params(3);
ac = params(4);
beta = params(5);
cmax = params(6);
deval = params(7);

%% initialize
sched.R = arm;
sched.I = sched.R;
sched.model = model;     % which lesioned model to run
sched.acost = ac;   % action cost
sched.beta = beta;     % starting beta; high beta = low cost. beta should increase for high contingency
sched.cmax = cmax;     % max complexity (low v high)
% for VI only
sched.times = contingency_generateVI(sched.I); % pre-generated wait times before reward
% for VR only
sched.actions = contingency_generateVR(sched.R); % pre-generated number of actions before reward
sched.k = 1;
sched.timeSteps = timeSteps;

if deval == 1
    sched.devalTime = timeSteps;                % timestep where outcome gets devalued
    sched.timeSteps = timeSteps+round(timeSteps/2);   % devaluation test period
else
    sched.devalTime = timeSteps;                % timestep where outcome gets devalued
end

for i = 1:length(type)
    sched.type = type{i};
    [O,T] =  habitSchedule(sched);
    results(i) = habitAgent(O,T, sched);
end

makePlots(results, sched, type)

end


function makePlots(results, sched, type)
%% init color map
map = brewermap(4,'*RdBu');
temp = map(3,:);
map(3,:) = map(4,:);
map(4,:) = temp;  % swap colors so darker ones are fixed
set(0, 'DefaultAxesColorOrder', map) % first three rows
addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools');                  % various statistical tools

timeSteps = sched.timeSteps;

%% cumulative sum plots
num = 100; % last how many # trials to look at

figure; hold on;
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
prettyplot

%% probability of action (learned policy weights)
for i = 1:length(type)
    results(i).Pa = exp(results(i).theta);
    results(i).Pa = results(i).Pa./sum(results(i).Pa,2);
end

figure; subplot 121;
colormap(flipud(gray))
imagesc([results(1).Pa(:,2,end) results(3).Pa(:,2,end)])
set(gca,'xtick',[1:2],'xticklabel',type([1 3]))
ylabel('state')
subplot 122;
colormap(flipud(gray))
imagesc([results(2).Pa(:,2,end) results(4).Pa(:,2,end)])
set(gca,'xtick',[1:2],'xticklabel',type([2 4]))
subprettyplot(1,2)

%% outcome and action rates over time
win = 500; % # seconds moving window
for i = 1:length(type)
    results(i).outRate = sum(results(i).x==2)/timeSteps;
    results(i).actRate = sum(results(i).a==2)/timeSteps;
    
    % moving average
    results(i).movOutRate = movsum(results(i).x-1, win,'Endpoints','discard')/win;
    results(i).movActRate = movsum(results(i).a-1, win,'Endpoints','discard')/win;
    
    % moving avg reward
    results(i).avgRew = movsum(results(i).r, win,'Endpoints','discard')/win;
    
end

figure; hold on;
subplot(3,4,1:3); hold on;
for i = 1:length(type)
    plot(results(i).movOutRate,'LineWidth',1.5)
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
prettyplot
ylabel('outcome rate')
legend([type 'devaluation'],'FontSize',15); legend('boxoff')

subplot(3,4,5:7); hold on;
for i = 1:length(type)
    plot(results(i).movActRate,'LineWidth',1.5)
end
xlabel('time (s)')
ylabel('action rate')
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
prettyplot

subplot(3,4,4); hold on;
for i = 1:length(type)
    bar(i,results(i).outRate);
end
legend(type); legend('boxoff')
set(gca,'xtick',[1:4],'xticklabel',type)
prettyplot
subplot(3,4,8); hold on;
for i = 1:length(type)
    bar(i,results(i).actRate);
end
set(gca,'xtick',[1:4],'xticklabel',type)
prettyplot

subplot(3,4,9:11); hold on;
for i = 1:length(type)
    plot(results(i).avgRew,'LineWidth',1.5)
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
xlabel('time (s)')
ylabel('avg reward')
prettyplot

set(gcf, 'Position',  [300, 300, 800, 500])

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
    plot(movmean(results(i).cost,win,'Endpoints','discard'),'LineWidth',1.5)
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel('policy cost C(\pi_\theta)')

subplot 413; hold on;
for i = 1:length(type)
    plot(movmean((1./results(i).beta).* results(i).cost', win,'Endpoints','discard'),'LineWidth',1.5)
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel('\beta^{-1} C(\pi_\theta)')

subplot 414; hold on;
for i = 1:length(type)
    plot(movmean(results(i).mi, win,'Endpoints','discard'),'LineWidth',1.5)
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel('I(S;A)')
xlabel('time (s)')
subprettyplot(4,1)

set(gcf, 'Position',  [300, 300, 700, 500])

%% action rate before and after devaluation
if sched.devalTime ~= timeSteps
    devalWin = timeSteps-sched.devalTime;
    
    figure; subplot 121; hold on;
    for i = 1:length(type)
        b = bar(i,[mean(results(i).movActRate(sched.devalTime-devalWin:sched.devalTime)) mean(results(i).movActRate(sched.devalTime+1:end))],'FaceColor',map(i,:));
        b(2).FaceColor = [1 1 1];
        b(2).EdgeColor = map(i,:);
        b(2).LineWidth = 2;
        
    end
    ylabel('action (press) rate')
    legend('before','after','FontSize',15,'Location','NorthWest'); legend('boxoff')
    set(gca,'xtick',[1:4],'xticklabel',type)
    
    
    subplot 122; hold on; % change in action rate
    % change in action rate
    for i = 1:length(type)
        results(i).devalScore = mean(results(i).movActRate(sched.devalTime-devalWin:sched.devalTime)) - mean(results(i).movActRate(sched.devalTime+1:end));
        bar(i,results(i).devalScore,'FaceColor',map(i,:));
    end
    ylabel('devaluation score (pre-post)')
    axis([0 5 0 0.05])
    set(gca,'xtick',[1:4],'xticklabel',type)
     
    % beta at time of deval vs deval score
    subprettyplot(1,2)
    
end
set(gcf, 'Position',  [300, 300, 800, 300])


%% relationship between a changing beta and a changing policy complexity

% moving est of mutual information
for i = 1:length(type)
    results(i).movMI = movmean(results(i).mi, win,'Endpoints','discard');
    results(i).movCost = movmean(results(i).cost,win,'Endpoints','discard');
    results(i).movRew = movmean(results(i).r,win,'Endpoints','discard');
    results(i).movBeta = movmean(results(i).beta,win,'Endpoints','discard');
    results(i).estBeta = gradient(results(i).movCost)./gradient(results(i).movRew);
end

% what is relationship between changing beta and a changing policy complexity
% does beta predict the policy complexity? --> YES, the bigger the beta,
% the bigger the policy complexity is allowed to be

% if dbeta is 0, v changes w I the same
% if dbeta is big, policy cost changes a lot (there's a lot of room to move) for the same reward

%% estimated mutual information (at end of learning)

figure; hold on; subplot 131; hold on;
num = 1000;
for i = 1:length(type)
    bar(i,mean(results(i).cost(end-num:end)),'FaceColor',map(i,:));
end
set(gca,'xtick',[1:4],'xticklabel',type)
ylabel('policy cost C(\pi_\theta)')
prettyplot

% omg CONFIMED cost and mi are the same
%figure; hold on;
%num = 1000;
%for i = 1:length(type)
%    bar(i,mean(results(i).mi(end-num:end)),'FaceColor',map(i,:));
%end

subplot 132; hold on;
for i = 1:length(type)
    plot(results(i).movBeta,results(i).movCost,'.','Color',map(i,:),'MarkerSize',20 )
end
xlabel('\beta')
ylabel('C(\pi_\theta)')
prettyplot

subplot 133; hold on;
for i = 1:length(type)
    plot(results(i).movCost,results(i).movRew,'.','Color',map(i,:),'MarkerSize',20 )
end
xlabel('C(\pi_\theta)')
ylabel('reward')
prettyplot
set(gcf, 'Position',  [300, 300, 1000, 300])

%% MI (action = press)

figure; hold on;
for i = 1:length(type)
    bar(i,mean(results(i).mi(end-num:end)),'FaceColor',map(i,:));
end
set(gca,'xtick',[1:4],'xticklabel',type)
ylabel('mutual information I(S;A)')
prettyplot

%% rho and rpe
% figure; hold on; subplot 121; hold on;
% for i = 1:length(type)
%     plot(movmean(results(i).rho,win));
% end
% 
% ylabel('avg rew')
% prettyplot
% 
% subplot 122; hold on;
% for i = 1:length(type)
%     plot(movmean(results(i).rpe,win));
% end
% xlabel('time(s)')
% ylabel('\delta')
% prettyplot

end
