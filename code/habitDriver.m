function [results] = habitDriver(params,type)
% PURPOSE: for simulating model behavior across all 4 operant schedules
%
% INPUT: params. - structure with the following fields: (and their bounds)
%               arm - arming parameter (e.g., [2 5 10 20])
%               model - which model is being simulated 
%                       (1 = no action cost / no policy cost, 
%                        2 = action cost only, 
%                        3 = action and fixed policy cost, 
%                        4 = action and changing policy cost)
%               timeSteps - total number of timesteps, (in seconds)
%               ac - action cost (e.g., 0-1)
%               beta - tradeoff parameter, only relevant for model 3 & 4 (e.g., 0-10)
%               cmax - max capacity / upperbound, only relevant for model 4 (e.g., 0-100) 
%               deval - whether devaluation is simulated or not (standard
%                       protocol is 5 minutes of "test," or 300 timesteps/seconds)
% NOTES:
%
% Written by Lucy Lai (May 2020)

clear all
% close all

maxiter = 10; % how many iterations to run / avg over

addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools');                  % various plot tools

% params: [arm model timeSteps ac beta cmax deval]
if nargin <1
    params =  [20 4 1800 0.12 2 20 1]; % 1800 timesteps = 30 minutes, 3600 = 1hr
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
sched.acost = ac;        % action cost
sched.beta = beta;       % starting beta; high beta = low cost. beta should increase for high contingency
sched.cmax = cmax;       % max complexity (low v high)

% for VI only
sched.times = contingency_generateVI(sched.I); % pre-generated wait times before reward
% for VR only
sched.actions = contingency_generateVR(sched.R); % pre-generated number of actions before reward
sched.k = 1;
sched.timeSteps = timeSteps;

if deval == 1
    sched.devalTime = timeSteps;                % timestep where outcome gets devalued
    sched.timeSteps = timeSteps+300;             % devaluation test period = 5 mins (300 timeSteps)
else
    sched.devalTime = timeSteps;                % timestep where outcome gets devalued
end

% agent parameters
agent.alpha_w = 0.1;         % value learning rate
agent.alpha_t = 0.1;         % policy learning rate
agent.alpha_r = 0.1;         % rho learning rate
agent.alpha_b = 2;           % beta learning rate

% run simulations
for i = 1:length(type) % for each schedule
    for iter = 1:maxiter % for each iteration
        sched.type = type{i};
        results(i,iter) = habitAgent(sched, agent);
    end
end

if maxiter>1 % for averaging over many simulations
    makePlotsBS(results, sched, type)
else
    makePlots(results, sched, type)
end

end


function makePlotsBS(results, sched, type)
%% init color map
map = brewermap(4,'*RdBu');
temp = map(3,:);
map(3,:) = map(4,:);
map(4,:) = temp;  % swap colors so darker ones are fixed
set(0, 'DefaultAxesColorOrder', map) % first three rows

timeSteps = sched.timeSteps;

%% outcome and action rates over time
win = 200; % # seconds moving window
iters = size(results,2); % get number of iterations

for i = 1:length(type)
    for it = 1:iters
        results(i,it).outRate = sum(results(i,it).x==2)/timeSteps;
        results(i,it).actRate = sum(results(i,it).a==2)/timeSteps;
        results(i,it).avgRew = mean(results(i,it).r);
        
        % moving average
        results(i,it).movOutRate = movmean(results(i,it).x-1, win,'Endpoints','shrink');
        results(i,it).movActRate = movmean(results(i,it).a-1, win,'Endpoints','shrink');
        
        % moving avg reward (includes action cost)
        results(i,it).movAvgRew = movmean(results(i,it).rho, win,'Endpoints','shrink');
        
    end
end

figure; hold on;
subplot(3,4,1:3); hold on;
for i = 1:length(type)
    h(i,:) = shadedErrorBar(1:timeSteps,mean(cat(1, results(i,:).movOutRate)), std(cat(1, results(i,:).movOutRate)),{'LineWidth',1.5,'Color',map(i,:)},1);
end

plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
prettyplot
ylabel('outcome rate')
legend([h.mainLine],[type 'devaluation'],'FontSize',15); legend('boxoff')

subplot(3,4,5:7); hold on;
for i = 1:length(type)
    shadedErrorBar(1:timeSteps,mean(cat(1, results(i,:).movActRate)), std(cat(1, results(i,:).movActRate)),{'LineWidth',1.5,'Color',map(i,:)},1);
end
xlabel('time (s)')
ylabel('action rate')
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
prettyplot

subplot(3,4,4);  hold on;
for i = 1:length(type)
    bar(i,mean(cat(1, results(i,:).outRate)));
end
set(gca,'xtick',[1:4],'xticklabel',type)
prettyplot

subplot(3,4,8); hold on;
for i = 1:length(type)
    bar(i,mean(cat(1, results(i,:).actRate)));
end
set(gca,'xtick',[1:4],'xticklabel',type)
prettyplot

subplot(3,4,9:11); hold on;
for i = 1:length(type)
    shadedErrorBar(1:timeSteps,mean(cat(1, results(i,:).movAvgRew)), std(cat(1, results(i,:).movAvgRew)),{'LineWidth',1.5,'Color',map(i,:)},1);
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
xlabel('time (s)')
ylabel('avg reward')
prettyplot

subplot(3,4,12); hold on;
for i = 1:length(type)
    bar(i,mean(cat(1, results(i,:).avgRew)));
end
set(gca,'xtick',[1:4],'xticklabel',type)
prettyplot

set(gcf, 'Position',  [300, 300, 800, 500])

%% beta and policy cost
figure; hold on;
subplot 411; hold on;
for i = 1:length(type)
    h(i,:) = shadedErrorBar(1:timeSteps,mean(cat(1, results(i,:).beta)), std(cat(1, results(i,:).beta)),{'LineWidth',1.5,'Color',map(i,:)},1);
    %plot(mean(cat(1, results(i,:).beta)),'LineWidth',1.5);
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel(' \beta')
legend([h.mainLine],[type 'devaluation'],'FontSize',15); legend('boxoff')

subplot 412; hold on;
for i = 1:length(type)
    shadedErrorBar(1:timeSteps,mean(movmean(cat(2, results(i,:).cost),500,'Endpoints','shrink')'),std(movmean(cat(2, results(i,:).cost),500,'Endpoints','shrink')'),{'LineWidth',1.5,'Color',map(i,:)},1);
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel('policy cost C(\pi_\theta)')

subplot 413; hold on;

for i = 1:length(type)
    shadedErrorBar(1:timeSteps,movmean(mean((1./cat(1, results(i,:).beta)).*cat(2, results(i,:).cost)'), 500,'Endpoints','shrink'), movmean(std((1./cat(1, results(i,:).beta)).* cat(2, results(i,:).cost)'), win,'Endpoints','shrink'),{'LineWidth',1.5,'Color',map(i,:)},1);
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel('\beta^{-1} C(\pi_\theta)')

subplot 414; hold on;
for i = 1:length(type)
    h(i,:) = shadedErrorBar(1:timeSteps,mean(cat(1, results(i,:).mi)), std(cat(1, results(i,:).mi)),{'LineWidth',1.5,'Color',map(i,:)},1);
    %plot(mean(cat(1, results(i,:).beta)),'LineWidth',1.5);
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel('I(S;A)')
xlabel('time (s)')
prettyplot

subprettyplot(4,1)

set(gcf, 'Position',  [300, 300, 700, 500])


%% action rate before and after devaluation (avg over many iters)
if sched.devalTime ~= timeSteps
    devalWin = timeSteps-sched.devalTime;
    
    figure; subplot 121; hold on;
    for i = 1:length(type)
        all_a = cat(1,results(i,:).a);
        pre_deval = sum(sum(all_a(:,sched.devalTime-devalWin:sched.devalTime)==2))/(devalWin*size(results,2));
        post_deval =  sum(sum(all_a(:,sched.devalTime+1:end)==2))/(devalWin*size(results,2));
        b = bar(i,[pre_deval post_deval],'FaceColor',map(i,:));
        b(2).FaceColor = [1 1 1];
        b(2).EdgeColor = map(i,:);
        b(2).LineWidth = 2;
        
        results(i).devalScore = pre_deval - post_deval;
    end
    ylabel('action (press) rate')
    legend('before','after','FontSize',15,'Location','NorthWest'); legend('boxoff')
    set(gca,'xtick',[1:4],'xticklabel',type)
    
    
    subplot 122; hold on; % change in action rate
    for i = 1:length(type)
        bar(i,results(i).devalScore,'FaceColor',map(i,:));
    end
    ylabel('devaluation score (pre-post)')
    axis([0 5 0 0.5])
    set(gca,'xtick',[1:4],'xticklabel',type)
    
    subprettyplot(1,2)
    
    set(gcf, 'Position',  [300, 300, 800, 300])
    
    %% beta relationship with deval score
    
    figure(100); subplot 121;hold on;
    for i = 1:length(type)
        plot(mean(results(i).beta(sched.devalTime-win:sched.devalTime)),results(i).devalScore,'.','Color',map(i,:),'MarkerSize',30);
    end
    xlabel('\beta')
    ylabel('devaluation score')
    prettyplot
end


%% relationship between a changing beta and a changing policy complexity

% moving est of mutual information
for i = 1:length(type)
    results(i).movMI = movmean(results(i).mi, win,'Endpoints','shrink');
    results(i).movCost = movmean(results(i).cost,win,'Endpoints','shrink');
    results(i).movRew = movmean(results(i).rho,win,'Endpoints','shrink');
    results(i).movBeta = movmean(results(i).beta,win,'Endpoints','shrink');
    results(i).estBeta = gradient(results(i).movCost)./gradient(results(i).movRew);
end

% what is relationship between changing beta and a changing policy complexity
% does beta predict the policy complexity? --> YES, the bigger the beta,
% the bigger the policy complexity is allowed to be

% if dbeta is 0, v changes w I the same
% if dbeta is big, policy cost changes a lot (there's a lot of room to move) for the same reward

%% estimated mutual information (at end of learning)

% figure; hold on; subplot 131; hold on;
% num = 1000;
% for i = 1:length(type)
%     bar(i,mean(results(i).cost(end-num:end)),'FaceColor',map(i,:));
% end
% set(gca,'xtick',[1:4],'xticklabel',type)
% ylabel('policy cost C(\pi_\theta)')
% prettyplot
% 
% subplot 132; hold on;
% for i = 1:length(type)
%     plot(results(i).movBeta,results(i).movCost,'.','Color',map(i,:),'MarkerSize',20 )
% end
% xlabel('\beta')
% ylabel('C(\pi_\theta)')
% prettyplot
% 
% subplot 133; hold on;
% for i = 1:length(type)
%     plot(results(i).movCost,results(i).movRew,'.','Color',map(i,:),'MarkerSize',20 )
% end
% xlabel('C(\pi_\theta)')
% ylabel('reward')
% prettyplot
% set(gcf, 'Position',  [300, 300, 1000, 300])


%% MI (action = press)

% figure; hold on;
% for i = 1:length(type)
%     bar(i,mean(results(i).mi(end-num:end)),'FaceColor',map(i,:));
% end
% set(gca,'xtick',[1:4],'xticklabel',type)
% ylabel('mutual information I(S;A)')
% prettyplot
% 
% 
% figure; hold on;
% for i = 1:length(type)
%     plot(results(i).mi,'Color',map(i,:));
%     plot(results(i).rho,'Color',map(i,:));
% end
% set(gca,'xtick',[1:4],'xticklabel',type)
% ylabel('mutual information I(S;A)')
% prettyplot


% %% rho and rpe
% figure; hold on; subplot 221; hold on;
% for i = 1:length(type)
%     plot(movmean(results(i).rho,win,'Endpoints','shrink'),'LineWidth',1.5);
% end
%
% ylabel('avg rew')
% prettyplot
%
% subplot 222; hold on;
% for i = 1:length(type)
%     plot(movmean(results(i).rpe,win,'Endpoints','shrink'),'LineWidth',1.5);
% end
% xlabel('time(s)')
% ylabel('\delta')
% prettyplot
%
% subplot 223; hold on;
% for i = 1:length(type)
%     plot(results(i).movCost,results(i).movRew,'.');
% end
%
% subplot 224; hold on;
% for i = 1:length(type)
%     plot(results(i).movMI,results(i).movRew,'.');
% end
%
% %% instantaneous contingency
% figure; hold on; subplot 121; hold on;
% for i = 1:length(type)
%     plot(movmean(results(i).cont,win,'Endpoints','shrink'),'LineWidth',2);
% end
%
% ylabel('contingency')
% xlabel('time (s)')
% prettyplot
%
% % contingency and press rate
% subplot 122; hold on;
% for i = 1:length(type)
%     plot(results(i).movActRate,movmean(results(i).cont,win,'Endpoints','shrink'),'.');
% end
% ylabel('contingency')
% xlabel('action rate \lambda_a')
% prettyplot

figure(100); subplot 122;hold on;

for i = 1:length(type)
    plot(mean(results(i).cost(sched.devalTime-win:sched.devalTime)), mean(results(i).rho(sched.devalTime-win:sched.devalTime)),'.','Color',map(i,:),'MarkerSize',30 )
end
xlabel('policy complexity')
ylabel('average reward')
prettyplot
end


function makePlots(results, sched, type)
%% init color map
map = brewermap(4,'*RdBu');
temp = map(3,:);
map(3,:) = map(4,:);
map(4,:) = temp;  % swap colors so darker ones are fixed
set(0, 'DefaultAxesColorOrder', map) % first three rows

timeSteps = sched.timeSteps;

%% cumulative sum plots
num = 200; % last how many # trials to look at

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

suptitle('policy weights, exp(\theta)')

%% probability of action (actual policy)
for i = 1:length(type)
    ps = (sum(results(i).ps,2)./sum(results(i).ps(:)));
    results(i).PAS = exp(results(i).beta(end)*results(i).theta(2:end,:,end) + log(results(i).paa(end,:)));
    results(i).PAS = results(i).PAS./sum(results(i).PAS,2);
    
    results(i).MI = sum(results(i).PAS.*log(results(i).PAS./results(i).paa(end,:)),2);
    results(i).MIMI = sum(ps.*sum(results(i).PAS.*log(results(i).PAS./results(i).paa(end,:)),2));
end

figure; subplot 121;
colormap(flipud(gray))
imagesc([results(1).PAS(:,2) results(3).PAS(:,2)])
set(gca,'xtick',[1:2],'xticklabel',type([1 3]))
ylabel('state')
subplot 122;
colormap(flipud(gray))
imagesc([results(2).PAS(:,2) results(4).PAS(:,2)])
set(gca,'xtick',[1:2],'xticklabel',type([2 4]))
subprettyplot(1,2)
suptitle('p(a|s) = exp[\beta*\theta + log P(a)]')

figure; hold on; subplot 211; imagesc([results(1).MI results(3).MI results(2).MI results(4).MI])
subplot 212;bar([results(1).MIMI results(3).MIMI results(2).MIMI results(4).MIMI])
%% visualize weights
figure; subplot 121;
colormap(flipud(gray))
imagesc([results(1).w(end,:)' results(3).w(end,:)'])
set(gca,'xtick',[1:2],'xticklabel',type([1 3]))
ylabel('state')
subplot 122;
colormap(flipud(gray))
imagesc([results(2).w(end,:)' results(4).w(end,:)'])
set(gca,'xtick',[1:2],'xticklabel',type([2 4]))
subprettyplot(1,2)


%% outcome and action rates over time
win = 100; % # seconds moving window
for i = 1:length(type)
    results(i).outRate = sum(results(i).x==2)/timeSteps;
    results(i).actRate = sum(results(i).a==2)/timeSteps;
    
    % moving average
    results(i).movOutRate = movmean(results(i).x-1, win,'Endpoints','shrink');
    results(i).movActRate = movmean(results(i).a-1, win,'Endpoints','shrink');
    
    % moving avg reward (includes action cost)
    results(i).avgRew = movmean(results(i).rho, win,'Endpoints','shrink');
    
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

%% beta and policy cost
figure; hold on;
subplot 311; hold on;
for i = 1:length(type)
    plot(results(i).beta,'LineWidth',1.5)
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel(' \beta')
legend([type 'devaluation'],'FontSize',15); legend('boxoff')

subplot 312; hold on;
for i = 1:length(type)
    plot(movmean(results(i).cost,500,'Endpoints','shrink'),'LineWidth',1.5)
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel('policy cost C(\pi_\theta)')

subplot 313; hold on;
for i = 1:length(type)
    plot(movmean((1./results(i).beta).* results(i).cost', win,'Endpoints','shrink'),'LineWidth',1.5)
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel('\beta^{-1} C(\pi_\theta)')

subprettyplot(3,1)

set(gcf, 'Position',  [300, 300, 700, 500])


%% mutual information and log(pa) vs beta(theta*phi)
figure;
subplot 311; hold on;
for i = 1:length(type)
    plot(movmean(results(i).cost, 500),'LineWidth',1.5)
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel('policy cost')
xlabel('time (s)')
prettyplot

subplot 312; hold on;
for i = 1:length(type)
    plot(movmean(results(i).mi, 500),'LineWidth',1.5)
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel('I(S;A)')
xlabel('time (s)')
prettyplot
%
% subplot 312; hold on;
% for i = 1:length(type)
%     plot(movmean(log(results(i).pa), win),'Color',map(i,:),'LineWidth',1.5)
% end
% plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
% ylabel('p(a)')
% xlabel('time (s)')
% title('marginal action probability (of current action)')
%
% subplot 313; hold on;
% for i = 1:length(type)
%     plot(movmean(results(i).betatheta, win),'Color',map(i,:),'LineWidth',1.5)
% end
% plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
% ylabel('exp[\beta(\theta \phi)]')
% xlabel('time (s)')



%% action rate before and after devaluation
if sched.devalTime ~= timeSteps
    devalWin = timeSteps-sched.devalTime;
    
    figure; subplot 121; hold on;
    for i = 1:length(type)
        b = bar(i,[sum(results(i).a(sched.devalTime-devalWin:sched.devalTime)-1)/devalWin sum(results(i).a(sched.devalTime+1:end)-1)/devalWin],'FaceColor',map(i,:));
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
        results(i).devalScore = sum(results(i).a(sched.devalTime-devalWin:sched.devalTime)-1)/devalWin - sum(results(i).a(sched.devalTime+1:end)-1)/devalWin;
        bar(i,results(i).devalScore,'FaceColor',map(i,:));
    end
    ylabel('devaluation score (pre-post)')
    axis([0 5 0 0.5])
    set(gca,'xtick',[1:4],'xticklabel',type)
    
    % beta at time of deval vs deval score
    subprettyplot(1,2)
    
    set(gcf, 'Position',  [300, 300, 800, 300])
    
    %% beta relationship with deval score
    
    figure(100); subplot 121;hold on;
    for i = 1:length(type)
        plot(mean(results(i).beta(sched.devalTime-win:sched.devalTime)),results(i).devalScore,'.','Color',map(i,:),'MarkerSize',30);
    end
    xlabel('\beta')
    ylabel('devaluation score')
    prettyplot
end


%% relationship between a changing beta and a changing policy complexity

% moving est of mutual information
for i = 1:length(type)
    results(i).movMI = movmean(results(i).mi, win,'Endpoints','shrink');
    results(i).movCost = movmean(results(i).cost,win,'Endpoints','shrink');
    results(i).movRew = movmean(results(i).rho,win,'Endpoints','shrink');
    results(i).movBeta = movmean(results(i).beta,win,'Endpoints','shrink');
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
%
% % omg CONFIMED cost and mi are the same
% %figure; hold on;
% %num = 1000;
% %for i = 1:length(type)
% %    bar(i,mean(results(i).mi(end-num:end)),'FaceColor',map(i,:));
% %end
%
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


figure; hold on;
for i = 1:length(type)
    plot(results(i).mi,'Color',map(i,:));
    plot(results(i).rho,'Color',map(i,:));
end
set(gca,'xtick',[1:4],'xticklabel',type)
ylabel('mutual information I(S;A)')
prettyplot


% %% rho and rpe
% figure; hold on; subplot 221; hold on;
% for i = 1:length(type)
%     plot(movmean(results(i).rho,win,'Endpoints','shrink'),'LineWidth',1.5);
% end
%
% ylabel('avg rew')
% prettyplot
%
% subplot 222; hold on;
% for i = 1:length(type)
%     plot(movmean(results(i).rpe,win,'Endpoints','shrink'),'LineWidth',1.5);
% end
% xlabel('time(s)')
% ylabel('\delta')
% prettyplot
%
% subplot 223; hold on;
% for i = 1:length(type)
%     plot(results(i).movCost,results(i).movRew,'.');
% end
%
% subplot 224; hold on;
% for i = 1:length(type)
%     plot(results(i).movMI,results(i).movRew,'.');
% end
%
% %% instantaneous contingency
% figure; hold on; subplot 121; hold on;
% for i = 1:length(type)
%     plot(movmean(results(i).cont,win,'Endpoints','shrink'),'LineWidth',2);
% end
%
% ylabel('contingency')
% xlabel('time (s)')
% prettyplot
%
% % contingency and press rate
% subplot 122; hold on;
% for i = 1:length(type)
%     plot(results(i).movActRate,movmean(results(i).cont,win,'Endpoints','shrink'),'.');
% end
% ylabel('contingency')
% xlabel('action rate \lambda_a')
% prettyplot

figure(300); subplot 122;hold on;

for i = 1:length(type)
    plot(mean(results(i).cost(sched.devalTime-win:sched.devalTime)), mean(results(i).rho(sched.devalTime-win:sched.devalTime)),'.','Color',map(i,:),'MarkerSize',30 )
end
xlabel('policy complexity')
ylabel('average reward')
prettyplot
end
