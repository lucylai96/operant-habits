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
%               beta - tradeoff parameter, only relevant for model 3 & 4 (e.g., 0-10)
%               cmax - max capacity / upperbound, only relevant for model 4 (e.g., 0-100)
%               deval - whether devaluation is simulated or not (standard
%                       protocol is 5 minutes of "test," or 300 timesteps/seconds)
%               ac - action cost (e.g., 0-1)
% NOTES:
%
% Written by Lucy Lai (May 2020)

clear all
close all

addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools');                  % various plot tools

% params: [R  I  model timeSteps  beta  cmax  deval]
if nargin <1
    params =  [20 20 4 3000 1 100 0]; % 1800 timesteps = 30 minutes, 3600 = 1hr
    aparams = [0.1 0.1 0.01 1 0.05];
    % aparams = [1 1 1 1 1];
    type = {'VR','VI'};
    type = {'FR','VR','FI','VI'};
    % type = 'VI';
end

% agent parameters
agent.alpha_w = aparams(1);         % value learning rate
agent.alpha_t = aparams(2);         % policy learning rate
agent.alpha_r = aparams(3);         % rho learning rate
agent.alpha_b = aparams(4);         % beta learning rate
agent.acost = aparams(5);           % act

% FR-20: 30 daily sessions, sessions terminated when 50 rewards were earned or 1 hour elapsed. Satiety devaluation test after sessions 2, 10, 20, and 30.
% VR-20: same as above
% FI-45: 20 daily sessions (lasted a fixed 38 minutes). Satiety devaluation test after sessions 2, 10, and 20.
% VI-45: 20 daily sessions and tested after sessions 2, 10, and 20.

% FR/VR 50 rewards OR 60 minutes = 3600 timesteps x 30 sessions
% FI/VI 38 minutes = 2280 timesteps x 20 sessions

% need code that intakes number of sessions

%% unpack parameters
sched.R = params(1);
sched.I = params(2);
model = params(3);
timeSteps = params(4);
beta = params(5);
cmax = params(6);
deval = params(7);


%% using data?
garr = 0;
if garr == 1
    %load('all_data_cleaned.mat')
    %sched.times =
    %sched.actions =
else
    % for VI only
    sched.times = contingency_generateVI(sched.I);   % pre-generated wait times before reward
    % for VR only
    sched.actions = contingency_generateVR(sched.R); % pre-generated number of actions before reward
end


%% initialize
sched.model = model;     % which lesioned model to run
sched.beta = beta;       % starting beta; high beta = low cost. beta should increase for high contingency
sched.cmax = cmax;       % max complexity (low v high)
sched.type = type;

sched.k = 1;
sched.timeSteps = timeSteps;

% repetitions
sched.sessnum = 1;               % number of sessions
numsub = 5;                       % how many iterations/agents to run / avg over
sched.devalTime = timeSteps;      % timestep where outcome gets devalued
sched.setrew = 0;                 % ONLY FR/VR - are we setting # of rewards that need to be achieved? 0 for no, 1 for until 50 rewards reached
sched.deval = 1;
sched.devalsess = [2 10 20];  % sessions where devaluation will occur
% sched.devalsess = [2 10];  % sessions where devaluation will occur

run = 1;

if sched.sessnum > 1                 % multiple sessions
    % run simulations
    if run == 1
        for i = 1:numsub % for each iteration
            results(i).sess = habitAgent(sched, agent); % # schedule types x 20 sessions
        end
        save('results.mat','results')
    else
        load('results.mat')
    end
    
    makePlotsSess(results, sched,type)
else
    
    % run simulations
    if run == 1
        for i = 1:length(type)% for each iteration
            sched.type = type{i};
            results(i,:) = habitAgent(sched, agent); % # schedule types x 20 sessions
        end
        save('results_1.mat','results')
    else
        load('results_1.mat')
    end
    
    makePlots(results, sched, type)
end

end
function makePlots(results, sched, type)
diagn = 0;
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
if length(type)==4 % if doing all 4 schedules at once
    figure; subplot 121;
    colormap(flipud(gray))
    imagesc([results(1).Pa(:,2,end) results(2).Pa(:,2,end)])
    set(gca,'xtick',[1:2],'xticklabel',type([1 2]))
    ylabel('state feature')
    subplot 122;
    colormap(flipud(gray))
    imagesc([results(3).Pa(:,2,end) results(4).Pa(:,2,end)])
    set(gca,'xtick',[1:2],'xticklabel',type([3 4]))
    subprettyplot(1,2)
    
    suptitle('policy weights, exp(\theta)')
else
    figure; hold on;
    for i = 1:length(type)
        subplot(1,length(type),i);
        colormap(flipud(gray))
        imagesc(results(i).Pa(:,2,end))
        set(gca,'xtick',1,'xticklabel',type{i})
        ylabel('state feature');
        prettyplot
    end
    suptitle('policy weights, exp(\theta)')
end


%% probability of action (actual policy)
for i = 1:length(type)
    ps = (sum(results(i).ps,2)./sum(results(i).ps(:)));
    results(i).PAS = exp(results(i).theta(:,:,end) + log(results(i).paa(end,:)));
    results(i).PAS = results(i).PAS./sum(results(i).PAS,2);
    
    results(i).MI = sum(results(i).PAS.*log(results(i).PAS./results(i).paa(end,:)),2);
    results(i).MIMI = sum(ps.*sum(results(i).PAS.*log(results(i).PAS./results(i).paa(end,:)),2));
end

if length(type)==4 % if doing all 4 schedules at once
    figure; subplot 121;
    colormap(flipud(gray))
    imagesc([results(1).PAS(:,2) results(2).PAS(:,2)])
    set(gca,'xtick',[1:2],'xticklabel',type([1 2]))
    ylabel('state feature')
    subplot 122;
    colormap(flipud(gray))
    imagesc([results(3).PAS(:,2) results(4).PAS(:,2)])
    set(gca,'xtick',[1:2],'xticklabel',type([3 4]))
    subprettyplot(1,2)
    suptitle('p(a|s) = exp[\theta + log P(a)]')
else
    figure; hold on;
    for i = 1:length(type)
        subplot(1,length(type),i);
        colormap(flipud(gray))
        imagesc(results(i).PAS(:,2))
        set(gca,'xtick',1,'xticklabel',type{i})
        ylabel('state feature');
        prettyplot
    end
    suptitle('p(a|s) = exp[\theta + log P(a)]')
end

%% visualize weights
% figure; subplot 121;
% colormap(flipud(gray))
% imagesc([results(1).w(end,:)' results(3).w(end,:)'])
% set(gca,'xtick',[1:2],'xticklabel',type([1 3]))
% ylabel('state feature')
% subplot 122;
% colormap(flipud(gray))
% imagesc([results(2).w(end,:)' results(4).w(end,:)'])
% set(gca,'xtick',[1:2],'xticklabel',type([2 4]))
% title('state feature value weights')
% subprettyplot(1,2)


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


%% diagnostics
if diagn ==1
    for i = 1:length(type)
        figure; subplot 421; histogram(results(i).NA.rew,50); ylabel('rew'); title('# actions');subplot 422; histogram(results(i).NT.rew,50);title('# seconds');
        subplot 423; histogram(results(i).NA.nonrew,50); ylabel('nonrew'); subplot 424; histogram(results(i).NT.nonrew,50);
        % how is theta being updated? plot rpe vs rewarded and non rewarded nt
        subplot 425; hold on; plot(results(i).NA.nonrew,results(i).RPE.nonrew,'r.'); plot(results(i).NA.rew,results(i).RPE.rew,'g.'); ylabel('RPE \delta')
        subplot 426; hold on; plot(results(i).NT.nonrew,results(i).RPE.nonrew,'r.'); plot(results(i).NT.rew,results(i).RPE.rew,'g.');
        subplot 427; hold on; plot(results(i).VS,results(i).RPE.nonrew,'r.'); plot(results(i).VS,results(i).RPE.rew,'g.');title('V(s`)-V(s)');
        subplot 428; hold on; plot((1./results(i).beta).*results(i).cost,results(i).RPE.nonrew,'r.'); plot((1./results(i).beta).*results(i).cost,results(i).RPE.rew,'g.'); title('\beta^{-1} C(\pi)');
        subprettyplot(4,2)
        suptitle(type{i})
        
    end
end


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
    plot(movmean(results(i).mi, 500),'LineWidth',1.5)
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel('I(S;A)')
xlabel('time (s)')
prettyplot

% for i = 1:length(type)
%     plot(movmean((1./results(i).beta).* results(i).cost', win,'Endpoints','shrink'),'LineWidth',1.5)
% end
% plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
% ylabel('\beta^{-1} C(\pi_\theta)')

subprettyplot(3,1)

set(gcf, 'Position',  [300, 300, 700, 500])


%% mutual information and log(pa) vs beta(theta*phi)
% figure;
% subplot 311; hold on;
% for i = 1:length(type)
%     plot(movmean(results(i).cost, 500),'LineWidth',1.5)
% end
% plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
% ylabel('policy cost')
% xlabel('time (s)')
% prettyplot
%
% subplot 312; hold on;
% for i = 1:length(type)
%     plot(movmean(results(i).mi, 500),'LineWidth',1.5)
% end
% plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
% ylabel('I(S;A)')
% xlabel('time (s)')
% prettyplot

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
%
% figure(300); subplot 122;hold on;
%
% for i = 1:length(type)
%     plot(mean(results(i).cost(sched.devalTime-win:sched.devalTime)), mean(results(i).rho(sched.devalTime-win:sched.devalTime)),'.','Color',map(i,:),'MarkerSize',30 )
% end
% xlabel('policy complexity')
% ylabel('average reward')
% prettyplot
end



function makePlotsSess(results, sched, type)
dvlen = 300; % length of devaluation session
map = habitColors;


%% data
load('all_data_cleaned.mat') % actual data
FI45.test = FI45.test./60;
VI45.test = VI45.test./60;

%% outcome and action rates over time
win = 200; % # seconds moving window
sess = size(results(1).sess,2); % get number of sessions
for i = 1:length(results) % for each subject
    for s = 1:sess % for each session
        if ismember(s,sched.devalsess)
            results(i).sess(s).timeSteps = size(results(i).sess(s).a,2); % session length, timesteps;
            timeSteps = results(i).sess(s).timeSteps;
            devalTime = results(i).sess(s).timeSteps - dvlen;
            results(i).sess(s).outRate = sum(results(i).sess(s).x(1:devalTime)==2)/devalTime;
            results(i).sess(s).actRate = sum(results(i).sess(s).a(1:devalTime)==2)/devalTime;
            results(i).sess(s).avgRew = mean(results(i).sess(s).rho(1:devalTime));
            results(i).sess(s).beta = mean(results(i).sess(s).beta(1:devalTime));
            results(i).sess(s).postTestRate = sum(results(i).sess(s).a(devalTime:end)==2)/dvlen;
            results(i).sess(s).preTestRate = sum(results(i).sess(s).a(devalTime-dvlen:devalTime)==2)/dvlen;
            
        else
            results(i).sess(s).timeSteps = size(results(i).sess(s).a,2); % session length, timesteps;
            timeSteps = results(i).sess(s).timeSteps;
            results(i).sess(s).outRate = sum(results(i).sess(s).x==2)/timeSteps;
            results(i).sess(s).actRate = sum(results(i).sess(s).a==2)/timeSteps;
            results(i).sess(s).avgRew = mean(results(i).sess(s).rho);
            results(i).sess(s).beta = mean(results(i).sess(s).beta);
            
        end
        
    end
end

% if it's a devaluation session, separate that data out into results.test
figure; hold on;
subplot 311; hold on;
for i = 1:length(results)
    outRate(i,:) = cat(2,results(i).sess.actRate);
end
h(1,:) = errorbar(mean(outRate),sem(outRate,1),'LineWidth',2,'Color',map(4,:));
prettyplot
ylabel('outcome rate')
legend([type],'FontSize',15); legend('boxoff')
axis([0 22 0 1])
plot([2.5 10.5 20.5;2.5 10.5 20.5],[ylim' ylim' ylim'],'k--') %

subplot 312; hold on;
for i = 1:length(results)
    actRate(i,:) = cat(2,results(i).sess.actRate);
end
h(1,:) = errorbar(mean(actRate),sem(actRate,1),'LineWidth',2,'Color',map(4,:));
%errorbar(mean(FI45.actRate),sem(FI45.actRate,1),'LineWidth',2,'Color',[0 0 0]);
errorbar(mean(VI45.actRate),sem(VI45.actRate,1),'LineWidth',2,'Color',[0 0 0]); % data RI
xlabel('session #')
ylabel('action rate')
axis([0 22 0 1])
plot([2.5 10.5 20.5;2.5 10.5 20.5],[ylim' ylim' ylim'],'k--')
prettyplot

subplot 313;  hold on;
for i = 1:length(results)
    beta(i,:) = cat(2,results(i).sess.beta);
end
h(1,:) = errorbar(mean(beta),sem(beta,1),'LineWidth',2,'Color',map(4,:));
axis([0 22 ylim])
xlabel('session #')
ylabel('\beta')
prettyplot

set(gcf, 'Position',  [300, 300, 800, 500])

for d = 1:3
    for i = 1:length(results)
        test(i,:,d) = [results(i).sess([sched.devalsess(d)]).preTestRate results(i).sess([sched.devalsess(d)]).postTestRate]
        %deval(i,:,d) = cat(2,results(i).sess([sched.devalsess(d)]).preTestRate);
        %val(i,:,d) = cat(2,results(i).sess([sched.devalsess(d)]).postTestRate);
    end
end
figure; hold on;
for d = 1:3
    subplot 121; hold on;
    [b e] = barwitherr(sem(VI45.test(:,:,d),1),d, mean(VI45.test(:,:,d)),'FaceColor',[0 0 0]);
    b(2).FaceColor = [1 1 1];
    b(2).EdgeColor = [0 0 0];
    b(2).LineWidth = 2;
    e.Color = [0 0 0];
    e.LineWidth = 2;
    
    title('data')
    
    subplot 122; hold on;
    [b e] = barwitherr(sem(test(:,:,d),1),d, mean(test(:,:,d)),'FaceColor',map(4,:));
    b(2).FaceColor = [1 1 1];
    b(2).EdgeColor = map(4,:);
    b(2).LineWidth = 2;
    e.Color = map(4,:);
    e.LineWidth = 2;
    
    title('model')
end
subprettyplot(1,2)


for i = 1:length(results)
    mR = [];
    for s = 1:sess
        mR = [mR movmean(results(i).sess(s).a(1:2290)==2,200)];
    end
    moveAR(i,:) = mR;
    clear mR
end
figure; plot(moveAR')



%% beta and policy cost
figure; hold on;
subplot 211; hold on;
for i = 1:length(type)
    %h(i,:) = shadedErrorBar(1:timeSteps,mean(cat(1, results(i,:).beta)), std(cat(1, results(i,:).beta)),{'LineWidth',1.5,'Color',map(i,:)},1);
    plot(movmean(cat(2, results(i,:).beta),500,'Endpoints','shrink'),'LineWidth',1.5);
end
%plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel(' \beta')
legend([type 'devaluation'],'FontSize',15); legend('boxoff')

subplot 212; hold on;
for i = 1:length(type)
    %shadedErrorBar(1:timeSteps,mean(movmean(cat(2,results(i,:).cost),500,'Endpoints','shrink')'),std(movmean(cat(2, results(i,:).cost),500,'Endpoints','shrink')'),{'LineWidth',1.5,'Color',map(i,:)},1);
    plot(movmean(cat(2,results(i,:).cost),500,'Endpoints','shrink'),'LineWidth',1.5);
end
%plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel('policy cost C(\pi_\theta)')

subprettyplot(2,1)
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

% figure(100); subplot 122;hold on;
%
% for i = 1:length(type)
%     plot(mean(results(i).cost(sched.devalTime-win:sched.devalTime)), mean(results(i).rho(sched.devalTime-win:sched.devalTime)),'.','Color',map(i,:),'MarkerSize',30 )
% end
% xlabel('policy complexity')
% ylabel('average reward')
% prettyplot
end
