function habitBetaCurve
% purpose of this is to simulate a curve with different values of beta,
% model 3

clear all
%close all

addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools');                  % various plot tools

% params: [R  I  model timeSteps  beta  cmax  deval]
params =  [20 20 3 5000 0.1 100 1]; % 1800 timesteps = 30 minutes, 3600 = 1hr  % 2:4560, 5:11400  10:22800, 20:45600
aparams = [0.1 0.1 1 0.04];
type = {'FR','VR','FI','VI'};
sched.sessnum = 1;              % number of sessions

% agent parameters
agent.alpha_w = aparams(1);         % value learning rate
agent.alpha_t = aparams(2);         % policy learning rate
agent.alpha_b = aparams(3);         % beta learning rate
agent.acost = aparams(4);           % action cost
agent.alpha_r = 0.001;               % rho learning rate
agent.alpha_e = 0.001;               % policy cost learning rate


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
cmax = params(6);
deval = params(7);


%% using data?
garr = 0;
if garr == 1
    load('all_data_cleaned.mat')
else
    % for VI only
    sched.times = contingency_generateVI(sched.I);   % pre-generated wait times before reward
    % for VR only
    sched.actions = contingency_generateVR(sched.R); % pre-generated number of actions before reward
end

%% initialize
sched.model = model;     % which lesioned model to run
sched.cmax = cmax;       % max complexity (low v high)
sched.type = type;

sched.k = 1;
sched.timeSteps = timeSteps;

% repetitions
numrats = 5;                     % how many iterations/agents to run / avg over
sched.trainEnd = timeSteps;      % timestep where outcome gets devalued
sched.deval = deval;
sched.devalsess = [2 10 20];     % sessions where devaluation will occur
% sched.devalsess = [2 10];      % sessions where devaluation will occur

if sched.deval == 1
    sched.devalWin = 30; % 1 hour of satiation manipulation
    sched.testWin = 300;
    sched.trainEnd = sched.timeSteps;
    sched.devalEnd = sched.trainEnd + sched.devalWin;
end

run = 0;
beta = [0.01 0.1 0.5 1 2 10];

if run == 1                % run simulations for all types
    sched.devalsess = 1;
    for b = 1:length(beta)
        sched.beta = beta(b);
        for t = 1:length(type) % for each iteration
            sched.type = type{t};
            results(b,t) = habitAgent(sched, agent);   % # schedule types x 20 sessions
        end
    end
    save('results_betacurve.mat','results')
else
    load('results_betacurve.mat')
end
makePlots(results, sched, type);


end


function makePlots(results, sched, type)

%% init color map
map = brewermap(4,'*RdBu');
temp = map(3,:);
map(3,:) = map(4,:);
map(4,:) = temp;  % swap colors so darker ones are fixed
set(0, 'DefaultAxesColorOrder', map) % first three rows

diagn = 0;
beta = [0.01 0.1 0.5 1 2 10];
%% loop through betas
% policy cost
figure; hold on;
for b = 1:size(results,1)
    subplot(2,3,b);hold on;
    for i = 1:size(results,2)
        plot([results(b,i).ecost])
        endCost(b,i) = results(b,i).ecost(sched.trainEnd);
        axis([0 5000 0 .3])
       title(strcat('\beta = ',num2str(beta(b))))
    end
end 

 subplot(2,3,4);ylabel('policy cost C(\pi_\theta)')
xlabel('timesteps')
legend(type)
legend('boxoff')
subprettyplot(2,3)

% avg reward
figure; hold on;
for b = 1:size(results,1)
    subplot(2,3,b);hold on;
    for i = 1:size(results,2)
        plot([results(b,i).rho(1:sched.trainEnd)])
        endRho(b,i) = results(b,i).rho(sched.trainEnd);
        title(strcat('\beta = ',num2str(beta(b))))
    end
end 
 subplot(2,3,4);ylabel('avg reward V^\pi')
xlabel('timesteps')
subprettyplot(2,3)

figure;hold on;axis([0 0.25 0 0.07])
for t = 1:4 
    for b = 1:size(results,1)
    plot(endCost(b,t),endRho(b,t),'o','MarkerFaceColor',map(t,:),'MarkerSize',20,'MarkerEdgeColor',[1 1 1]);
    end
end
prettyplot
ylabel('avg reward V^\pi')
xlabel('policy complexity C(\pi_\theta)')

%% make sure actions look correct
% policy cost
figure; hold on;
for b = 1:size(results,1)
    subplot(2,4,b);hold on;
    for i = 1:size(results,2)
        plot([results(b,i).ecost])
        endCost(b,i) = results(b,i).ecost(sched.trainEnd);
        title(strcat('\beta = ',num2str(beta(b))))
        plot([sched.trainEnd sched.trainEnd],ylim,'k--','LineWidth',2)
plot([sched.devalEnd sched.devalEnd],ylim,'k.-','LineWidth',2)
    end
end 
equalabscissa(2,4)
subprettyplot(2,4)

% avg reward
figure; hold on;
for b = 1:size(results,1)
    subplot(2,4,b);hold on;
    for i = 1:size(results,2)
        plot([results(b,i).rho])
        endRho(b,i) = results(b,i).rho(sched.trainEnd);
        title(strcat('\beta = ',num2str(beta(b))))
        plot([sched.trainEnd sched.trainEnd],ylim,'k--','LineWidth',2)
        plot([sched.devalEnd sched.devalEnd],ylim,'k.-','LineWidth',2)
    end
end 
equalabscissa(2,4)
subprettyplot(2,4)


%% probability of action (learned policy weights BEFORE devaluation)
if sched.deval == 1
    
    %% beta duration satiation
    % from trainEnd to devalEnd
    figure; hold on; subplot 231; hold on;
    for i = 1:length(type)
        plot(results(i).rpe(sched.trainEnd+1:sched.devalEnd-1))
    end
    prettyplot
    ylabel('\delta_t RPE')
    xlabel('satiation period (s)')
    
    subplot 232; hold on;
    for i = 1:length(type)
        plot(results(i).r(sched.trainEnd+1:sched.devalEnd-1)-results(i).rho(sched.trainEnd+1:sched.devalEnd-1))
    end
    prettyplot
    ylabel('r-\rho')
    xlabel('satiation period (s)')
    
    subplot 233; hold on;
    for i = 1:length(type)
        plot((1./results(i).beta(sched.trainEnd+1:sched.devalEnd-1)).*results(i).ecost(sched.trainEnd+1:sched.devalEnd-1))
    end
    prettyplot
    ylabel('\beta^{-1}*E[C(\pi)]')
    xlabel('satiation period (s)')
    % plot the RPEs before and after devaluation to show that
    % in FR/FI, since beta is high, the (r - 1/beta * C) should be higher and
    % should increase theta(not press) more than in VR/VI where beta is low, so
    % 1/beta * C is big a
    
    % need 1/beta*C to also be higher in VR/VI > FR/FI
    
    % also, the longer you train, the more policy weights increase, more beta
    % increases..
    
    % from devalEnd to end
    subplot 234; hold on;
    for i = 1:length(type)
        plot(results(i).rpe(sched.devalEnd:end))
    end
    prettyplot
    ylabel('\delta_t RPE')
    xlabel('test period (s)')
    
    subplot 235; hold on;
    for i = 1:length(type)
        plot(results(i).r(sched.devalEnd:end)-results(i).rho(sched.devalEnd:end))
    end
    prettyplot
    ylabel('r-\rho')
    xlabel('test period (s)')
    
    subplot 236; hold on;
    for i = 1:length(type)
        plot((1./results(i).beta(sched.devalEnd:end)).*results(i).ecost(sched.devalEnd:end))
    end
    prettyplot
    ylabel('\beta^{-1}*E[C(\pi)]')
    xlabel('test period (s)')
    
else
    %% probability of action (learned policy weights)
    if length(type) == 4 % if doing all 4 schedules at once
        figure; subplot 121;
        colormap(flipud(gray))
        imagesc([results(1).pas(:,2,end) results(2).pas(:,2,end)])
        caxis([0 1])
        set(gca,'xtick',[1:2],'xticklabel',type([1 2]))
        ylabel('state feature')
        subplot 122;
        colormap(flipud(gray))
        imagesc([results(3).pas(:,2,end) results(4).pas(:,2,end)])
        caxis([0 1])
        set(gca,'xtick',[1:2],'xticklabel',type([3 4]))
        subprettyplot(1,2)
        
        suptitle('policy weights, exp(\theta)')
    else
        figure; hold on;
        for i = 1:length(type)
            subplot(1,length(type),i);
            colormap(flipud(gray))
            imagesc(results(i).pas(:,2,end))
            caxis([0 1])
            set(gca,'xtick',1,'xticklabel',type{i})
            ylabel('state feature');
            prettyplot
        end
        suptitle('policy weights, exp(\theta)')
    end
end

%% outcome and action rates over time
win = 100; % # seconds moving window
for b = 1:length(beta)
for i = 1:length(type)
    results(b,i).outRate = sum(results(i).x(1:sched.trainEnd)==2)/sched.trainEnd;
    results(b,i).actRate = sum(results(i).a(1:sched.trainEnd)==2)/sched.trainEnd;
    
    % moving average
    results(b,i).movOutRate = movmean(results(i).x-1, win,'Endpoints','shrink');
    results(b,i).movActRate = movmean(results(i).a-1, win,'Endpoints','shrink');
    
    % moving avg reward (includes action cost)
    results(b,i).avgRew = movmean(results(i).rho, win,'Endpoints','shrink');
end
end
figure; hold on;
subplot(3,4,1:3); hold on;
for b = 1:length(beta)
for i = 1:length(type)
    plot(results(b,i).movOutRate,'LineWidth',1.5)
end
end
plot([sched.trainEnd sched.trainEnd],ylim,'k--','LineWidth',2)
plot([sched.devalEnd sched.devalEnd],ylim,'k.-','LineWidth',2)
prettyplot
ylabel('outcome rate')
legend([type 'devaluation' 'test'],'FontSize',15); legend('boxoff')

subplot(3,4,5:7); hold on;
for b = 1:length(beta)
for i = 1:length(type)
    plot(results(b,i).movActRate,'LineWidth',1.5)
end
end
xlabel('time (s)')
ylabel('action rate')
plot([sched.trainEnd sched.trainEnd],ylim,'k--','LineWidth',2)
plot([sched.devalEnd sched.devalEnd],ylim,'k.-','LineWidth',2)
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
plot([sched.trainEnd sched.trainEnd],ylim,'k--','LineWidth',2)
plot([sched.devalEnd sched.devalEnd],ylim,'k.-','LineWidth',2)
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
plot([sched.trainEnd sched.trainEnd],ylim,'k--','LineWidth',2)
plot([sched.devalEnd sched.devalEnd],ylim,'k.-','LineWidth',2)
ylabel(' \beta')
legend([type 'devaluation' 'test'],'FontSize',15); legend('boxoff')

subplot 312; hold on;
for i = 1:length(type)
    %plot(movmean(results(i).cost,500,'Endpoints','shrink'),'LineWidth',1.5)
    plot(results(i).ecost,'LineWidth',1.5)
end
plot([sched.trainEnd sched.trainEnd],ylim,'k--','LineWidth',2)
plot([sched.devalEnd sched.devalEnd],ylim,'k.-','LineWidth',2)
ylabel('avg policy cost E[C(\pi_\theta)]')

subplot 313; hold on;
for i = 1:length(type)
    plot((1./results(i).beta).*results(i).ecost,'LineWidth',1.5)
end
plot([sched.trainEnd sched.trainEnd],ylim,'k--','LineWidth',2)
plot([sched.devalEnd sched.devalEnd],ylim,'k.-','LineWidth',2)
ylabel('\beta^{-1} C(\pi_\theta)')
xlabel('time (s)')

subprettyplot(3,1)

set(gcf, 'Position',  [300, 300, 700, 500])

%% action rate before and after devaluation
if sched.deval==1
    
    figure; subplot 121; hold on;
    for i = 1:length(type)
        preDevalRate(i) = sum(results(i).a(sched.trainEnd-sched.testWin:sched.trainEnd)==2)/sched.testWin;
        postDevalRate(i) = sum(results(i).a(sched.devalEnd+1:end)==2)/sched.testWin;
        b = bar(i,[preDevalRate(i) postDevalRate(i)],'FaceColor',map(i,:));
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
        results(i).devalScore = preDevalRate(i)-postDevalRate(i);
        bar(i,results(i).devalScore,'FaceColor',map(i,:));
    end
    ylabel('devaluation score (pre-post)')
    axis([0 5 0 0.5])
    set(gca,'xtick',[1:4],'xticklabel',type)
    
    % beta at time of deval vs deval score
    subprettyplot(1,2)
    set(gcf, 'Position',  [300, 300, 800, 300])
    
    %% beta relationship with deval score
    figure; hold on;
    for i = 1:length(type)
        plot(mean(results(i).beta(sched.trainEnd-sched.testWin:sched.trainEnd)),results(i).devalScore,'.','Color',map(i,:),'MarkerSize',30);
    end
    xlabel('\beta')
    ylabel('devaluation score')
    prettyplot
end

%% plot cost and beta at end of learning
figure; hold on;
for i = 1:length(type)
    plot(results(i).ecost(sched.trainEnd),results(i).rho(sched.trainEnd),'.','Color',map(i,:),'MarkerSize',30);
end
xlabel('I^\pi')
ylabel('V^\pi')
prettyplot
end
