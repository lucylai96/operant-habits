function [results] = habitDriverOld
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

% FR-20: 30 daily sessions, sessions terminated when 50 rewards were earned or 1 hour elapsed. Satiety devaluation test after sessions 2, 10, 20, and 30.
% VR-20: same as above
% FI-45: 20 daily sessions (lasted a fixed 38 minutes). Satiety devaluation test after sessions 2, 10, and 20.
% VI-45: 20 daily sessions and tested after sessions 2, 10, and 20.

% FR/VR 50 rewards OR 60 minutes = 3600 timesteps x 30 sessions
% FI/VI 38 minutes = 2280 timesteps x 20 sessions

clear all
close all

addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools');  % various plot tools
map = habitColors;

run = 1;

% agent parameters
agent.C = 0.5;                 % max complexity
agent.beta = 0.1;              % starting beta; high beta = low cost. beta should increase for high contingency
agent.lrate_w = 0.001;         % value learning rate
agent.lrate_theta = 0.001;     % policy learning rate
agent.lrate_p = 0.005;         % marginal policy learning rate
agent.lrate_b = 0.001;         % beta learning rate
agent.lrate_r = 0.001;         % rho learning rate
agent.lrate_e = 0.001;         % policy cost learning rate
agent.acost = 0.01;            % action cost

% schedule parameters
type = {'FR','VR','FI','VI'};
sched.R = 20;
sched.I = 45;
sched.sessnum = 1;             % number of sessions
model = 4;
timeSteps = 10000;   % 1800 timesteps = 30 minutes, 3600 = 1hr  % sess2:4560, sess5:11400  sess10:22800, sesss20:456000
deval = 0;

%% using data?
garr = 0;
if garr == 1
    load('all_data_cleaned.mat')
    sched.times = cat(1,VI45n(1).session(:).times)';
    sched.actions = cat(2,VR20n(1).session(:).actions);
else
    % for VI only
    sched.times = contingency_generateVI(sched.I);   % pre-generated wait times before reward
    % for VR only
    sched.actions = contingency_generateVR(sched.R); % pre-generated number of actions before reward
end


%% initialize
sched.model = model;     % which lesioned model to run
sched.type = type;
sched.k = 1;
sched.timeSteps = timeSteps;

% repetitions
numrats = 1;                     % how many iterations/agents to run / avg over
sched.trainEnd = timeSteps;      % timestep where outcome gets devalued
sched.deval = deval;
sched.devalsess = [2 10 20];     % sessions where devaluation will occur
% sched.devalsess = [2 10];      % sessions where devaluation will occur

if sched.deval == 1
    sched.devalWin = 20; % 1 hour of satiation manipulation
    sched.testWin = 300;
    sched.trainEnd = sched.timeSteps;
    sched.devalEnd = sched.trainEnd + sched.devalWin;
else
    sched.trainEnd = sched.timeSteps;
    sched.devalEnd = sched.trainEnd;
end

if sched.sessnum > 1       % multiple sessions
    for t = 1:length(type) % for each iteration
        sched.type = type{t};
        if run == 1            % run simulations for 1 type at a time
            for i = 1:numrats  % for each iteration
                results(i).sess = habitAgent(sched, agent); % # schedule types x 20 sessions
            end
            save(strcat('results',sched.type,'.mat'),'results')
        else
            load(strcat('results',sched.type,'.mat'),'results')
        end
        makePlotsSess(results, sched,type)
    end
    
else
    if run == 1                % run simulations for all types
        sched.devalsess = 1;
        for t = 1:length(type) % for each iteration
            sched.type = type{t};
            if garr ==1
                if sched.type == 'FR'
                    data = FR20n(1);
                elseif sched.type == 'VR'
                    data = VR20n(1);
                elseif sched.type == 'FI'
                    data = FI45n(1);
                elseif sched.type == 'VI'
                    data = VI45n(1);
                end
                input.data = data;
                results(t,:) = habitAgent(sched, agent, input.data); % # schedule types x 20 sessions
            else
                results(t,:) = habitAgent(sched, agent); % # schedule types x 20 sessions
                
                
            end
        end % type
        
        %% plot R-C
        figure; hold on;
        for i = 1:4
            %plot(results(i).mi,results(i).avgr,'.','Color',map(i,:),'MarkerSize',20); hold on;
            %plot(results(i).mi(end),results(i).avgr(end),'ko','MarkerSize',20,'MarkerFaceColor',map(i,:))
            %figure(101); hold on;
            plot(results(i).ecost,results(i).avgr,'.','Color',map(i,:),'MarkerSize',20); hold on;
            h(i,:) = plot(results(i).ecost(end),results(i).avgr(end),'ko','MarkerSize',20,'MarkerFaceColor',map(i,:))
        end
        xlabel('Policy complexity')
        ylabel('Average reward')
        l = legend(h,type);
        title(strcat('Arming parameter:',num2str(sched.R)))
        title(l,'Schedule')
        prettyplot(20)
        axis([0 0.01 0 0.1])
         save('results_all.mat','results')
    else
        load('results_all.mat')
    end
    makePlots(results, sched, type)
end

end
function makePlots(results, sched, type)

%% init color map
map = brewermap(4,'*RdBu');
temp = map(3,:);
map(3,:) = map(4,:);
map(4,:) = temp;  % swap colors so darker ones are fixed
set(0, 'DefaultAxesColorOrder', map) % first three rows

diagn = 0;

%% cumulative sum plots (Fig 1)
num = 4000; % last how many # trials to look at

figure; hold on;
for i = 1:length(type)
    results(i).cs = cumsum(results(i).a(sched.trainEnd-num:sched.trainEnd)==2);
    if i ==2
        h(i,:) = plot(results(i).cs+5,'Color',map(i,:),'LineWidth',2);
        xplot = find(results(i).x(sched.trainEnd-num:sched.trainEnd)==2);
        %line([xplot' xplot'+10]',[results(i).cs(xplot)' results(i).cs(xplot)']'+5,'LineWidth',2,'Color','k');
    else
        h(i,:) = plot(results(i).cs,'Color',map(i,:),'LineWidth',2);
        xplot = find(results(i).x(sched.trainEnd-num:sched.trainEnd)==2);
        %line([xplot' xplot'+10]',[results(i).cs(xplot)' results(i).cs(xplot)']','LineWidth',2,'Color','k');
    end
    
end

legend(h,type);
legend('boxoff')
xlabel('time (s)')
ylabel('cumulative # actions')
prettyplot

%% outcome and action rates over time (Fig 2)
win = 100; % # seconds moving window
for i = 1:length(type)
    results(i).outRate = sum(results(i).x(1:sched.trainEnd)==2)/sched.trainEnd;
    results(i).actRate = sum(results(i).a(1:sched.trainEnd)==2)/sched.trainEnd;
    results(i).value = mean(results(i).rho2(1:sched.trainEnd));
    
    % moving average
    results(i).movOutRate = movmean(results(i).x-1, win,'Endpoints','shrink');
    results(i).movActRate = movmean(results(i).a-1, win,'Endpoints','shrink');
    
    % moving avg reward (includes action cost)
    results(i).avgRew = movmean(results(i).rho2, win,'Endpoints','shrink');
    
end

figure; hold on;
subplot(6,4,1:3); hold on;
for i = 1:length(type)
    plot(results(i).movOutRate(1:sched.trainEnd-win),'LineWidth',1.5)
end
prettyplot
ylabel('outcome rate')
legend([type 'devaluation' 'test'],'FontSize',15); legend('boxoff')

subplot(6,4,5:7); hold on;
for i = 1:length(type)
    plot(results(i).movActRate(1:sched.trainEnd-win),'LineWidth',1.5)
end
ylabel('action rate')
prettyplot

subplot(6,4,4); hold on;
for i = 1:length(type)
    bar(i,results(i).outRate);
end
set(gca,'xtick',[1:4],'xticklabel',type)
prettyplot

subplot(6,4,8); hold on;
for i = 1:length(type)
    bar(i,results(i).actRate);
end
set(gca,'xtick',[1:4],'xticklabel',type)
prettyplot

subplot(6,4,9:11); hold on;
for i = 1:length(type)
    plot(results(i).avgRew(1:sched.trainEnd-win),'LineWidth',1.5)
end
%plot([sched.trainEnd sched.trainEnd],ylim,'k--','LineWidth',2)
%plot([sched.devalEnd sched.devalEnd],ylim,'k.-','LineWidth',2)
ylabel('V^\pi')
prettyplot

subplot(6,4,12); hold on;
for i = 1:length(type)
    bar(i,results(i).value);
end
set(gca,'xtick',[1:4],'xticklabel',type)
prettyplot

%% beta and policy cost (Fig 2)
subplot(6,4,13:15); hold on;
for i = 1:length(type)
    plot(results(i).beta(1:sched.trainEnd-win),'LineWidth',1.5)
end
ylabel(' \beta')
prettyplot

subplot(6,4,17:19); hold on;
for i = 1:length(type)
    plot(results(i).ecost(1:sched.trainEnd-win),'LineWidth',1.5)
end
ylabel('E[C(\pi_\theta)]')
prettyplot

subplot(6,4,21:23);  hold on;
for i = 1:length(type)
    plot((1./results(i).beta(1:sched.trainEnd-win)).*results(i).ecost(1:sched.trainEnd-win),'LineWidth',1.5)
end
ylabel('\beta^{-1} E[C(\pi_\theta)]')
xlabel('time (s)')
prettyplot

set(gcf, 'Position',  [100, 100, 800, 1000])

%% decompose policy (Fig 3)
% if i were to respond only based on the habit component vs the other

figure; hold on;
for i = 1:length(type)
    subplot(4,1,i); hold on;
    value_pi = 1./(1+exp(-(results(i).val(:,2) - results(i).val(:,1)))); plot(value_pi ,'b','LineWidth',2); % state-dep
    plot(results(i).pi_as(:,2),'k','LineWidth',2); % actual policy
    habit_pi = 1./(1+exp(-(results(i).hab(:,2) - results(i).hab(:,1)))); plot(habit_pi ,'r','LineWidth',2); % habit
    
    plot(results(i).pa(:,2) ,'g','LineWidth',2); % pa
    title(type{i})
end
ylabel('p(a=lever|s)')
xlabel('time (s)')
legend('state-dep. component','\pi(a|s)','habit component','pa')
equalabscissa(4,1)
subprettyplot(4,1)
set(gcf, 'Position',  [300, 300, 600, 800])

why
%% RPE deconstructed
% figure; hold on;
% subplot 411; hold on; % rpe
% for i = 1:length(type)
%     plot(results(i).rpe(1:sched.trainEnd-win),'LineWidth',1.5)
% end
% ylabel('\delta')
% prettyplot
%
% subplot 412; hold on; % r-rho
% for i = 1:length(type)
%     plot(results(i).r(1:sched.trainEnd-win)-results(i).rho(1:sched.trainEnd-win),'LineWidth',1.5)
% end
% ylabel('r-\rho')
% prettyplot
%
% subplot 413; hold on; % V(s')-V(s)
% for i = 1:length(type)
%     plot(results(i).VS(1:sched.trainEnd-win),'LineWidth',1.5)
% end
% ylabel('V(s`)-V(s)')
% prettyplot
%
% subplot 414; hold on; % 1\beta*cost
% for i = 1:length(type)
%     plot((1./results(i).beta(1:sched.trainEnd-win)).*results(i).ecost(1:sched.trainEnd-win),'LineWidth',1.5)
% end
% ylabel('\beta^{-1} E[C(\pi_\theta)]')
% xlabel('time (s)')
% prettyplot
%
% set(gcf, 'Position',  [100, 100, 800, 1000])


%% plot pas over time

% for i = 1:length(type)
%     figure; hold on;
%     subplot 311; hold on;
%     imagesc(squeeze(results(i).pas(:,2,:)))
%     colormap(flipud(gray))
%     set(gca,'YDir','reverse')
%     subplot 312; hold on;
%     plot(results(i).rpe)
%     subplot 313; hold on;
%     imagesc(results(i).w')
% end


%% action rate before and after devaluation (Fig 4)
if sched.deval==1
    
    figure; subplot 131; hold on;
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
    
    
    subplot 132; hold on; % change in action rate
    % change in action rate
    for i = 1:length(type)
        results(i).devalScore = preDevalRate(i)-postDevalRate(i);
        bar(i,results(i).devalScore,'FaceColor',map(i,:));
        
    end
    ylabel('\Delta press rate (pre-post)')
    axis([0 5 0 0.5])
    set(gca,'xtick',[1:4],'xticklabel',type)
    
    % beta at time of deval vs deval score
    
    %% beta relationship with deval score
    subplot 133; hold on;
    for i = 1:length(type)
        plot(mean(results(i).beta(sched.trainEnd-sched.testWin:sched.trainEnd)),results(i).devalScore,'.','Color',map(i,:),'MarkerSize',30);
    end
    xlabel('\beta')
    ylabel('\Delta press rate (pre-post)')
    prettyplot
    
    subprettyplot(1,3)
    set(gcf, 'Position',  [300, 300, 800, 300])
end

typCol = {'Blues','Greens','Reds','YlOrBr'};

%% moving plot of beta and policy cost/rho (Fig 5)
figure; hold on;
for i = 1:length(type)
    %subplot 211; hold on; axis([0 max(results(i).ecost(1:sched.trainEnd)) 0 max(results(i).rho(1:sched.trainEnd))])
    %subplot 212; hold on; axis([0 sched.trainEnd 0 max(results(i).beta(1:sched.trainEnd))])
    subplot 211; hold on;
    scatter(results(i).ecost(1:sched.trainEnd),results(i).rho2(1:sched.trainEnd), 20, brewermap(sched.trainEnd,typCol{i}), 'filled');
    
    subplot 212; hold on;
    scatter(1:sched.trainEnd,results(i).beta(1:sched.trainEnd), 20, brewermap(sched.trainEnd,typCol{i}), 'filled');
    colormap(brewermap(sched.trainEnd,typCol{i}))
end
subplot 211;
legend(h,type)
xlabel('policy cost C(\pi)')
ylabel('avg reward V^\pi')
title('reward-complexity during learning')

subplot 212; hold on;
xlabel('time (s)')
ylabel('\beta')
title('\beta over time')
% plot cost and beta at end of learning
subplot 211; hold on;
for i = 1:length(type)
    h(i,:) = plot(results(i).ecost(sched.trainEnd),results(i).rho2(sched.trainEnd),'o','MarkerFaceColor',map(i,:),'MarkerSize',20,'MarkerEdgeColor',[1 1 1]);
end
legend(h,type)
set(gcf, 'Position',  [400, 300, 500, 600])
subprettyplot(2,1)

%% diagnostics
if diagn ==1
    for i = 1:length(type)
        figure; subplot 421; histogram(results(i).NA.rew,50); ylabel('rew'); title('# actions');subplot 422; histogram(results(i).NT.rew,50);title('# seconds');
        subplot 423; histogram(results(i).NA.nonrew,50); ylabel('nonrew'); subplot 424; histogram(results(i).NT.nonrew,50);
        % how is theta being updated? plot rpe vs rewarded and non rewarded nt
        subplot 425; hold on; plot(results(i).NA.nonrew,results(i).RPE.nonrew,'r.'); plot(results(i).NA.rew,results(i).RPE.rew,'g.'); ylabel('RPE \delta')
        subplot 426; hold on; plot(results(i).NT.nonrew,results(i).RPE.nonrew,'r.'); plot(results(i).NT.rew,results(i).RPE.rew,'g.');
        subplot 427; hold on; plot(results(i).VS,results(i).RPE.nonrew,'r.'); plot(results(i).VS,results(i).RPE.rew,'g.');title('V(s`)-V(s)');
        subplot 428; hold on; plot((1./results(i).beta).*results(i).ecost,results(i).RPE.nonrew,'r.'); plot((1./results(i).beta).*results(i).ecost,results(i).RPE.rew,'g.'); title('\beta^{-1} C(\pi)');
        subprettyplot(4,2)
        suptitle(type{i})
        
    end
end


%% plot pas over time
% for i = 1:length(type)
%     figure; hold on;
%     subplot 311; hold on;
%     imagesc(squeeze(results(i).pas(:,2,:)))
%     colormap(flipud(gray))
%     set(gca,'YDir','reverse')
%     subplot 312; hold on;
%     plot(results(i).rpe)
%     subplot 313; hold on;
%     imagesc(results(i).w')
% end

%% probability of action (learned policy weights BEFORE devaluation) (Fig 6)
if sched.deval == 1
    if length(type) == 4 % if doing all 4 schedules at once
        figure; subplot 211;
        colormap(flipud(gray))
        imagesc([results(1).pas(:,2,sched.trainEnd) results(2).pas(:,2,sched.trainEnd) results(3).pas(:,2,sched.trainEnd) results(4).pas(:,2,sched.trainEnd)])
        caxis([0 1])
        set(gca,'xtick',[1:4],'xticklabel',type)
        ylabel('state feature')
        colorbar
        title('policy weights before deval, exp(\theta)')
    else
        figure; hold on; subplot 211;
        for i = 1:length(type)
            subplot(1,length(type),i);
            colormap(flipud(gray))
            imagesc(results(i).pas(:,2,sched.trainEnd))
            set(gca,'xtick',1,'xticklabel',type{i})
            ylabel('state feature');
            prettyplot
        end
        title('policy weights before deval, exp(\theta)')
    end
    
    %% probability of action (learned policy weights at the END OF TEST)
    if length(type) == 4 % if doing all 4 schedules at once
        subplot 212;
        colormap(flipud(gray))
        imagesc([results(1).pas(:,2,end) results(2).pas(:,2,end) results(3).pas(:,2,end) results(4).pas(:,2,end)])
        caxis([0 1])
        set(gca,'xtick',[1:4],'xticklabel',type)
        ylabel('state feature')
        colorbar
        subprettyplot(2,1)
        title('policy weights at end of test, exp(\theta)')
    else
        subplot 212;
        for i = 1:length(type)
            subplot(1,length(type),i);
            colormap(flipud(gray))
            imagesc(results(i).pas(:,2,end))
            set(gca,'xtick',1,'xticklabel',type{i})
            ylabel('state feature');
            prettyplot
        end
        title('policy weights at end of test, exp(\theta)')
    end
    set(gcf, 'Position',  [600, 300, 500, 800])
    
    % analyze difference in weights
    for i = 1:length(type)
        results(i).preDevalTheta = results(i).pas(:,:,sched.trainEnd);
        results(i).postDevalTheta = results(i).pas(:,:,sched.devalEnd);
        results(i).postTestTheta = results(i).pas(:,:,end);
        results(i).endTheta = results(i).pas(:,:,end);
        results(i).devalDiff = results(i).postDevalTheta - results(i).preDevalTheta; % post - pre, if this is positive, it means that theta has increased for that action
        results(i).testDiff = results(i).postTestTheta - results(i).preDevalTheta; % postTest - pre
        
    end
    %% average across state features (Fig 7)
    % plot change in theta
    figure; % subplot 411;
    %imagesc([results(1).devalDiff(:,1) (results(2).devalDiff(:,1)) (results(3).devalDiff(:,1)) (results(4).devalDiff(:,1))])
    subplot 211; bar([sum(results(1).devalDiff(:,1)) sum(results(2).devalDiff(:,1))  sum(results(3).devalDiff(:,1)) sum(results(4).devalDiff(:,1))])
    ylabel('post-preDeval \Delta \theta(~press)'); % positive means
    set(gca,'xtick',1:4,'xticklabel',type)
    %subplot 413;
    %imagesc([results(1).testDiff(:,1) (results(2).testDiff(:,1)) (results(3).testDiff(:,1)) (results(4).testDiff(:,1))])
    subplot 212; bar([sum(results(1).testDiff(:,1)) sum(results(2).testDiff(:,1))  sum(results(3).testDiff(:,1)) sum(results(4).testDiff(:,1))])
    ylabel('postTest-preDeval \Delta \theta(~press)'); % positive means
    subprettyplot(2,1)
    set(gcf, 'Position',  [500, 200, 500, 600])
    set(gca,'xtick',1:4,'xticklabel',type)
    
    %% beta duration satiation (Fig 8)
    % from trainEnd to devalEnd
    %     figure; hold on; subplot 231; hold on;
    %     for i = 1:length(type)
    %         plot(results(i).rpe(sched.trainEnd+1:sched.devalEnd-1))
    %     end
    %     prettyplot
    %     ylabel('\delta_t RPE')
    %     xlabel('satiation period (s)')
    %
    %     subplot 232; hold on;
    %     for i = 1:length(type)
    %         plot(results(i).r(sched.trainEnd+1:sched.devalEnd-1)-results(i).rho(sched.trainEnd+1:sched.devalEnd-1))
    %     end
    %     prettyplot
    %     ylabel('r-\rho')
    %     xlabel('satiation period (s)')
    %
    %     subplot 233; hold on;
    %     for i = 1:length(type)
    %         plot((1./results(i).beta(sched.trainEnd+1:sched.devalEnd-1)).*results(i).ecost(sched.trainEnd+1:sched.devalEnd-1))
    %     end
    %     prettyplot
    %     ylabel('\beta^{-1}*E[C(\pi)]')
    %     xlabel('satiation period (s)')
    %     % plot the RPEs before and after devaluation to show that
    %     % in FR/FI, since beta is high, the (r - 1/beta * C) should be higher and
    %     % should increase theta(not press) more than in VR/VI where beta is low, so
    %     % 1/beta * C is big a
    %
    %     % need 1/beta*C to also be higher in VR/VI > FR/FI
    %
    %     % also, the longer you train, the more policy weights increase, more beta
    %     % increases..
    %
    %     % from devalEnd to end
    %     subplot 234; hold on;
    %     for i = 1:length(type)
    %         plot(results(i).rpe(sched.devalEnd:end))
    %     end
    %     prettyplot
    %     ylabel('\delta_t RPE')
    %     xlabel('test period (s)')
    %
    %     subplot 235; hold on;
    %     for i = 1:length(type)
    %         plot(results(i).r(sched.devalEnd:end)-results(i).rho(sched.devalEnd:end))
    %     end
    %     prettyplot
    %     ylabel('r-\rho')
    %     xlabel('test period (s)')
    %
    %     subplot 236; hold on;
    %     for i = 1:length(type)
    %         plot((1./results(i).beta(sched.devalEnd:end)).*results(i).ecost(sched.devalEnd:end))
    %     end
    %     prettyplot
    %     ylabel('\beta^{-1}*E[C(\pi)]')
    %     xlabel('test period (s)')
    %
    %     set(gcf, 'Position',  [200, 200, 800, 500])
    %
else
    %% probability of action (learned policy weights)
    if length(type) == 4 % if doing all 4 schedules at once
        figure;
        colormap(flipud(gray))
        imagesc([results(1).pas(:,2,end) results(2).pas(:,2,end) results(3).pas(:,2,end) results(4).pas(:,2,end)])
        caxis([0 1])
        set(gca,'xtick',[1:4],'xticklabel',type)
        ylabel('state feature')
        title('policy weights, exp(\theta)')
        prettyplot
        colorbar
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


%% visualize weights (Fig 9)
% figure;
% colormap(flipud(gray))
% imagesc([results(1).w(end,:)' results(2).w(end,:)' results(3).w(end,:)' results(4).w(end,:)'])
% set(gca,'xtick',[1:4],'xticklabel',type)
% ylabel('state feature');
% prettyplot
% title('state feature value weights'); colorbar


%% policy / state weights over time (Fig 01)
% % bias weight (theta tap - theta not tap)
% figure; hold on;
% for i = 1:length(type)
%     subplot(2,4,i); hold on;
%     plot(results(i).w(:,1),'r')
%     %plot(results(i).w(:,2:end),'b')
%     plot(sum(results(i).w(:,2:end),2),'b')
%     plot([sched.trainEnd sched.trainEnd],ylim,'k--','LineWidth',2)
%     plot([sched.devalEnd sched.devalEnd],ylim,'k.-','LineWidth',2)
%     title(type{i})
% end
% subplot(2,4,1); hold on;
% ylabel('state value')
%
% for i = 1:length(type)
%     subplot(2,4,i+4); hold on;
%     plot(squeeze(results(i).theta(1,2,:)-results(i).theta(1,1,:)),'r');
%     %plot(squeeze(results(i).theta(2:end,2,:)-results(i).theta(2:end,1,:))','b');
%     plot(squeeze(sum(results(i).theta(2:end,2,:))-sum(results(i).theta(2:end,1,:)))','b');
%     plot([sched.trainEnd sched.trainEnd],ylim,'k--','LineWidth',2)
%     plot([sched.devalEnd sched.devalEnd],ylim,'k.-','LineWidth',2)
% end
% legend('habit component','state-dep. component')
% subplot(2,4,5); hold on;
% ylabel('\theta(press)-\theta(~press)')
% xlabel('time (s)')
% subprettyplot(2,4)
end



function makePlotsSess(results, sched, type)
map = habitColors;

%% data
load('all_data_cleaned.mat') % actual data

% format of test is: # rates x [val deval] columns , converted to actions
% per second
FI45.test = FI45.test./60;
VI45.test = VI45.test./60;
FR20.test = FR20.test./60;
VR20.test = VR20.test./60;

if sched.type=='FR'
    data = FR20;
    col = map(1,:);
elseif sched.type=='VR'
    data = VR20;
    col = map(2,:);
elseif sched.type=='FI'
    data = FI45;
    col = map(3,:);
elseif sched.type=='VI'
    data = VI45;
    col = map(4,:);
end

%% outcome and action rates over time
win = 100; % # seconds moving window
sess = size(results(1).sess,2); % get number of sessions
for i = 1:length(results) % for each subject
    for s = 1:sess % for each session
        if ismember(s,sched.devalsess)
            %results(i).sess(s).timeSteps = size(results(i).sess(s).a,2); % session length, timesteps;
            results(i).sess(s).outRate = sum(results(i).sess(s).x(1:sched.devalEnd)==2)/sched.devalEnd;
            results(i).sess(s).actRate = sum(results(i).sess(s).a(1:sched.devalEnd)==2)/sched.devalEnd;
            results(i).sess(s).avgRew = mean(results(i).sess(s).rho(1:sched.devalEnd));
            results(i).sess(s).avgBeta = mean(results(i).sess(s).beta(1:sched.devalEnd));
            results(i).sess(s).postTestRate = sum(results(i).sess(s).a(sched.devalEnd:end)==2)/sched.testWin;
            results(i).sess(s).preTestRate = sum(results(i).sess(s).a(sched.trainEnd-sched.testWin:sched.trainEnd)==2)/sched.testWin;
        else
            results(i).sess(s).timeSteps = size(results(i).sess(s).a,2); % session length, timesteps;
            timeSteps = results(i).sess(s).timeSteps;
            results(i).sess(s).outRate = sum(results(i).sess(s).x==2)/sched.trainEnd;
            results(i).sess(s).actRate = sum(results(i).sess(s).a==2)/sched.trainEnd;
            results(i).sess(s).avgRew = mean(results(i).sess(s).rho);
            results(i).sess(s).avgBeta = mean(results(i).sess(s).beta);
        end
    end
end

% if it's a devaluation session, separate that data out into results.test
figure; hold on;
subplot 311; hold on;
for i = 1:length(results)
    outRate(i,:) = cat(2,results(i).sess.outRate);
end
errorbar(mean(outRate),sem(outRate,1),'LineWidth',2,'Color',col);
prettyplot
ylabel('outcome rate')
legend([type],'FontSize',15); legend('boxoff')
axis([0 22 0 1])
plot([2.5 10.5 20.5;2.5 10.5 20.5],[ylim' ylim' ylim'],'k--')

subplot 312; hold on;
for i = 1:length(results)
    actRate(i,:) = cat(2,results(i).sess.actRate);
end
errorbar(mean(data.actRate),sem(data.actRate,1),'LineWidth',2,'Color',[0 0 0]);
errorbar(mean(actRate),sem(actRate,1),'LineWidth',2,'Color',col);
xlabel('session #')
ylabel('action rate')
axis([0 22 0 1])
plot([2.5 10.5 20.5;2.5 10.5 20.5],[ylim' ylim' ylim'],'k--')
prettyplot

subplot 313;  hold on;
for i = 1:length(results)
    beta(i,:) = cat(2,results(i).sess.avgBeta);
end
errorbar(mean(beta),sem(beta,1),'LineWidth',2,'Color',col);
axis([0 22 ylim])
xlabel('session #')
ylabel('\beta')
prettyplot

set(gcf, 'Position',  [300, 300, 800, 500])

%% devaluation
for d = 1:3
    for i = 1:length(results)
        test(i,:,d) = [results(i).sess([sched.devalsess(d)]).preTestRate results(i).sess([sched.devalsess(d)]).postTestRate];
    end
end

figure; hold on;
for d = 1:3
    subplot 121; hold on;
    [b e] = barwitherr(sem(data.test(:,:,d),1),d, mean(data.test(:,:,d)),'FaceColor',[0 0 0]);
    b(2).FaceColor = [1 1 1];
    b(2).EdgeColor = [0 0 0];
    b(2).LineWidth = 2;
    e.Color = [0 0 0];
    e.LineWidth = 2;
    
    title(strcat('data - ',sched.type))
    
    subplot 122; hold on;
    [b e] = barwitherr(sem(test(:,:,d),1),d, mean(test(:,:,d)),'FaceColor',col);
    b(2).FaceColor = [1 1 1];
    b(2).EdgeColor = col;
    b(2).LineWidth = 2;
    e.Color = col;
    e.LineWidth = 2;
    
    title('model')
end
subprettyplot(1,2)


why

%
% %% beta and policy cost
% figure; hold on;
% subplot 211; hold on;
% for i = 1:length(type)
%     %h(i,:) = shadedErrorBar(1:timeSteps,mean(cat(1, results(i,:).beta)), std(cat(1, results(i,:).beta)),{'LineWidth',1.5,'Color',map(i,:)},1);
%     plot(movmean(cat(2, results(i,:).beta),500,'Endpoints','shrink'),'LineWidth',1.5);
% end
% %plot([sched.devalEnd sched.devalEnd],ylim,'k--','LineWidth',2)
% ylabel(' \beta')
% legend([type 'devaluation'],'FontSize',15); legend('boxoff')
%
% subplot 212; hold on;
% for i = 1:length(type)
%     %shadedErrorBar(1:timeSteps,mean(movmean(cat(2,results(i,:).cost),500,'Endpoints','shrink')'),std(movmean(cat(2, results(i,:).cost),500,'Endpoints','shrink')'),{'LineWidth',1.5,'Color',map(i,:)},1);
%     plot(movmean(cat(2,results(i,:).cost),500,'Endpoints','shrink'),'LineWidth',1.5);
% end
% %plot([sched.devalEnd sched.devalEnd],ylim,'k--','LineWidth',2)
% ylabel('policy cost C(\pi_\theta)')
%
% subprettyplot(2,1)
% set(gcf, 'Position',  [300, 300, 700, 500])
%
%
% %% action rate before and after devaluation (avg over many iters)
% if sched.devalEnd ~= timeSteps
%     testWin = timeSteps-sched.devalEnd;
%
%     figure; subplot 121; hold on;
%     for i = 1:length(type)
%         all_a = cat(1,results(i,:).a);
%         pre_deval = sum(sum(all_a(:,sched.devalEnd-testWin:sched.devalEnd)==2))/(testWin*size(results,2));
%         post_deval =  sum(sum(all_a(:,sched.devalEnd+1:end)==2))/(testWin*size(results,2));
%         b = bar(i,[pre_deval post_deval],'FaceColor',map(i,:));
%         b(2).FaceColor = [1 1 1];
%         b(2).EdgeColor = map(i,:);
%         b(2).LineWidth = 2;
%
%         results(i).devalScore = pre_deval - post_deval;
%     end
%     ylabel('action (press) rate')
%     legend('before','after','FontSize',15,'Location','NorthWest'); legend('boxoff')
%     set(gca,'xtick',[1:4],'xticklabel',type)
%
%
%     subplot 122; hold on; % change in action rate
%     for i = 1:length(type)
%         bar(i,results(i).devalScore,'FaceColor',map(i,:));
%     end
%     ylabel('devaluation score (pre-post)')
%     axis([0 5 0 0.5])
%     set(gca,'xtick',[1:4],'xticklabel',type)
%
%     subprettyplot(1,2)
%
%     set(gcf, 'Position',  [300, 300, 800, 300])
%
%     %% beta relationship with deval score
%
%     figure(100); subplot 121;hold on;
%     for i = 1:length(type)
%         plot(mean(results(i).beta(sched.devalEnd-win:sched.devalEnd)),results(i).devalScore,'.','Color',map(i,:),'MarkerSize',30);
%     end
%     xlabel('\beta')
%     ylabel('devaluation score')
%     prettyplot
% end
%
%
% %% relationship between a changing beta and a changing policy complexity
%
% % moving est of mutual information
% for i = 1:length(type)
%     results(i).movMI = movmean(results(i).mi, win,'Endpoints','shrink');
%     results(i).movCost = movmean(results(i).cost,win,'Endpoints','shrink');
%     results(i).movRew = movmean(results(i).rho,win,'Endpoints','shrink');
%     results(i).movBeta = movmean(results(i).beta,win,'Endpoints','shrink');
%     results(i).estBeta = gradient(results(i).movCost)./gradient(results(i).movRew);
% end
%
% % what is relationship between changing beta and a changing policy complexity
% % does beta predict the policy complexity? --> YES, the bigger the beta,
% % the bigger the policy complexity is allowed to be
%
% % if dbeta is 0, v changes w I the same
% % if dbeta is big, policy cost changes a lot (there's a lot of room to move) for the same reward

end
