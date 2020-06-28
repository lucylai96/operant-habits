
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
figure;subplot 121; hold on;
colormap(flipud(gray))
for i = 1:length(type)
    results(i).Pa = exp(results(i).theta);
    results(i).Pa = results(i).Pa./sum(results(i).Pa,2);
    imagesc([results(i).Pa(:,2,end)])
end

set(gca,'xtick',[1:size(results,2)],'xticklabel',type)
ylabel('state')
title('p(a=press)')
prettyplot

%% visualize weights
subplot 122; hold on 
colormap(flipud(gray))
for i = 1:length(type)
imagesc([results(i).w(end,:)'])
end
set(gca,'xtick',[1:size(results,2)],'xticklabel',type)
ylabel('state')
title('V(s)')
prettyplot

%% outcome and action rates over time
win = 100; % # seconds moving window
for i = 1:length(type)
    results(i).outRate = sum(results(i).x==2)/timeSteps;
    results(i).actRate = sum(results(i).a==2)/timeSteps;
    
    % moving average
    results(i).movOutRate = movmean(results(i).x-1, win,'Endpoints','shrink');
    results(i).movActRate = movmean(results(i).a-1, win,'Endpoints','shrink');
    
    % moving avg reward
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
    plot(movmean(results(i).cost,win,'Endpoints','shrink'),'LineWidth',1.5)
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel('policy cost C(\pi_\theta)')

subplot 413; hold on;
for i = 1:length(type)
    plot(movmean((1./results(i).beta).* results(i).cost', win,'Endpoints','shrink'),'LineWidth',1.5)
end
plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
ylabel('\beta^{-1} C(\pi_\theta)')

subplot 414; hold on;
for i = 1:length(type)
    plot(movmean(results(i).mi, win),'LineWidth',1.5)
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
    
end
set(gcf, 'Position',  [300, 300, 800, 300])


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

% figure; hold on;
% for i = 1:length(type)
%     bar(i,mean(results(i).mi(end-num:end)),'FaceColor',map(i,:));
% end
% set(gca,'xtick',[1:4],'xticklabel',type)
% ylabel('mutual information I(S;A)')
% prettyplot

%% rho and rpe
figure; hold on; subplot 221; hold on;
for i = 1:length(type)
    plot(movmean(results(i).rho,win,'Endpoints','shrink'),'LineWidth',1.5);
end

ylabel('avg rew')
prettyplot

subplot 222; hold on;
for i = 1:length(type)
    plot(movmean(results(i).rpe,win,'Endpoints','shrink'),'LineWidth',1.5);
end
xlabel('time(s)')
ylabel('\delta')
prettyplot

subplot 223; hold on;
for i = 1:length(type)
    plot(results(i).movCost,results(i).movRew,'.');
end

subplot 224; hold on;
for i = 1:length(type)
    plot(results(i).movMI,results(i).movRew,'.');
end

%% instantaneous contingency
figure; hold on; subplot 121; hold on;
for i = 1:length(type)
    plot(movmean(results(i).cont,win,'Endpoints','shrink'),'LineWidth',2);
end

ylabel('contingency')
xlabel('time (s)')
prettyplot

% contingency and press rate
subplot 122; hold on;
for i = 1:length(type)
    plot(results(i).movActRate,movmean(results(i).cont,win,'Endpoints','shrink'),'.');
end
ylabel('contingency')
xlabel('action rate \lambda_a')
prettyplot
end
