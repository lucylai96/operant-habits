
function makePlotsData(data,model,type)
%% init color map
map = brewermap(4,'*RdBu');
temp = map(3,:);
map(3,:) = map(4,:);
map(4,:) = temp;  % swap colors so darker ones are fixed
set(0, 'DefaultAxesColorOrder', map) % first three rows

timeSteps = length(data.lever);

%% cumulative sum plots
num = 500; % last how many # trials to look at
figure; hold on;
cs = cumsum(data.lever(end-num:end));
h(1,:) = plot(cs,'Color',map(1,:),'LineWidth',2);
xplot = find(data.reward(end-num:end)==1);
line([xplot' xplot'+10]',[cs(xplot)' cs(xplot)']','LineWidth',2,'Color','k');

cs = cumsum(model.a(timeSteps-num:timeSteps)-1);
h(2,:) = plot(cs,'Color',map(2,:),'LineWidth',2);
xplot = find(model.x(timeSteps-num:timeSteps)==2);
line([xplot' xplot'+10]',[cs(xplot)' cs(xplot)']','LineWidth',2,'Color','k');

legend(h,{'data','model'});
legend('boxoff')
xlabel('time')
ylabel('cumulative # actions')
prettyplot

%% outcome and action rates over time
win = 100; % # seconds moving window
for i = 1:length(type)
    outRate = sum(data.reward==1)/timeSteps;
    actRate = sum(data.lever==1)/timeSteps;
    
    % moving average
    movOutRate = movmean(data.reward, win,'Endpoints','shrink');
    movActRate = movmean(data.lever, win,'Endpoints','shrink');
    
    
end

figure; hold on;
subplot(2,4,1:3); hold on;
%for i = 1:length(type)
plot(movOutRate,'LineWidth',1.5)
%end
%plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
prettyplot
ylabel('outcome rate')
legend([type 'devaluation'],'FontSize',15); legend('boxoff')

subplot(2,4,5:7); hold on;

plot(movActRate,'LineWidth',1.5)

xlabel('time (s)')
ylabel('action rate')
prettyplot

subplot(2,4,4); hold on;

bar(1,outRate);

legend(type); legend('boxoff')
set(gca,'xtick',[1:2],'xticklabel',type)
prettyplot
subplot(2,4,8); hold on;
for i = 1:length(type)
    bar(1,actRate);
end
set(gca,'xtick',[1:2],'xticklabel',type)
prettyplot



for i = 1:length(type)
    outRate = sum(model.x(1:timeSteps)==2)/timeSteps;
    actRate = sum(model.a(1:timeSteps)==2)/timeSteps;
    
    % moving average
    movOutRate = movmean(model.x-1, win,'Endpoints','shrink');
    movActRate = movmean(model.a-1, win,'Endpoints','shrink');
    
    
end

subplot(2,4,1:3); hold on;
%for i = 1:length(type)
plot(movOutRate,'LineWidth',1.5)
%end
plot([timeSteps timeSteps],ylim,'k--','LineWidth',2)
prettyplot
ylabel('outcome rate')
legend([type 'devaluation'],'FontSize',15); legend('boxoff')

subplot(2,4,5:7); hold on;
for i = 1:length(type)
    plot(movActRate,'LineWidth',1.5)
end
plot([timeSteps timeSteps],ylim,'k--','LineWidth',2)
xlabel('time (s)')
ylabel('action rate')
prettyplot

subplot(2,4,4); hold on;

bar(2,outRate);

legend('data','model'); legend('boxoff')
set(gca,'xtick',[1:2],'xticklabel',type)
prettyplot

subplot(2,4,8); hold on;
bar(2,actRate);

set(gcf, 'Position',  [300, 300, 800, 500])
%
% %% action rate before and after devaluation
% if sched.devalTime ~= timeSteps
%     devalWin = timeSteps-sched.devalTime;
%
%     figure; subplot 121; hold on;
%     for i = 1:length(type)
%         b = bar(i,[sum(results(i).a(sched.devalTime-devalWin:sched.devalTime)-1)/devalWin sum(results(i).a(sched.devalTime+1:end)-1)/devalWin],'FaceColor',map(i,:));
%         b(2).FaceColor = [1 1 1];
%         b(2).EdgeColor = map(i,:);
%         b(2).LineWidth = 2;
%
%     end
%     ylabel('action (press) rate')
%     legend('before','after','FontSize',15,'Location','NorthWest'); legend('boxoff')
%     set(gca,'xtick',[1:4],'xticklabel',type)
%
%
%     subplot 122; hold on; % change in action rate
%     % change in action rate
%     for i = 1:length(type)
%         results(i).devalScore = sum(results(i).a(sched.devalTime-devalWin:sched.devalTime)-1)/devalWin - sum(results(i).a(sched.devalTime+1:end)-1)/devalWin;
%         bar(i,results(i).devalScore,'FaceColor',map(i,:));
%     end
%     ylabel('devaluation score (pre-post)')
%     axis([0 5 0 0.5])
%     set(gca,'xtick',[1:4],'xticklabel',type)
%
%     % beta at time of deval vs deval score
%     subprettyplot(1,2)
%
% end
% set(gcf, 'Position',  [300, 300, 800, 300])

end
