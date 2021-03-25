function example_rat_sim(sch,r)
% for plotting all results of 1 example animal (results)
% s = schedule (1 - FR, 2 - VR, 3 - FI, 4 - VI)
% r = rat #
% FR - 16 rats, VR - 16 rats, FI - 8 rats, VI- 8 rats
% Written by: Lucy Lai (Mar 2021)

map = habitColors;

% plot data
example_rat(sch,r)

% load fitted parameters
load(strcat('sch',num2str(sch),'_r',num2str(r),'.mat'));
%params = [0.0001    0.0040    0.0090    1.2096]; % 15067.3

% simulate results under parameters
results = sim_habit(params, sch, r);

a = results.a-1;
x = results.x-1;

%% moving avg
win = 200; % # seconds moving window
figure; hold on;
subplot 211; hold on;
plot(movmean(a, win,'Endpoints','shrink'),'LineWidth',1.5,'Color',map(sch,:))
plot([results.timeSteps;results.timeSteps],repmat(ylim',1,20),'k--')
ylabel('lever press rate (/sec)')
prettyplot; ylim([0 1])

subplot 212; hold on;
plot(movmean(x, win,'Endpoints','shrink'),'LineWidth',1.5,'Color',map(sch,:))
plot([results.timeSteps;results.timeSteps],repmat(ylim',1,20),'k--')
prettyplot; ylim([0 0.1])

xlabel('time (s)')
ylabel('reward rate (/sec)')

%% cumsum
figure; hold on;
cs = cumsum(a);
plot(cs,'Color',map(sch,:),'LineWidth',2);
xplot = find(x==1);
%line([xplot' xplot'+100]',[cs(xplot)' cs(xplot)']','LineWidth',2,'Color','k');
xlabel('time (s)')
ylabel('cumulative # actions')
prettyplot


%% R-C curve
figure(300); hold on;
plot(results.ecost,results.avgr,'.','Color',map(sch,:),'MarkerSize',20); hold on;
plot(results.ecost(end),results.avgr(end),'ko','MarkerSize',20,'MarkerFaceColor',map(sch,:))
xlabel('Policy complexity')
ylabel('Average reward')
%title(strcat('Arming parameter:',num2str(sched.R)))
prettyplot(20)
end

function overlay
% not same animal but good way to compare across schedules

example_rat(1,1)
example_rat(2,1)
example_rat(3,1)
example_rat(4,1)

example_rat(1,2)
example_rat(2,2)
example_rat(3,2)
example_rat(4,2)

end