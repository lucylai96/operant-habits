function example_rat(sch,r)
% for plotting all data of 1 example animal (data)
% s = schedule (1 - FR, 2 - VR, 3 - FI, 4 - VI)
% r = rat #
% FR - 16 rats, VR - 16 rats, FI - 8 rats, VI- 8 rats
% Written by: Lucy Lai (Mar 2021)

% example input: example_rat(1,1)

load('data.mat'); % more compact form
map = habitColors;

data = schedule(sch).rat(r);
a = []; x = []; % unroll

for s = 1:20 % 20 sessions
    a = [a data.session(s).training.lever_binned];
    x = [x data.session(s).training.reward_binned];
end

win = 200; % # seconds moving window
figure(100); hold on;
subplot 211; hold on;
plot(movmean(a, win,'Endpoints','shrink'),'LineWidth',1.5,'Color',map(sch,:))
plot([cumsum(data.timeSteps);cumsum(data.timeSteps)],repmat(ylim',1,20),'k--')
ylabel('lever press rate (/sec)')
prettyplot; axis tight

subplot 212; hold on;
plot(movmean(x, win,'Endpoints','shrink'),'LineWidth',1.5,'Color',map(sch,:))
plot([cumsum(data.timeSteps);cumsum(data.timeSteps)],repmat(ylim',1,20),'k--')
prettyplot; axis tight

xlabel('time (s)')
ylabel('reward rate (/sec)')

%% cumsum
figure(200); hold on;
cs = cumsum(a);
plot(cs,'Color',map(sch,:),'LineWidth',2);
xplot = find(x==1);
%line([xplot' xplot'+100]',[cs(xplot)' cs(xplot)']','LineWidth',2,'Color','k');

xlabel('time (s)')
ylabel('cumulative # actions')
prettyplot
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

example_rat(1,3)
example_rat(2,3)
example_rat(3,3)
example_rat(4,3)

end