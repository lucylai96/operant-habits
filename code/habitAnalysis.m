function habitAnalysis(fig)
% PURPOSE: reproducing major habit findings
% Written by Lucy Lai

% plot settings
addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools'); % various plotting tools
map = brewermap(4,'*RdBu');
temp = map(3,:);
map(3,:) = map(4,:);
map(4,:) = temp;                     % swap colors so darker ones are fixed
set(0, 'DefaultAxesColorOrder', map) % first three rows

% some default parameters
sched.model = 4;      % which lesioned model to run
sched.acost = 0.1;   % action cost
sched.beta = 1;     % starting beta; high beta = low cost. beta should increase for high contingency
%sched.cmax = 0.5;     % max complexity (low v high)

switch fig
    
    case 1
        agent.alpha_w = 0.1;
        agent.alpha_t = 0.1;
        agent.alpha_r = 0.1;
        agent.alpha_b = 5;
        sched.cmax = 100;
        %% REED 2001
        
        sched.R = 8;
        sched.I = 20;
        % In Experiment 1, the time between outcomes obtained on a VR8 (1-16);
        % schedule became the intervals for a yoked VI schedule.
        
        % for VR only
        sched.actions = contingency_generateVR(sched.R);   % pre-generated number of actions before reward
        sched.actions(sched.actions > 16) = [];
        
        sched.k = 1;
        sched.timeSteps = 180;
        sched.devalTime = sched.timeSteps;                 % timestep where outcome gets devalued
        sched.type = 'VR';
        
        %[O,T] =  habitSchedule(sched);
        VR = habitAgent(sched,agent);
        
        outTimes = find(VR.x==2);                     % find time between outcomes
        
        sched.times = [diff(outTimes) diff(outTimes)];
        sched.type = 'VI';
        
        %[O,T] =  habitSchedule(sched);
        VIyoked = habitAgent(sched,agent);
        
        % Response rates were higher on the VR than on the VI schedule.
        figure; hold on; subplot 121; hold on;
        actRate(1) = sum(VR.a-1)/sched.timeSteps; % action rate for VR
        actRate(2) = sum(VIyoked.a-1)/sched.timeSteps; % action rate for VI yoked
        bar(1,actRate(1)*60,'FaceColor',map(2,:));
        bar(2,actRate(2)*60,'FaceColor',map(4,:));
        ylabel('responses per min')
        xlabel('schedule')
        set(gca,'xtick',[1 2],'xticklabel',{'ratio','interval'})
        title('Reed(2001) Expt 1: VI yoked to VR')
        prettyplot
        
        
        % In Experiment 2, the number of responses required per outcome on a
        % VR schedule were matched to that on a master VI 20-s (1-40s) schedule.
        
        % for VI only
        sched.times = contingency_generateVI(sched.I);     % pre-generated wait times before reward
        sched.times(sched.times>40) = [];
        sched.type = 'VI';
        
        %[O,T] =  habitSchedule(sched);
        VI = habitAgent(sched,agent);
        
        outTimes = find(VI.x==2);                     % find time between outcomes
        k = 1;
        for i = 1:length(outTimes)-1
            numAct(k) = sum(VI.a(outTimes(i):outTimes(i+1))-1); % sum up the number of actions between outTimes
            k = k+1;
        end
        
        sched.actions = [numAct numAct];
        sched.type = 'VR';
        
        %[O,T] =  habitSchedule(sched);
        VRyoked = habitAgent(sched,agent);
        
        % Response rates were higher on the VR than on the VI schedule.
        
        subplot 122; hold on;
        actRate(1) = sum(VR.a-1)/sched.timeSteps; % action rate for VR
        actRate(2) = sum(VIyoked.a-1)/sched.timeSteps; % action rate for VI yoked
        bar(1,actRate(1)*60,'FaceColor',map(2,:));
        bar(2,actRate(2)*60,'FaceColor',map(4,:));
        ylabel('responses per min')
        xlabel('schedule')
        set(gca,'xtick',[1 2],'xticklabel',{'ratio','interval'})
        title('Reed(2001) Expt 2: VR yoked to VI')
        prettyplot
        
        
        % Both ratings of causal effectiveness and response rates were higher in the VR schedule.
        % (Analyze contingency)
        
    case 2
        %% decreasing action rates for increasing ratio and interval parameter schedules
        % double check if original studies were variable or fixed
        
        clear VI
        clear VR
        R = [20 40 80];
        I = [30 53 109];
        % delay = [16 32 64];
        sched.k = 1;
        sched.timeSteps = 3000;
        sched.devalTime = sched.timeSteps;                 % timestep where outcome gets devalued
        
        for i = 1:length(R)
            % for VR only
            sched.R = R(i);
            sched.actions = contingency_generateVR(sched.R);   % pre-generated number of actions before reward
            sched.actions(sched.actions > 2*sched.R) = [];
            sched.type = 'VR';
            
            [O,T] =  habitSchedule(sched);
            VR = habitAgent(O,T, sched);
            VRactRate(i) = sum(VR.a-1)/sched.timeSteps;
            
            % for VI only
            sched.I = I(i);
            sched.times = contingency_generateVI(sched.I);   % pre-generated number of actions before reward
            sched.times(sched.times > 2*sched.I) = [];
            sched.type = 'VI';
            
            [O,T] =  habitSchedule(sched);
            VI = habitAgent(O,T, sched);
            VIactRate(i) = sum(VI.a-1)/sched.timeSteps;
        end
        
        figure; hold on; subplot 121;
        plot(R,VRactRate*60,'.-','Color',map(1,:),'MarkerSize',20)
        xlabel('ratio parameter')
        ylabel('press rate')
        subplot 122;
        plot(I,VIactRate*60,'.-','Color',map(3,:),'MarkerSize',20)
        xlabel('interval parameter')
        ylabel('press rate')
        subprettyplot(1,2)
        
    case 3
        map = brewermap(4,'RdGy');
        set(0, 'DefaultAxesColorOrder', map) % first three rows
        %% Garr's data
        type = {'FR' 'VR' 'FI' 'VI'};
        load('example_rats_cleaned.mat'); % load data
        
        sched.model = 4; % fit the beta
        sched.R = 20; % in deciseconds (VR/FR20*10 = 200)
        sched.I = 45; % in deciseconds  (VI/FI45*10 = 450)
        sp = 1;
        for typ = 1:length(type)  % for all sched types
            for ses = 1:3         % for all sessions
                sched.type = type{typ};
                sched.timeSteps = timeSteps(typ,ses);
                load(strcat(sched.type,'_params_s',num2str(ses),'.mat')); % get params and error
                %params = [0.0151    0.0122    0.7151    0.0702    2.1222];
                %params = mean(params);
                
                %% for FR only
                if sched.type == 'FR'
                    data.lever = FR20.session(ses).training.lever_binned;
                    data.reward = FR20.session(ses).training.reward_binned;
                end
                
                %% for VR only
                if sched.type == 'VR'
                    sched.actions = [VR20.session(ses).actions];
                    data.lever = VR20.session(ses).training.lever_binned;
                    data.reward = VR20.session(ses).training.reward_binned;
                end
                
                %% for FI only
                if sched.type == 'FI'
                    data.lever = FI45.session(ses).training.lever_binned;
                    data.reward = FI45.session(ses).training.reward_binned;
                end
                
                %% for VI only
                if sched.type == 'VI'
                    sched.times = [VI45.session(ses).times];
                    data.lever = VI45.session(ses).training.lever_binned;
                    data.reward = VI45.session(ses).training.reward_binned;
                end
                
                %% sanity checks (plots)
                %makePlotsData(data.lever,data.reward,{sched.type})
                % simulate valued and devalued
                sched.deval = 1;
                [model_devalued, sched] = habitSimulate(params, sched);
                makePlotsData(data, model_devalued, {sched.type})
                
                sched.deval = 0;
                [model_valued, sched] = habitSimulate(params, sched);
                makePlotsData(data, model_valued, {sched.type})
                
                
                %% sanity checks (compare training)
                MD.train_reward(typ,ses) = mean(cat(1,model_devalued.outRate));
                MD.train_lever(typ,ses) = mean(cat(1,model_devalued.actRate));
                
                MV.train_reward(typ,ses) = mean(cat(1,model_valued.outRate));
                MV.train_lever(typ,ses) = mean(cat(1,model_valued.actRate));
                
                DD.train_reward(typ,ses) = sum(data.reward==1)/timeSteps(typ,ses);
                DD.train_lever(typ,ses) = sum(data.lever==1)/timeSteps(typ,ses);
                
                % compare test
                MD.test_lever(typ,ses) = mean(cat(1,model_devalued.actRateTest));
                MV.test_lever(typ,ses) = mean(cat(1,model_valued.actRateTest));
                
                DD.test_lever(typ,ses) = test_lever_rates.devalued(typ,ses)/60; % per minute / 60 = seconds
                DV.test_lever(typ,ses) = test_lever_rates.valued(typ,ses)/60;
                
                % plot change in lever press rate
                figure(100); hold on;
                subplot(4,3,sp); hold on;
                bar(1,[DV.test_lever(typ,ses) DD.test_lever(typ,ses)]./DD.train_lever(typ,ses)); % data (devalued, valued)
                bar(2,[MV.test_lever(typ,ses) MD.test_lever(typ,ses)]./[MV.train_lever(typ,ses) MD.train_lever(typ,ses)]); % model (devalued, valued)
                set(gca,'xtick',[1 2],'xticklabel',{'data' 'model'})
                prettyplot
                
                figure(200); hold on;
                subplot(4,3,sp); hold on;
                bar(1,[DV.test_lever(typ,ses) DD.test_lever(typ,ses)]); % data (devalued, valued)
                bar(2,[MV.test_lever(typ,ses) MD.test_lever(typ,ses)]); % model (devalued, valued)
                
                
                set(gca,'xtick',[1 2],'xticklabel',{'data' 'model'})
                prettyplot
                sp = sp+1;
            end
            
            
        end
        subplot (4,3,1)
        title('2 days')
       legend('valued', 'devalued', 'valued', 'devalued'); legend('boxoff')
        subplot (4,3,2)
        title('10 days')
        subplot (4,3,3)
        title('20 days')
        
        subplot (4,3,10)
        ylabel('percent baseline lever press')
        legend('valued', 'devalued', 'valued', 'devalued'); legend('boxoff')
        % to convert from mins to deciseconds:
        % 5 mins * 600 decisecs per min = 300 "timesteps"
        % 27361 decisecs / 600 decisecs per min ~= 45 mins = 2736 seconds
end

%% undertraining vs overtraining
% more habit in overtraining

%% contingency degredation (give free rewards)

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



end


function makePlotsData(data,model,type)
%% init color map
map = brewermap(4,'*RdBu');
temp = map(3,:);
map(3,:) = map(4,:);
map(4,:) = temp;  % swap colors so darker ones are fixed
set(0, 'DefaultAxesColorOrder', map) % first three rows

timeSteps = length(data.lever);

% %% cumulative sum plots
% num = 500; % last how many # trials to look at
% figure; hold on;
% cs = cumsum(data.lever(end-num:end));
% h(1,:) = plot(cs,'Color',map(1,:),'LineWidth',2);
% xplot = find(data.reward(end-num:end)==1);
% line([xplot' xplot'+10]',[cs(xplot)' cs(xplot)']','LineWidth',2,'Color','k');
% 
% cs = cumsum(model.a(timeSteps-num:timeSteps)-1);
% h(2,:) = plot(cs,'Color',map(2,:),'LineWidth',2);
% xplot = find(model.x(timeSteps-num:timeSteps)==2);
% line([xplot' xplot'+10]',[cs(xplot)' cs(xplot)']','LineWidth',2,'Color','k');
% 
% legend(h,{'data','model'});
% legend('boxoff')
% xlabel('time')
% ylabel('cumulative # actions')
% prettyplot

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
    outRate = mean(cat(1,model.outRate));
    actRate = mean(cat(1,model.actRate));
    
    % moving average
    movOutRate = mean(cat(1,model.movOutRate));
    movActRate = mean(cat(1,model.movActRate));
    
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

%%

% 
% figure; hold on;
% subplot 311; hold on;
% for i = 1:length(type)
%     plot(model(i).beta,'LineWidth',1.5)
% end
% plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
% ylabel(' \beta')
% legend([type 'devaluation'],'FontSize',15); legend('boxoff')
% 
% subplot 312; hold on;
% for i = 1:length(type)
%     plot(movmean(model.cost,500,'Endpoints','shrink'),'LineWidth',1.5)
% end
% plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
% ylabel('policy cost C(\pi_\theta)')
% 
% subplot 313; hold on;
% for i = 1:length(type)
%     plot(movmean((1./model.beta).* model.cost', win,'Endpoints','shrink'),'LineWidth',1.5)
% end
% plot([sched.devalTime sched.devalTime],ylim,'k--','LineWidth',2)
% ylabel('\beta^{-1} C(\pi_\theta)')
% 
% subprettyplot(3,1)
% 
% set(gcf, 'Position',  [300, 300, 700, 500])
% 
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