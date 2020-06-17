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
sched.acost = 0.05;   % action cost
sched.beta = 100;     % starting beta; high beta = low cost. beta should increase for high contingency
sched.cmax = 0.5;     % max complexity (low v high)

switch fig
    
    case 1
        %% REED 2001
        
        sched.R = 8;
        sched.I = 20;
        % In Experiment 1, the time between outcomes obtained on a VR8 (1-16);
        % schedule became the intervals for a yoked VI schedule.
        
        % for VR only
        sched.actions = contingency_generateVR(sched.R);   % pre-generated number of actions before reward
        sched.actions(sched.actions > 16) = [];
        
        sched.k = 1;
        sched.timeSteps = 2200;
        sched.devalTime = sched.timeSteps;                 % timestep where outcome gets devalued
        sched.type = 'VR';
        
        [O,T] =  habitSchedule(sched);
        VR = habitAgent(O,T, sched);
        
        outTimes = find(VR.x==2);                     % find time between outcomes
        
        sched.times = [diff(outTimes) diff(outTimes)];
        sched.type = 'VI';
        
        [O,T] =  habitSchedule(sched);
        VIyoked = habitAgent(O,T, sched);
        
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
        
        [O,T] =  habitSchedule(sched);
        VI = habitAgent(O,T, sched);
        
        outTimes = find(VI.x==2);                     % find time between outcomes
        k = 1;
        for i = 1:length(outTimes)-1
            numAct(k) = sum(VI.a(outTimes(i):outTimes(i+1))-1); % sum up the number of actions between outTimes
            k = k+1;
        end
        
        sched.actions = [numAct numAct];
        sched.type = 'VR';
        
        [O,T] =  habitSchedule(sched);
        VRyoked = habitAgent(O,T, sched);
        
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
        sched.timeSteps = 6000;
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
        
        sched.model = 2; % fit the beta
        sched.R = 20; % in deciseconds (VR/FR20*10 = 200)
        sched.I = 45; % in deciseconds  (VI/FI45*10 = 450)
        figure; hold on; sp = 1;
        for typ = 1:length(type)  % for all sched types
            for ses = 1:3         % for all sessions
                sched.type = type{typ};
                sched.timeSteps = timeSteps(typ,ses);
                load(strcat(sched.type,'_params_s',num2str(ses),'.mat')); % get params and error
                
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
                
                % simulate valued and devalued
                sched.deval = 1;
                model_devalued = habitSimulate(params, sched);
                
                sched.deval = 0; sched.timeSteps = sched.timeSteps+300;
                model_valued = habitSimulate(params, sched);
                
                
                % sanity checks (compare training)
                MD.train_reward(typ,ses) = sum(model_devalued.x(1:timeSteps(typ,ses))==2)/timeSteps(typ,ses);
                MD.train_lever(typ,ses) = sum(model_devalued.a(1:timeSteps(typ,ses))==2)/timeSteps(typ,ses);
                
                MV.train_reward(typ,ses) = sum(model_valued.x(1:timeSteps(typ,ses))==2)/timeSteps(typ,ses);
                MV.train_lever(typ,ses) = sum(model_valued.a(1:timeSteps(typ,ses))==2)/timeSteps(typ,ses);
                
                DD.train_reward(typ,ses) = sum(data.reward==1)/timeSteps(typ,ses);
                DD.train_lever(typ,ses) = sum(data.lever==1)/timeSteps(typ,ses);
                
                % compare test
                MD.test_lever(typ,ses) = sum(model_devalued.a(timeSteps(typ,ses)+1:end)==2)/length(model_devalued.a(timeSteps(typ,ses)+1:end));
                MV.test_lever(typ,ses) = sum(model_valued.a(timeSteps(typ,ses)+1:end)==2)/length(model_valued.a(timeSteps(typ,ses)+1:end));
                
                DD.test_lever(typ,ses) = test_lever_rates.devalued(typ,ses)/60; % per minute / 60 = seconds
                DV.test_lever(typ,ses) = test_lever_rates.valued(typ,ses)/60;
                
                % plot change in lever press rate
                % figure(1); hold on;
                subplot(4,3,sp); hold on;
                bar(1,[DV.test_lever(typ,ses) DD.test_lever(typ,ses)]./DD.train_lever(typ,ses)); % data (devalued, valued)
                bar(2,[MV.test_lever(typ,ses) MD.test_lever(typ,ses)]./[MV.train_lever(typ,ses) MD.train_lever(typ,ses)]); % model (devalued, valued)
                set(gca,'xtick',[1 2],'xticklabel',{'data' 'model'})
                prettyplot
                
                % figure(2); hold on;
                % subplot(4,3,sp); hold on;
                % bar(1,[DV.test_lever(typ,ses) DD.test_lever(typ,ses)]); % data (devalued, valued)
                % bar(2,[MV.test_lever(typ,ses) MD.test_lever(typ,ses)]); % model (devalued, valued)
                
                
                set(gca,'xtick',[1 2],'xticklabel',{'data' 'model'})
                prettyplot
                sp = sp+1;
            end
            
            
        end
        subplot (4,3,1)
        title('2 days')
       
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