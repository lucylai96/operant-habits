function habitAnalysis(fig)
% PURPOSE: reproducing major habit findings
% Written by Lucy Lai

%close all

% plot settings
addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools'); % various plotting tools
addpath('./params'); % parameters

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
    
    %% REED 2001
    case 1
        agent.alpha_w = 0.1;
        agent.alpha_t = 0.1;
        agent.alpha_r = 0.1;
        agent.alpha_b = 5;
        sched.cmax = 100;
        
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
        
    %% decreasing action rates for increasing ratio and interval parameter schedules 
    case 2
        
        % double check if original studies were variable or fixed
        clear VI
        clear VR
        R = [20 40 80];
        I = [30 53 109];
        % delay = [16 32 64]; 
        sched.k = 1;
        sched.timeSteps = 3000;                            % how long were these 
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
            sched.times = contingency_generateVI(sched.I);     % pre-generated number of actions before reward
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
    
    %% simulating fits to Garr's data
    case 3
        map = brewermap(4,'*RdBu');
        temp = map(3,:);
        map(3,:) = map(4,:);
        map(4,:) = temp;  % swap colors so darker ones are fixed
        %% Garr's data
        type = {'FR' 'VR' 'FI' 'VI'};
        load('all_data_cleaned.mat'); % load data
        numrats = 8;
        sched.model = 2;
        sched.R = 20; % in deciseconds (VR/FR20*10 = 200)
        sched.I = 45; % in deciseconds  (VI/FI45*10 = 450)
        sp = 1;
        
        for typ = 1:length(type)  % for all sched types
            for r = 1:2             % for each rat
                sched.type = type{typ};
                load(strcat(sched.type,'_params_s',num2str(r),'.mat')); % get params and error
   
                sched.type = type{typ};
                if sched.type == 'FR'
                    data = FR20n(r);
                    col = map(1,:);
                elseif sched.type == 'VR'
                    data = VR20n(r);
                    col = map(2,:);
                elseif sched.type == 'FI'
                    data = FI45n(r);
                    col = map(3,:);
                elseif sched.type == 'VI'
                    data = VI45n(r);
                    col = map(4,:);
                end
                input.data = data;
                
                %% sanity checks (plots)
                % simulate valued 
                sched.deval = 0; sched.devalWin = 0;
                input.sched = sched;
                model_valued(r).sess = habitSimulate(params,input);
                
                % simulate devalued
                sched.deval = 1; sched.devalWin = 100;
                input.sched = sched;
                model_devalued(r).sess = habitSimulate(params,input); 
                
            
                % look at theta at end of deval and at end of test
                % look at the RPE at end of deval and at end of test
%                 for s = model_de                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  valued(r).sess(1).sched.devalsess % for each session
%                     % mean RPE (end of deval - end of train) should expect to be positive
%                     rpe(s,1,r) = mean(model_devalued(r).sess(s).rpe(model_devalued(r).sess(s).sched.trainEnd:model_devalued(r).sess(s).sched.devalEnd));
%                     % mean RPE (end of test - end of deval) should expect to be positive
%                     rpe(s,2,r) = mean(model_devalued(r).sess(s).rpe(model_devalued(r).sess(s).sched.devalEnd:end));
%                     
%                     % look at theta at time right before deval
%                     thetad1(:,:,r) = [model_devalued(r).sess(2).theta(:,:,model_devalued(r).sess(s).sched.trainEnd) model_devalued(r).sess(10).theta(:,:,model_devalued(r).sess(10).sched.trainEnd) model_devalued(r).sess(20).theta(:,:,model_devalued(r).sess(20).sched.trainEnd)];
%                     thetad2(:,:,r) = [model_devalued(r).sess(2).theta(:,:,model_devalued(r).sess(s).sched.devalEnd) model_devalued(r).sess(10).theta(:,:,model_devalued(r).sess(10).sched.devalEnd) model_devalued(r).sess(20).theta(:,:,model_devalued(r).sess(20).sched.devalEnd)];
%                     
%                     % sum change in theta (end of deval - end of train)
%                     d_theta1(s,:,r) = sum(model_devalued(r).sess(s).theta(:,:,model_devalued(r).sess(s).sched.devalEnd)-model_devalued(r).sess(s).theta(:,:,model_devalued(r).sess(s).sched.trainEnd));
%                     
%                     % sum change in theta (end of test - end of deval) positive
%                     d_theta2(s,:,r) = sum(model_devalued(r).sess(s).theta(:,:,end)-model_devalued(r).sess(s).theta(:,:,model_devalued(r).sess(s).sched.devalEnd));
%                     beta(s,1,r) = model_devalued(r).sess(s).beta(model_devalued(r).sess(s).sched.trainEnd);
%                     ecost(s,1,r) = model_devalued(r).sess(s).ecost(model_devalued(r).sess(s).sched.trainEnd);
%                     
%                     % change in pi_as
%                     pas(s,1:2,r) = mean(model_devalued(r).sess(s).pi_as(model_devalued(r).sess(s).sched.trainEnd-300:model_devalued(r).sess(s).sched.trainEnd,:)); % during train
%                     pas(s,3:4,r) = mean(model_devalued(r).sess(s).pi_as(model_devalued(r).sess(s).sched.devalEnd:end,:)); % after deval
%                 end
               rew = []; cst = []; rho = [];
                for s = 1:20
                    endRho(r,s) = model_devalued(r).sess(s).rho(model_devalued(r).sess(s).sched.trainEnd);
                    endCost(r,s) = model_devalued(r).sess(s).ecost(model_devalued(r).sess(s).sched.trainEnd);
                    
                    sz= length(movmean([input.data.session(s).training.reward_binned],200));
                    rew = [rew movmean([input.data.session(s).training.reward_binned],200)];
                    rho = [rho model_devalued(r).sess(s).rho(1:sz)];
                    cst = [cst model_devalued(r).sess(s).ecost(1:sz)];
                  
                end
                    
                figure; hold on;
                scatter(cst,rew, 20, brewermap(length(rew),'Blues'), 'filled');
                ylabel('avg reward')
                xlabel('policy complexity')
                prettyplot
                
%                  figure; hold on;
%                 scatter(cst,rho, 20, brewermap(length(rew),'Blues'), 'filled');
%                 ylabel('avg reward')
%                 xlabel('policy complexity')
%                 prettyplot
                
                
%                 figure(1); hold on;
%                 subplot 311;hold on;plot([model_devalued(r).sess(:).movActRate]);plot([model_devalued(r).sess(:).movOutRate]);
%                 legend('action rate', 'outcome rate')
%                 subplot 312;hold on;plot([model_devalued(r).sess(:).beta]);ylabel('\beta')
%                 plot([find([model_devalued(r).sess(:).beta]==0)' find([model_devalued(r).sess(:).beta]==0)'],[ylim],'k--','LineWidth',1)
%                 subplot 313;hold on;plot([model_devalued(r).sess(:).ecost]); ylabel('E[C(\pi)]')
%                 se = plot([find([model_devalued(r).sess(:).beta]==0)' find([model_devalued(r).sess(:).beta]==0)'],[ylim],'k--','LineWidth',1)
%                 legend(se,'sessions'); legend('boxoff'); 
%                 subprettyplot(3,1)
                %subplot 414;hold on;plot([model_devalued(r).sess(:).pa]);hold on;plot([model_devalued(r).sess(:).pi_as(:,2)])
                
                %subplot 414;hold on;plot(1./[model_devalued(r).sess(:).beta].* [model_devalued(r).sess(:).ecost])
                %figure; hold on;
                %plot([model_devalued(r).sess(:).ecost],[model_devalued(r).sess(:).rho],'o')
                
                %figure; %look at theta at time right before deval
                
                
                %                 figure (2);hold on;subplot 121; hold on;plot(beta,rpe(:,1),'o'); % big beta means bigger RPE during devaluation, RPE should increase through learning
                %                 subplot 122; hold on; bar([[model_valued(r).sess(:).postTestRate];[model_devalued(r).sess(:).postTestRate]]')
                %                 legend('valued','devalued')
            end % for each rat
            
            
            makePlotsData(data, model_devalued, model_valued)
                 
          
            
            
%             beta = squeeze(beta);ecost = squeeze(ecost);
%             rpe = squeeze(rpe(:,1,:));dtheta = squeeze(d_theta1(:,1,:));
%             beta(beta==0) = [];
%             ecost(ecost==0) = [];
%             rpe(rpe==0) = [];
%             dtheta(dtheta==0) = [];
%             figure(3);
%             subplot 131;hold on;plot(rpe,dtheta,'o','Color',col); ylabel('\Delta \theta(no press)');xlabel('RPE \delta');prettyplot
%             subplot 132;hold on;plot(beta,rpe,'o','Color',col); ylabel('RPE \delta');xlabel('\beta');prettyplot
%             subplot 133;hold on;plot((1./beta).*ecost,rpe,'o','Color',col); xlabel('1/\beta * C(\pi)');ylabel('RPE \delta');prettyplot
%       
            %rpe
            %d_theta1
            %beta
            %pas
            
            
            %% sanity checks (compare training)
            %MD.train_reward(typ,s) = mean(cat(1,model_devalued.outRate));
            %MD.train_lever(typ,s) = mean(cat(1,model_devalued.actRate));
            
            %MV.train_reward(typ,s) = mean(cat(1,model_valued.outRate));
            %MV.train_lever(typ,s) = mean(cat(1,model_valued.actRate));
            
            %DD.train_reward(typ,s) = sum(data.reward==1)/timeSteps(typ,s);
            %DD.train_lever(typ,s) = sum(data.lever==1)/timeSteps(typ,s);
            
            % compare test
            %MD.test_lever(typ,s) = mean(cat(1,model_devalued.actRateTest));
            %MV.test_lever(typ,s) = mean(cat(1,model_valued.actRateTest));
            
            %DD.test_lever(typ,s) = test_lever_rates.devalued(typ,s)/60; % per minute / 60 = seconds
            %DV.test_lever(typ,s) = test_lever_rates.valued(typ,s)/60;
            
            % plot change in lever press rate
            %                 figure(100); hold on;
            %                 subplot(4,3,sp); hold on;
            %                 bar(1,[DV.test_lever(typ,s) DD.test_lever(typ,s)]./DD.train_lever(typ,s)); % data (devalued, valued)
            %                 bar(2,[MV.test_lever(typ,s) MD.test_lever(typ,s)]./[MV.train_lever(typ,s) MD.train_lever(typ,s)]); % model (devalued, valued)
            %                 set(gca,'xtick',[1 2],'xticklabel',{'data' 'model'})
            %                 prettyplot
            %
            %                 figure(200); hold on;
            %                 subplot(4,3,sp); hold on;
            %                 bar(1,[DV.test_lever(typ,s) DD.test_lever(typ,s)]); % data (devalued, valued)
            %                 bar(2,[MV.test_lever(typ,s) MD.test_lever(typ,s)]); % model (devalued, valued)
            %
            %
            %                 set(gca,'xtick',[1 2],'xticklabel',{'data' 'model'})
            %                 prettyplot
            %                 sp = sp+1;
            
        end % type
        
        
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


function makePlotsData(data,model,modelv)
%% init color map
map = brewermap(4,'*RdBu');
temp = map(3,:);
map(3,:) = map(4,:);
map(4,:) = temp;  % swap colors so darker ones are fixed
set(0, 'DefaultAxesColorOrder', map) % first three rows

load('all_data_cleaned.mat') % actual data
sched = model(2).sess.sched;
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
sess = size(model(2).sess,2); % get number of sessions
% for i = 1:length(model) % for each subject
%     for s = 1:sess % for each session
%         if ismember(s,sched.devalsess)
%             %model(i).sess(s).timeSteps = size(model(i).sess(s).a,2); % session length, timesteps;
%             model(i).sess(s).outRate = sum(model(i).sess(s).x(1:sched.devalEnd)==2)/sched.devalEnd;
%             model(i).sess(s).actRate = sum(model(i).sess(s).a(1:sched.devalEnd)==2)/sched.devalEnd;
%             model(i).sess(s).avgRew = mean(model(i).sess(s).rho(1:sched.devalEnd));
%             model(i).sess(s).avgBeta = mean(model(i).sess(s).beta(1:sched.devalEnd));
%             model(i).sess(s).postTestRate = sum(model(i).sess(s).a(sched.devalEnd:end)==2)/sched.testWin;
%             model(i).sess(s).preTestRate = sum(model(i).sess(s).a(sched.trainEnd-sched.testWin:sched.trainEnd)==2)/sched.testWin;
%         else
%             model(i).sess(s).timeSteps = size(model(i).sess(s).a,2); % session length, timesteps;
%             timeSteps = model(i).sess(s).timeSteps;
%             model(i).sess(s).outRate = sum(model(i).sess(s).x==2)/sched.trainEnd;
%             model(i).sess(s).actRate = sum(model(i).sess(s).a==2)/sched.trainEnd;
%             model(i).sess(s).avgRew = mean(model(i).sess(s).rho);
%             model(i).sess(s).avgBeta = mean(model(i).sess(s).beta);
%         end
%     end
% end

% if it's a devaluation session, separate that data out into model.test
figure; hold on;
%subplot 211; hold on;
% for r = 1:length(model)
%     outRate(r,:) = cat(2,model(r).sess.outRate);
% end
% plot(outRate,'LineWidth',2,'Color',col);
% %errorbar(mean(outRate),sem(outRate,1),'LineWidth',2,'Color',col);
% prettyplot
% ylabel('outcome rate')
% legend([sched.type],'FontSize',15); legend('boxoff')
% axis([0 22 0 1])
% plot([2.5 10.5 20.5;2.5 10.5 20.5],[ylim' ylim' ylim'],'k--')
% 
% subplot 312; hold on;
for r = 1:length(model)
    actRate(r,:) = cat(2,model(r).sess.actRate);
end
errorbar(mean(data.actRate),sem(data.actRate,1),'LineWidth',2,'Color',[0 0 0]);
%plot(actRate,'LineWidth',2,'Color',col);
errorbar(mean(actRate),sem(actRate,1),'LineWidth',2,'Color',col);
xlabel('session #')
ylabel('action rate')
axis([0 22 0 1])
plot([2.5 10.5 20.5;2.5 10.5 20.5],[ylim' ylim' ylim'],'k--')
prettyplot

% subplot 212;  hold on;
% for r = 1:length(model)
%     beta(r,:) = cat(2,model(r).sess.avgBeta);
% end
% errorbar(mean(beta),sem(beta,1),'LineWidth',2,'Color',col);
% axis([0 22 ylim])
% xlabel('session #')
% ylabel('\beta')
% prettyplot

set(gcf, 'Position',  [500, 500, 800, 300])

%% devaluation
for d = 1:3
    for r = 1:length(model)
        test(r,:,d) = [modelv(r).sess([sched.devalsess(d)]).postTestRate model(r).sess([sched.devalsess(d)]).postTestRate];
    end
end

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
equalabscissa(1,2)
subprettyplot(1,2)

set(gcf, 'Position',  [500, 150, 800, 300])

%% individual

 % individual animals
        figure; hold on;
        k = [1 5 9]; % subplot indices
        for d = 1:3
            subplot(3,1,d); hold on;
            b = bar(test(:,:,d),'FaceColor',map(1,:));
            b(2).FaceColor = [1 1 1];
            b(2).EdgeColor = map(1,:);
            b(2).LineWidth = 2;
%             
%             subplot(3,4,k(d)+1); hold on;
%             b  = bar(VR20.test(:,:,d),'FaceColor',map(2,:));
%             b(2).FaceColor = [1 1 1];
%             b(2).EdgeColor = map(2,:);
%             b(2).LineWidth = 2;
%             
%             subplot(3,4,k(d)+2); hold on;
%             b = bar(FI45.test(:,:,d),'FaceColor',map(3,:));
%             b(2).FaceColor = [1 1 1];
%             b(2).EdgeColor = map(3,:);
%             b(2).LineWidth = 2;
%              
%             subplot(3,4,k(d)+3); hold on;
%             b = bar(VI45.test(:,:,d),'FaceColor',map(4,:));
%             b(2).FaceColor = [1 1 1];
%             b(2).EdgeColor = map(4,:);
%             b(2).LineWidth = 2;
            
        end
        equalabscissa(3,1)
%         subplot 341;  title('FR20'); ylabel('session 2');legend('valued','devalued'); legend('boxoff');
%         subplot 342;  title('VR20'); 
%         subplot 343;  title('FI45')
%         subplot 344;  title('VI45')
%         subplot 345; ylabel('session 10');
%         subplot 349; ylabel('session 20');
%         subplot(3,4,12)
%         ylabel('lever presses/sec'); xlabel('rat #')
%         subprettyplot(3,4)
end