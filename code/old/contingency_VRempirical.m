% function describing behavior of an agent -
% agent samples the

function contingency_VRempirical
addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools/');
% here, we verify that lambda_o is 1/R * lambda_a


% %% VI schedule
% % first, compute a vector of required outcome times (not contingent on actions)
% int = 10; % avg required interval (VI)
% [VI.I, VI.waitTimes] = contingency_generateVI(int);
%
map = brewermap(9,'Blues');
set(0, 'DefaultAxesColorOrder', map([3,5,7,9],:)) % first three rows


%% new agent
T = 2000; % timesteps
ratio = [2 5 10 20]; % rewarded every R actions

la = linspace(0.1,1,20); % action rate (presses per second)
plt = 0;
for r = 1:length(ratio)
    R = ratio(r);
    for a = 1:length(la)
        lambda_a = la(a);
      %  A = exprnd(lambda_a*ones(1,T)); % generate vector of actions using agent's probability of action (assuming poisson responding)
        A = binornd(1,lambda_a*ones(1,T)); % generate vector of actions using agent's probability of action (assuming poisson responding)
        aWaitTimes = diff(find(A==1)); % agent's action wait times
        if plt ==1
            figure; hold on; histogram(aWaitTimes,50); title(strcat('agent wait times, \lambda_a=',num2str(lambda_a))); prettyplot;
        end
        mean(aWaitTimes);
        
        %% FR world | fixed ratio
        t = 1;
        agent.O_FR = zeros(1,T); agent.O_FR(1) = 1;
        
        while t<T
            % step through, take actions
            % A is action vector
            % t is time
            % VR_k(t) keep track of the actions at any given time
            % agent.O_VR is outcomes
            % VR.numActions is actions needed for a reward...
            [agent.O_FR] = contingency_FRworld(A, agent.O_FR, t, R); % outputs every time an agent saw a reward
            t = t+1;
        end
        
        
        FR.outTimes = find(agent.O_FR==1);
        FR.waitTimes = diff(FR.outTimes); % this is my observed outcome wait times, lambda_o
        if plt==1
            figure; hold on; histogram(FR.waitTimes,50); title(strcat('FR outcomes wait times')); prettyplot;
            disp('FR wait time mean:'); disp(mean(FR.waitTimes))
        end
        FR.waitTimeMean(r,a) = mean(FR.waitTimes);
        FR.lambda_o(r,a) = sum(agent.O_FR==1)/T; % estimated outcome rate. CONFIRMED lambda_o = 1/R * lambda_a;
        
        %% FR: now we ask what the unconditional and conditional entropies are:
       
        % conditional: the entropy of the action->outcome times
        % this is NOT the action rate, lambda_a
        % for every action, get time from action to outcome
         k = 2;
        numAct = zeros(1,length(FR.outTimes)); %this is for counting the number of actions before a reward
        ATO = NaN(1,T); % A->O; conditional dist
        RTO = NaN(1,T); % random points -> O; unconditional/basal dist
        
        rand_points = sort(datasample([1:T],sum(A),'Replace',true));
        RT = zeros(1,T);
        RT(rand_points) = 1;
        
        for t = 1:T
            if  k<length(FR.outTimes)&&t<=FR.outTimes(k)
                if A(t) == 1
                    ATO(t) = FR.outTimes(k)-t;% calculate time-to-next outcome, Action To Outcome
                    numAct(k) = numAct(k)+1;% how many actions before next outcome
                end
                if RT(t) == 1
                    RTO(t) = FR.outTimes(k)-t;% calculate time-to-next outcome, rand To Outcome
                end
            elseif k<length(FR.outTimes) %if current time is not less than the time of next reward
                
                k = k+1; %go to next reward time
                if A(t) == 1
                    ATO(t) = FR.outTimes(k)-t;% calculate time-to-next outcome, Action To Outcome
                    numAct(k) = numAct(k)+1;% how many actions before next outcome
                end
                
                if RT(t) == 1
                    RTO(t) = FR.outTimes(k)-t;% calculate time-to-next outcome, rand To Outcome
                end
                
                
            end
        end
        
        if plt ==1
            % avg number of actions:
            figure; hold on; histogram(numAct,50); title(strcat('number of actions before reward')); prettyplot;
            disp('mean num actions before reward:'); disp(mean(numAct));
            
            % conditional distribution O|A:
            figure; hold on; histogram(ATO,50); title(strcat('H_{A|O} a->o wait times')); prettyplot;
            disp('A->O wait time mean:'); disp(nanmean(ATO))% mean... i wonder if this can be parameterized (it doesn't look like any exponential distribution...)
           
            % unconditional distribution O:
            figure; hold on; histogram(RTO,50); title(strcat('H_{O} rand->o wait times')); prettyplot;
            disp('R->O wait time mean:'); disp(nanmean(RTO))% mean... i wonder if this can be parameterized (it doesn't look like any exponential distribution...)
      
        end
        
        
        
        FR.meanNumAct(r,a) = mean(numAct);
        FR.meanATO(r,a) = nanmean(ATO);
        
        %FR.HOA(r,a) = calc_entropy(ATO);
        FR.HOA(r,a) = log(ratio(r)/lambda_a);
        %if a < 5
        %figure(6); hold on; histogram(ATO,20)
        %pause(1)
        %end
        
        % unconditional: the entropy of the observed outcome times distribution = FR.waitTimes
        % this is parameterized by lambda_o
       % FR.HO(r,a) = 1-log(FR.lambda_o(r,a)); %1-log(lambda_o) = 1-log(lambda_a/R)
        
        %FR.HO(r,a) = calc_entropy(RTO);
        FR.HO(r,a) = 1-log(lambda_a/R);
        %% VR world | variable ratio
        % verify that lambda_o is 1/R * lambda_a
        %R = 10; %
        p = 1/R; % 1/10 percent chance of reward after every press
        
        t = 1;
        agent.O_VR = zeros(1,T);
        
        while t<T
            % step through, take actions
            % A is action vector
            % t is time
            % VR_k(t) keep track of the actions at any given time
            % agent.O_VR is outcomes
            % VR.numActions is actions needed for a reward...
            
            [agent.O_VR] = contingency_VRworld(A, agent.O_VR, t, p); %outputs every time an agent saw a reward
            t = t+1;
        end
        
        
        VR.outTimes = find(agent.O_VR==1);
        VR.waitTimes = diff(VR.outTimes); % this is my observed outcome wait times, lambda_o
        if plt ==1
            figure; hold on; histogram(VR.waitTimes,50); title(strcat('VR outcomes wait times')); prettyplot;
            disp('VR wait time mean:'); disp(mean(VR.waitTimes))
        end
        
        VR.waitTimeMean(r,a) = mean(VR.waitTimes);
        VR.lambda_o(r,a) = sum(agent.O_VR==1)/T; % estimated outcome rate. CONFIRMED lambda_o = 1/R * lambda_a;
        
        %% VR: now we ask what the unconditional and conditional entropies are:
       
        % conditional: the entropy of the action->outcome times
        % this is NOT the action rate, lambda_a
        % for every action, get time from action to outcome
        k = 2;
        numAct = zeros(1,length(VR.outTimes)); %this is for counting the number of actions before a reward
        ATO = NaN(1,T); % A->O; conditional dist
        RTO = NaN(1,T); % random points -> O; unconditional/basal dist
        
        rand_points = sort(datasample([1:T],sum(A),'Replace',true));
        RT = zeros(1,T);
        RT(rand_points) = 1;
        
        for t = 1:T
            if  k<length(VR.outTimes)&&t<=VR.outTimes(k)
                if A(t) == 1
                    ATO(t) = VR.outTimes(k)-t;% calculate time-to-next outcome, Action To Outcome
                    numAct(k) = numAct(k)+1;% how many actions before next outcome
                end
                if RT(t) == 1
                    RTO(t) = VR.outTimes(k)-t;% calculate time-to-next outcome, rand To Outcome
                end
            elseif k<length(VR.outTimes) %if current time is not less than the time of next reward
                
                k = k+1; %go to next reward time
                if A(t) == 1
                    ATO(t) = VR.outTimes(k)-t;% calculate time-to-next outcome, Action To Outcome
                    numAct(k) = numAct(k)+1;% how many actions before next outcome
                end
                
                if RT(t) == 1
                    RTO(t) = VR.outTimes(k)-t;% calculate time-to-next outcome, rand To Outcome
                end
                
                
            end
        end
        
        if plt ==1
            % avg number of actions:
            figure; hold on; histogram(numAct,50); title(strcat('number of actions before reward')); prettyplot;
            disp('mean num actions before reward:'); disp(mean(numAct));
            
            % conditional distribution O|A:
            figure; hold on; histogram(ATO,50); title(strcat('H_{A|O} a->o wait times')); prettyplot;
            disp('A->O wait time mean:'); disp(nanmean(ATO))% mean... i wonder if this can be parameterized (it doesn't look like any exponential distribution...)
           
            % unconditional distribution O:
            figure; hold on; histogram(RTO,50); title(strcat('H_{O} rand->o wait times')); prettyplot;
            disp('R->O wait time mean:'); disp(nanmean(RTO))% mean... i wonder if this can be parameterized (it doesn't look like any exponential distribution...)
      
        end
        
        
        VR.meanNumAct(r,a) = mean(numAct);
        VR.meanATO(r,a) = nanmean(ATO);
        % simulate an exp with that mean
        % N = length(ATO);
        % A = binornd(1,1/nanmean(ATO)*ones(1,N));
        % WT = diff(find(A==1)); % agent's action wait times
        % histogram(WT,50); title(strcat('Wait times')); prettyplot;
        % histogram(A,50);
        
        %VR.HOA(r,a) = calc_entropy(ATO);
        
        %figure(6); hold on; histogram(ATO,20)
        %pause(1)
        
         % unconditional: the entropy of the observed outcome times distribution = VR.waitTimes
        % this is parameterized by lambda_o
        %VR.HO(r,a) = 1-log(VR.lambda_o(r,a)); %1-log(lambda_o)
        
        %VR.HO(r,a) = calc_entropy(RTO);
        VR.HO(r,a) = 1-log(lambda_a/ratio(r));
        
        
        temp = ((-lambda_a/ratio(r))*(log(ratio(r))))/(lambda_a-(lambda_a/ratio(r)));
        VR.HOA(r,a) = exp(temp)*VR.HO(r,a);
        VR.C(r,a) = 1-exp(temp);
    end
end

%% test
%FR.HO = 1-log(FR.lambda_o);
%VR.HO = 1-log(FR.lambda_o);
%% calculate information
FR.I = FR.HO-FR.HOA;
VR.I = VR.HO-VR.HOA;



%% plots
figure; hold on;
subplot 221; hold on;
plot(la,VR.lambda_o,'LineWidth', 1.5)
% plot(a',beta','LineWidth'4, 1.5)
% text(1.5,90, 'less habit','FontSize',14)
% text(1.5,10, 'more habit','FontSize',14)
xlabel('action rate \lambda_a')
ylabel('outcome rate \lambda_o')
% ylabel('resource constraint \beta')
axis square
% axis([0 2 0 100])
legend('VR2','VR5','VR10','VR20')
legend('boxoff')
prettyplot


subplot 222; hold on;
plot(la',VR.HO','--','LineWidth', 1.5);
plot(la',VR.HOA','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('conditional entropy H_{O|A}')
legend('VR2','VR5','VR10','VR20','Location','southeast')
legend('boxoff')
axis([0 1 0 5])
axis square
prettyplot

% mutual information between each action and outcome grows as a function of
% the agent's action rate. this is because doing more actions per unit time
% gives you more information about when outcome will appear based on ratio
% parameter
subplot 223
plot(la',VR.I','LineWidth', 1.5)
axis([0 1 0 4])
xlabel('action rate \lambda_a')
ylabel('I(O;A)')
axis square
prettyplot

% VR schedules, A-O contingency is strong, because the faster they respond,
% the sooner on average a reinforcer is delivered.
subplot 224
VR.C = VR.I./VR.HO;
plot(la',VR.C','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('A-O contingency')
axis([0 1 0 1])
axis square
prettyplot

suptitle('VR')
save('VR_empirical.mat','VR')

%% FR
figure; hold on;
subplot 221; hold on;
plot(la,FR.lambda_o,'LineWidth', 1.5)
% plot(a',beta','LineWidth'4, 1.5)
% text(1.5,90, 'less habit','FontSize',14)
% text(1.5,10, 'more habit','FontSize',14)
xlabel('action rate \lambda_a')
ylabel('outcome rate \lambda_o')
% ylabel('resource constraint \beta')
axis square
% axis([0 2 0 100])
legend('FR2','FR5','FR10','FR20')
legend('boxoff')
prettyplot


subplot 222; hold on;
plot(la',FR.HO','--','LineWidth', 1.5);
plot(la',FR.HOA','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('conditional entropy H_{O|A}')
legend('FR2','FR5','FR10','FR20','Location','southeast')
legend('boxoff')
axis([0 1 0 5])
axis square
prettyplot

% mutual information between each action and outcome grows as a function of
% the agent's action rate. this is because doing more actions per unit time
% gives you more information about when outcome will appear based on ratio
% parameter
subplot 223
plot(la',FR.I','LineWidth', 1.5)
axis([0 1 0 4])
xlabel('action rate \lambda_a')
ylabel('I(O;A)')
axis square
prettyplot

% VR schedules, A-O contingency is strong, because the faster they respond,
% the sooner on average a reinforcer is delivered.
subplot 224
FR.C = FR.I./FR.HO;
plot(la',FR.C','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('A-O contingency')
axis([0 1 0 1])
axis square
prettyplot


suptitle('FR')

save('FR_empirical.mat','FR')

%% other metrics to look at
figure; hold on;
map = brewermap(9,'Blues');
set(0, 'DefaultAxesColorOrder', map([3,5,7,9],:)) % first three rows
subplot 221; plot(la,VR.meanNumAct,'LineWidth', 1.5);ylabel('mean #A until reward');axis square;prettyplot; % mean number of actions till reward
% mean number of actions till reward
title ('VR')
subplot 222; plot(la,VR.meanATO,'LineWidth', 1.5);;ylabel('mean time A->O');axis square;prettyplot; % mean number of actions till reward
%mean of the action to outcome distribtion (conditional entropy)
legend('VR2','VR5','VR10','VR20')
legend('boxoff')
title ('VR')

subplot 223; plot(la,FR.meanNumAct,'LineWidth', 1.5);xlabel('\lambda_a (actions/sec)');ylabel('mean #A until reward');axis square;prettyplot; % mean number of actions till reward
title ('FR')
subplot 224; plot(la,FR.meanATO,'LineWidth', 1.5);xlabel('\lambda_a (actions/sec)');ylabel('mean time A->O (s)');axis square;prettyplot; % mean number of actions till reward
legend('FR2','FR5','FR10','FR20')
legend('boxoff')
%mean of the action to outcome distribtion (conditional entropy)
title ('FR')
end