function contingency_VIempirical
% here, we verify that lambda_o is 1/T
addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools/');

map = brewermap(9,'Reds');
set(0, 'DefaultAxesColorOrder', map([3,5,7,9],:)) % first three rows


%% new agent
T = 2000; % timesteps
int = [2 5 10 20]; % rewarded every t seconds

la = linspace(0.1,1,20); % action rate (presses per second)

plt = 0;
for i = 1:length(int)
    I = int(i);
    % generate wait times
    [VI.intTimes] = contingency_generateVI(I); 
    if plt ==1
        figure; histogram(VI.intTimes,50)
        title('interval wait times (VI)') % mean(VI.waitTimes)
        prettyplot
    end
    
    for a = 1:length(la)
        lambda_a = la(a);
        A = binornd(1,lambda_a*ones(1,T)); % generate vector of actions using agent's probability of action (assuming poisson responding)
        aWaitTimes = diff(find(A==1)); % agent's action wait times
        if plt ==1
            figure; hold on; histogram(aWaitTimes,50); title(strcat('agent wait times, \lambda_a=',num2str(lambda_a))); prettyplot;
        end
        mean(aWaitTimes);
        
        %% FI world | fixed interval
        t = 1;
        agent.O_FI = zeros(1,T); agent.O_FI(1) = 1;
        k = 1;
        while t<T
            % step through, take actions
            % A is action vector
            % t is time
            % agent.O_FI is outcomes
            
            [agent.O_FI,k] = contingency_FIworld(A, agent.O_FI, t, k, I); % outputs every time an agent saw a reward
            t = t+1;
        end
        
        
        FI.outTimes = find(agent.O_FI==1);
        FI.waitTimes = diff(FI.outTimes); % this is my observed outcome wait times, lambda_o
        if plt==1
            figure; hold on; histogram(FI.waitTimes,50); title(strcat('FI outcomes wait times')); prettyplot;
            disp('FI wait time mean:'); disp(mean(FI.waitTimes))
        end
        FI.waitTimeMean(i,a) = mean(FI.waitTimes);
        FI.lambda_o(i,a) = sum(agent.O_FI==1)/T; % estimated outcome rate. CONFIRMED lambda_o = 1/I;
        
        %% FI: now we ask what the unconditional and conditional entropies are:
        
        % conditional: the entropy of the action->outcome times (should be
        % around same as unconditional)
        % this is NOT the action rate, lambda_a
        % for every action, get time from action to outcome
        k = 2;
        numAct = zeros(1,length(FI.outTimes)); %this is for counting the number of actions before a reward
        ATO = NaN(1,T); % A->O; conditional dist
        RTO = NaN(1,T); % random points -> O; unconditional/basal dist
        
        rand_points = sort(datasample([1:T],sum(A),'Replace',false));
        R = zeros(1,T);
        R(rand_points) = 1;
        
        for t = 1:T
            if  k<length(FI.outTimes)&&t<=FI.outTimes(k)
                if A(t) == 1
                    ATO(t) = FI.outTimes(k)-t;% calculate time-to-next outcome, Action To Outcome
                    numAct(k) = numAct(k)+1;% how many actions before next outcome
                end
                if R(t) == 1
                    RTO(t) = FI.outTimes(k)-t;% calculate time-to-next outcome, rand To Outcome
                end
            elseif k<length(FI.outTimes) %if current time is not less than the time of next reward
                
                k = k+1; %go to next reward time
                if A(t) == 1
                    ATO(t) = FI.outTimes(k)-t;% calculate time-to-next outcome, Action To Outcome
                    numAct(k) = numAct(k)+1;% how many actions before next outcome
                end
                
                if R(t) == 1
                    RTO(t) = FI.outTimes(k)-t;% calculate time-to-next outcome, rand To Outcome
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
            figure; hold on; histogram(ATO,50); title(strcat('H_{O} rand->o wait times')); prettyplot;
            disp('R->O wait time mean:'); disp(nanmean(RTO))% mean... i wonder if this can be parameterized (it doesn't look like any exponential distribution...)
      
        end
        
        
        FI.meanNumAct(i,a) = mean(numAct);
        FI.meanATO(i,a) = nanmean(ATO);
        
       %FI.HOA(i,a) = calc_entropy(ATO);
        %(a*FI.HO)./(a+exp(-a))
        
        % unconditional: the entropy of the observed outcome times distribution = FI.waitTimes
        % this is parameterized by lambda_o(??)
        
        % ho is distribution of intervals to outcome from the *same
        % number* of randomly chosen moments in time as the a-->o
        
        
      %  FI.HO(i,a) = 1-log(FI.lambda_o(i,a)); %1-log(lambda_o) = 1-log(1/I)
        %FI.HO(i,a) = calc_entropy(RTO);
        FI.HO(i,a) = 0;
       
        temp = ((-1/int(i))*(log(lambda_a)-log(1/int(i))))/(lambda_a-(1/int(i)));
        FI.HOA(i,a)  = exp(temp)*FI.HO(i,a);
        FI.C(i,a) =  1-exp(temp);
        
        
        %% VR world | variable ratio
        t = 1;
        agent.O_VI = zeros(1,T); agent.O_VI(1) = 1;
        k = 1;
        while t<T
            % step through, take actions
            % A is action vector
            % t is time
            % VI_k(t) keep track of the actions at any given time
            % agent.O_VI is outcomes
            % VI.waitTimes is actions needed for a reward...
            if k>size(VI.intTimes)
                keyboard
            end
            [agent.O_VI, k] = contingency_VIworld(A, agent.O_VI, t, k, VI.intTimes);
            t = t+1;
        end
        
        
        VI.outTimes = find(agent.O_VI==1);
        VI.waitTimes = diff(VI.outTimes); % this is my observed outcome wait times, lambda_o
        if plt ==1
            figure; hold on; histogram(VI.waitTimes,50); title(strcat('VI outcomes wait times')); prettyplot;
            disp('VI wait time mean:'); disp(mean(VI.waitTimes))
        end
        
        VI.waitTimeMean(i,a) = mean(VI.waitTimes);
        VI.lambda_o(i,a) = sum(agent.O_VI==1)/T; % estimated outcome rate. CONFIRMED lambda_o = 1/R * lambda_a;
        
        %% VI: now we ask what the unconditional and conditional entropies are:
        
        % conditional: the entropy of the action->outcome times
        % this is NOT the action rate, lambda_a
        % for every action, get time from action to outcome
        k = 2;
        numAct = zeros(1,length(VI.outTimes)); %this is for counting the number of actions before a reward
        ATO = NaN(1,T); % A->O; conditional dist
        RTO = NaN(1,T); % random points -> O; unconditional/basal dist
        
        rand_points = sort(datasample([1:T],sum(A),'Replace',false));
        R = zeros(1,T);
        R(rand_points) = 1;
        
        for t = 1:T
            if  k<length(VI.outTimes)&&t<=VI.outTimes(k)
                if A(t) == 1
                    ATO(t) = VI.outTimes(k)-t;% calculate time-to-next outcome, Action To Outcome
                    numAct(k) = numAct(k)+1;% how many actions before next outcome
                end
                if R(t) == 1
                    RTO(t) = VI.outTimes(k)-t;% calculate time-to-next outcome, rand To Outcome
                end
            elseif k<length(VI.outTimes) %if current time is not less than the time of next reward
                
                k = k+1; %go to next reward time
                if A(t) == 1
                    ATO(t) = VI.outTimes(k)-t;% calculate time-to-next outcome, Action To Outcome
                    numAct(k) = numAct(k)+1;% how many actions before next outcome
                end
                
                if R(t) == 1
                    RTO(t) = VI.outTimes(k)-t;% calculate time-to-next outcome, rand To Outcome
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
            figure; hold on; histogram(ATO,50); title(strcat('H_{O} rand->o wait times')); prettyplot;
            disp('R->O wait time mean:'); disp(nanmean(RTO))% mean... i wonder if this can be parameterized (it doesn't look like any exponential distribution...)
      
        end
        
        
        VI.meanNumAct(i,a) = mean(numAct);
        VI.meanATO(i,a) = nanmean(ATO);
        % simulate an exp with that mean
        % N = length(ATO);
        % A = binornd(1,1/nanmean(ATO)*ones(1,N));
        % WT = diff(find(A==1)); % agent's action wait times
        % histogram(WT,50); title(strcat('Wait times')); prettyplot;
        % histogram(A,50);
        
        %VI.HOA(i,a) = calc_entropy(ATO);
        
        
        % unconditional: the entropy of the observed outcome times distribution = VR.waitTimes
        % this is parameterized by lambda_o
        
        % ho is distribution of intervals to outcome from the *same
        % number* of randomly chosen moments in time as the a-->o
       
       % VI.HO(i,a) = 1-log(VI.lambda_o(i,a)); %1-log(lambda_o) = 1-log(1/I)
        %VI.HO(i,a) = calc_entropy(RTO);
        VI.HO(i,a) = 1-log(1/int(i));
        
        temp = ((-1/int(i))*(log(lambda_a)-log(1/int(i))))/(lambda_a-(1/int(i)));
        VI.HOA(i,a) = exp(temp)*VI.HO(i,a);
        VI.C(i,a) = 1-exp(temp);
        %[VI.Cyx(r,a),VI.MI(r,a),Hy(r,a),Hx(r,a),Hxy(r,a)] = JntDistContingN(,,bs(a),bs(a));
       
        
        
    end
end

%% calculate information
FI.I = FI.HO-FI.HOA;
VI.I = VI.HO-VI.HOA;

%% plots
figure; hold on;
subplot 221; hold on;
plot(la,VI.lambda_o,'LineWidth', 1.5)
% plot(a',beta','LineWidth'4, 1.5)
% text(1.5,90, 'less habit','FontSize',14)
% text(1.5,10, 'more habit','FontSize',14)
xlabel('action rate \lambda_a')
ylabel('outcome rate \lambda_o')
% ylabel('resource constraint \beta')
axis square
% axis([0 2 0 100])
legend('VI2','VI5','VI10','VI20')
legend('boxoff')
prettyplot


subplot 222; hold on;
plot(la',VI.HO','--','LineWidth', 1.5);
plot(la',VI.HOA','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('conditional entropy H_{O|A}')
legend('VI2','VI5','VI10','VI20','Location','southeast')
legend('boxoff')
axis([0 1 0 5])
axis square
prettyplot

% mutual information between each action and outcome grows as a function of
% the agent's action rate. this is because doing more actions per unit time
% gives you more information about when outcome will appear based on ratio
% parameter
subplot 223
plot(la',VI.I','LineWidth', 1.5)
axis([0 1 0 4])
xlabel('action rate \lambda_a')
ylabel('I(O;A)')
axis square
prettyplot

% VI schedules, A-O contingency is strong, because the faster they respond,
% the sooner on average a reinforcer is delivered.
subplot 224
VI.C = VI.I./VI.HO;
plot(la',VI.C','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('A-O contingency')
axis([0 1 0 1])
axis square
prettyplot

suptitle ('VI')

save('VI_empirical.mat','VI')

%% FI
figure; hold on;
subplot 221; hold on;
plot(la,FI.lambda_o,'LineWidth', 1.5)
% plot(a',beta','LineWidth'4, 1.5)
% text(1.5,90, 'less habit','FontSize',14)
% text(1.5,10, 'more habit','FontSize',14)
xlabel('action rate \lambda_a')
ylabel('outcome rate \lambda_o')
% ylabel('resource constraint \beta')
axis square
% axis([0 2 0 100])
legend('FI1','FI2','FI5','FI10','FI20')
legend('boxoff')
prettyplot


subplot 222; hold on;
plot(la',FI.HO','--','LineWidth', 1.5);
plot(la',FI.HOA','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('conditional entropy H_{O|A}')
legend('FI2','FI5','FI10','FI20','Location','southeast')
legend('boxoff')
axis([0 1 0 5])
axis square
prettyplot

% mutual information between each action and outcome grows as a function of
% the agent's action rate. this is because doing more actions per unit time
% gives you more information about when outcome will appear based on ratio
% parameter
subplot 223
plot(la',FI.I','LineWidth', 1.5)
axis([0 1 0 4])
xlabel('action rate \lambda_a')
ylabel('I(O;A)')
axis square
prettyplot

% VI schedules, A-O contingency is strong, because the faster they respond,
% the sooner on average a reinforcer is delivered.
subplot 224
C = FI.I./FI.HO;
plot(la',FI.C','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('A-O contingency')
axis([0 1 0 1])
axis square
prettyplot
suptitle ('FI')

%% other metrics to look at
figure; hold on;
map = brewermap(9,'Reds');
set(0, 'DefaultAxesColorOrder', map([3,5,7,9],:)) % first three rows
subplot 221; plot(la,VI.meanNumAct,'LineWidth', 1.5);ylabel('mean #A until reward');axis square;prettyplot; % mean number of actions till reward
% mean number of actions till reward
title ('VI')
subplot 222; plot(la,VI.meanATO,'LineWidth', 1.5);ylabel('mean time A->O (s)');axis square;prettyplot; % mean number of actions till reward
%mean of the action to outcome distribtion (conditional entropy)
legend('VI2','VI5','VI10','VI20')
legend('boxoff')
title ('VI')

subplot 223; plot(la,FI.meanNumAct,'LineWidth', 1.5);xlabel('\lambda_a (actions/sec)');ylabel('mean #A until reward');axis square;prettyplot; % mean number of actions till reward
title ('FI')
subplot 224; plot(la,FI.meanATO,'LineWidth', 1.5);xlabel('\lambda_a (actions/sec)');ylabel('mean time A->O');axis square;prettyplot; % mean number of actions till reward
legend('FI2','FI5','FI10','FI20')
legend('boxoff')
%mean of the action to outcome distribtion (conditional entropy)
title ('FI')