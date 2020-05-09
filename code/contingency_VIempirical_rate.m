function contingency_VIempirical_rate
% here, we verify that lambda_o is 1/T
addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools/');

map = brewermap(9,'Reds');
set(0, 'DefaultAxesColorOrder', map([3,5,7,9],:)) % first three rows


%% new agent
T = 7200; % timesteps
int = [2 5 10 20]; % rewarded every t seconds
%int = [10]; % rewarded every t seconds

la = linspace(0.01,1,20); % action rate (presses per second)
%la = .9;

bs = 20*ones(1,length(la));%round(sort(linspace(5,50,20),'descend'));
bs = round(sort(linspace(5,50,20),'descend'));
plt = 0;
for i = 1:length(int)
    I = int(i);
    % generate wait times
    [VI.intTimes] = contingency_generateVI(I); size(VI.intTimes)
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
        
        %% FI: now we ask what the unconditional and conditional rate entropies are:
        
        % for each outcome that occured, calculate (1) the IOI and (2) #actions that occured
        % unconditional distribution
        IOI = FI.waitTimes; %= diff(FI.outTimes); (1) these are the IOIs. confirm that lo=la for la<lo 
        
        numAct = zeros(1,length(IOI));
        IAI = zeros(1,length(IOI));
        for o = 1:length(IOI) % for each IOI, calculate the <IAI>
            numAct(o) = sum(A(FI.outTimes(o):FI.outTimes(o+1)));% how many actions between outcomes 
            IAI(o) = IOI(o)/numAct(o);
        end 
        
        est_la = 1./IAI; % estimated lambda_a
        est_lo = 1./IOI; % estimated lambda_o
        if plt==1
             figure; scatter(IOI,IAI); scatter(est_la,est_lo);
        end
        
        % entropy of IOI H(IOI)
        FI.HO(i,a) = calc_entropy(IOI);
        % FI.HO(i,a) = 1-log(FI.lambda_o(i,a));
        %VI.HA(i,a) = entropy(IAI);
     %   FI.HOA(i,a) = entropy([IAI;IOI]');
        
        % conditional: the entropy of the action->outcome times (should be
        % around same as unconditional)
        % this is NOT the action rate, lambda_a
        % for every action, get time from action to outcome
       % IOI_IAI = NaN(1,T); % IOI; conditional dist, inter-outcome interval
        % take only the IOIs that 
        
        [FI.Cyx(i,a),FI.MI(i,a),Hy(i,a),Hx(i,a),Hxy(i,a)] = JntDistContingN(IAI,IOI,bs(a),bs(a));
        FI.HOA(i,a) = Hxy(i,a)-Hx(i,a);
        
        FI.HO_fx(i,a) = Hy(i,a);
        
%         if plt ==1
%             % conditional distribution lo|la:
%             figure; hold on; histogram(ATO,50); title(strcat('H_{IOI|IAI}')); prettyplot;
%             disp('A->O wait time mean:'); disp(nanmean(ATO))% mean... i wonder if this can be parameterized (it doesn't look like any exponential distribution...)
%            
%             % unconditional distribution o:
%             figure; hold on; histogram(IOI,50); title(strcat('H_{IOI}')); prettyplot;
%             disp('R->O wait time mean:'); disp(nanmean(RTO))% mean... i wonder if this can be parameterized (it doesn't look like any exponential distribution...)
%       
%         end
      %  FI.meanATO(i,a) = nanmean(ATO);
     %   FI.HOA(i,a) = calc_entropy(ATO);
      
        
        
        %% VR world | variable ratio
        t = 1;
        agent.O_VI = zeros(1,T); agent.O_VI(1) = 1;
        k = 1;
        while t<T
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
        
        % for each outcome that occured, calculate (1) the IOI and (2) #actions that occured
        % unconditional distribution
        IOI = VI.waitTimes; %= diff(VI.outTimes); (1) these are the IOIs. confirm that lo=la for la<lo 
        
        numAct = zeros(1,length(IOI));
        IAI = zeros(1,length(IOI));
        for o = 1:length(IOI) % for each IOI, calculate the <IAI>
            numAct(o) = sum(A(VI.outTimes(o):VI.outTimes(o+1)));% how many actions between outcomes 
            IAI(o) = IOI(o)/numAct(o);
        end 
        
        est_la = 1./IAI; % estimated lambda_a
        est_lo = 1./IOI; % estimated lambda_o
        if plt==1
             figure; scatter(IAI,IOI); figure;scatter(est_la,est_lo);
        end
        
        % entropy of IOI H(IOI)
        VI.HO(i,a) = calc_entropy(IOI);
        %VI.HA(i,a) = entropy(IAI);
       % VI.HOA(i,a) = calc_entropy([IAI;IOI]');
 
        
        % conditional: the entropy of the action->outcome times (should be
        % around same as unconditional)
        % this is NOT the action rate, lambda_a
        % for every action, get time from action to outcome
       % IOI_IAI = NaN(1,T); % IOI; conditional dist, inter-outcome interval
        % take only the IOIs that 
        %VI.HOA(i,a) = calc_entropy(IAI);
       
        [VI.Cyx(i,a),VI.MI(i,a),Hy(i,a),Hx(i,a),Hxy(i,a)] = JntDistContingN(IAI,IOI,bs(a),bs(a));
        VI.HOA(i,a) = Hxy(i,a)-Hx(i,a);
       
       %VI.HOA_anal(i,a) = (lambda_a.*(1-log(VI.lambda_o(i,a))))./(lambda_a+exp(-lambda_a));
        
       VI.HO_fx(i,a) = Hy(i,a);
        
       % if VI.HOA(i,a)<0 
        %    VI.HOA(i,a) = 0;
        %end
%         
%         if plt ==1
%             % avg number of actions:
%             figure; hold on; histogram(numAct,50); title(strcat('number of actions before reward')); prettyplot;
%             disp('mean num actions before reward:'); disp(mean(numAct));
%             
%             % conditional distribution O|A:
%             figure; hold on; histogram(ATO,50); title(strcat('H_{A|O} a->o wait times')); prettyplot;
%             disp('A->O wait time mean:'); disp(nanmean(ATO))% mean... i wonder if this can be parameterized (it doesn't look like any exponential distribution...)
%            
%             % unconditional distribution O:
%             figure; hold on; histogram(ATO,50); title(strcat('H_{O} rand->o wait times')); prettyplot;
%             disp('R->O wait time mean:'); disp(nanmean(RTO))% mean... i wonder if this can be parameterized (it doesn't look like any exponential distribution...)
%       
%         end
% %         
%        IAI = reshape(IAI,9,11);
%        IOI = reshape(IOI,9,11);
%        for i =1:11
%         [~,MI(i),~,~,~] = JntDistContingN(IAI(:,i),IOI(:,i),bs(a),bs(a));
%        end
       
    end
end

%look at error
%note that below the interval expectation, there is an error in calculating

VI.err = VI.HO_fx-VI.HO;
FI.err = FI.HO_fx-FI.HO;

% correct for the error
VI.HOA = VI.HOA - VI.err;
FI.HOA = FI.HOA - FI.err;
%VI.HOA(VI.HOA<0) = 0;
%FI.HOA(FI.HOA<0) = 0;
%% calculate information

FI.I = FI.HO-FI.HOA;
VI.I = VI.HO-VI.HOA;
% 
%FI.I = FI.HO_fx-FI.HOA;
%VI.I = VI.HO_fx-VI.HOA;

%FI.HO = FI.HO_fx;
%VI.HO = VI.HO_fx;

%FI.I = FI.MI;
%VI.I = VI.MI;

%% plots
figure; hold on;
subplot 221; hold on;
plot(la,VI.lambda_o,'LineWidth', 1.5);
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
ylabel('conditional entropy H_{\lambda_o|\lambda_a}')
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
ylabel('I(\lambda_o;\lambda_a)')
axis square
prettyplot

% VI schedules, A-O contingency is strong, because the faster they respond,
% the sooner on average a reinforcer is delivered.
subplot 224
C = VI.I./VI.HO;
plot(la',C','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('A-O rate contingency')
axis([0 1 0 1])
axis square
prettyplot

suptitle ('VI')

%% FI
figure; hold on;
subplot 221; hold on;
plot(la,FI.lambda_o,'LineWidth', 1.5);
% plot(a',beta','LineWidth'4, 1.5)
% text(1.5,90, 'less habit','FontSize',14)
% text(1.5,10, 'more habit','FontSize',14)
xlabel('action rate \lambda_a')
ylabel('outcome rate \lambda_o')
% ylabel('resource constraint \beta')
axis square
% axis([0 2 0 100])
legend('FI2','FI5','FI10','FI20')
legend('boxoff')
prettyplot


subplot 222; hold on;
plot(la',FI.HO','--','LineWidth', 1.5);
plot(la',FI.HOA','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('conditional entropy H_{\lambda_o|\lambda_a}')
legend('FI1','FI2','FI5','FI10','FI20','Location','southeast')
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
ylabel('I(\lambda_o;\lambda_a)')
axis square
prettyplot

% VI schedules, A-O contingency is strong, because the faster they respond,
% the sooner on average a reinforcer is delivered.
subplot 224
C = FI.I./FI.HO;
plot(la',C','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('A-O rate contingency')
axis([0 1 0 1])
axis square
prettyplot
suptitle ('FI')


 %% reward complexity stuff 
figure; plot(la,VI.I);
figure;plot(VI.I', VI.lambda_o','.','MarkerSize',30) % avg reward vs. complexity
xlabel('policy complexity I(\lambda_o,\lambda_a)'); ylabel('average reward V^\pi'); prettyplot

figure;hold on;
for i = 1:4
plot(mean(VI.I(i,:)), mean(VI.lambda_o(i,:)),'.','MarkerSize',50) % avg reward vs. complexity (avged over all action rates)
end
%% beta? (use a smaller step size)
%smoothI = movmean(VR.I,2);
%smoothLO = movmean(VR.lambda_o,2);
%for i = 1:size(VI.I,1)
%    VI.dIdA(i,:) = gradient(VI.I(i,:));
%    VI.dOdA(i,:) = gradient(VI.lambda_o(i,:)); %this is already with respect to lambda a
%end
%VI.beta = VI.dIdA.*(1./VI.dOdA);
% 
% figure; plot(la,VI.dIdA'); figure;plot(la,VI.dOdA');
% figure; plot(la,VI.beta');
% figure; bar(mean(VI.beta,2))
% 
% 
% VI.beta = VI.dIdA.*VI.dAdO;
% 
% figure; plot(la,VI.dIdA'); figure;plot(la,VI.dAdO'); 
% figure; plot(la,VI.beta'); 
end
