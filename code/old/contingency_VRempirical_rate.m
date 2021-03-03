% function describing behavior of an agent -
% agent samples the

function contingency_VRempirical_rate
clear all;
addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools/');
map = brewermap(9,'Blues');
set(0, 'DefaultAxesColorOrder', map([3,5,7,9],:)) % first three rows

%% new agent
T = 7200; % timesteps

ratio = [2 5 10 20]; % rewarded every R actions
%ratio = [10]; % rewarded every R actions

la = linspace(0.01,1,20); % action rate (presses per second)
%la=1;
plt = 0;
bs = round(sort(linspace(5,50,20),'descend'));
bs = 20*ones(1,length(la));%round(sort(linspace(5,50,20),'descend'));

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
        
        % for each outcome that occured, calculate (1) the IOI and (2) #actions that occured
        % unconditional distribution
        IOI = FR.waitTimes; %= diff(FR.outTimes); (1) these are the IOIs. confirm that lo=la for la<lo
        
        numAct = zeros(1,length(IOI));
        IAI = zeros(1,length(IOI));
        for o = 1:length(IOI) % for each IOI, calculate the <IAI>
            numAct(o) = sum(A(FR.outTimes(o):FR.outTimes(o+1)));% how many actions between outcomes
            IAI(o) = IOI(o)/numAct(o);
        end
        
        est_la = 1./IAI; % estimated lambda_a
        est_lo = 1./IOI; % estimated lambda_o
        if plt==1
            figure; scatter(IOI,IAI); scatter(est_la,est_lo);
        end
        
        % entropy of IOI H(IOI)
        FR.HO(r,a) = calc_entropy(IOI);
        % since theres no entropy around the outcome times given the action
        % times, on FR schedule its actually smaller overall entropy
        
        % FR.HO(r,a) = 1-log(FR.lambda_o(r,a));
        
        % conditional: the entropy of the action->outcome times (should be
        % around same as unconditional)
        % this is NOT the action rate, lambda_a
        % for every action, get time from action to outcome
        % IOI_IAI = NaN(1,T); % IOI; conditional dist, inter-outcome interval
        % take only the IOIs that
        
        [FR.Cyx(r,a),FR.MI(r,a),Hy(r,a),Hx(r,a),Hxy(r,a)] = JntDistContingN(IAI,IOI,bs(a),bs(a));
        FR.HOA(r,a) = Hxy(r,a)-Hx(r,a);
        
        FR.HO_fx(r,a) = Hy(r,a);
        
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
        
        % for each outcome that occured, calculate (1) the IOI and (2) #actions that occured
        % unconditional distribution
        IOI = VR.waitTimes; %= diff(VR.outTimes); (1) these are the IOIs. confirm that lo=la for la<lo
        
        numAct = zeros(1,length(IOI));
        IAI = zeros(1,length(IOI));
        for o = 1:length(IOI) % for each IOI, calculate the <IAI>
            numAct(o) = sum(A(VR.outTimes(o):VR.outTimes(o+1)));% how many actions between outcomes
            IAI(o) = IOI(o)/numAct(o);
        end
        
        est_la = 1./IAI; % estimated lambda_a
        est_lo = 1./IOI; % estimated lambda_o
        if plt==1
            figure; scatter(IAI,IOI); figure;scatter(est_la,est_lo);
        end
        
        % entropy of IOI H(IOI)
        VR.HO(r,a) = calc_entropy(IOI);
        
        
        % conditional: the entropy of the action->outcome times (should be
        % around same as unconditional)
        % this is NOT the action rate, lambda_a
        % for every action, get time from action to outcome
        % IOI_IAI = NaN(1,T); % IOI; conditional dist, inter-outcome interval
        % take only the IOIs that
        %VR.HOA(i,a) = calc_entropy(IAI);
        
        [VR.Cyx(r,a),VR.MI(r,a),Hy(r,a),Hx(r,a),Hxy(r,a)] = JntDistContingN(IAI,IOI,bs(a),bs(a));
        VR.HOA(r,a) = Hxy(r,a)-Hx(r,a);
        %VR.HOA_anal(r,a) = (lambda_a.*(1-log(VR.lambda_o(r,a))))./(lambda_a+exp(-lambda_a));
        
        VR.HO_fx(r,a) = Hy(r,a);
    end
end
%% trying some stuff out
% moving mean of estimated outcome and action rates
figure;subplot 311;plot(movmean(IAI,20),'k');subplot 312;plot(movmean(IOI,20),'r')

% do
dO = gradient(movmean(IOI,100));
dA = gradient(movmean(IAI,100));
figure;subplot 211;plot(movmean(IOI,20),'k');subplot 212;plot(dO,'k')
%subplot 313;VR.dAdO = gradient(movmean(IAI,20))./gradient(movmean(IOI,20));plot(VR.dAdO,'b')
%% diagnostics

% true unconditional entropy
trueHO = 1-log(la./ratio');
figure; plot(VR.HO,trueHO,'ko'); axis equal;xlabel('empirically estimated H_{IRI}');ylabel('H_{IRI} = 1-log(\lambda_a / R)');dline;prettyplot;

VR.err = VR.HO_fx-trueHO;
FR.err = FR.HO_fx-FR.HO;

% gallistel est H_O
figure;subplot 221;plot(VR.HO_fx'); title('H_O')
subplot 222;plot(VR.HOA'); title('H_{O|A}')
subplot 223;plot(VR.MI'); title('MI')
subplot 224;plot(trueHO'-VR.HO_fx'); title('Error')
suptitle('Gallistel')

% our true analytical H_O
figure;plot(trueHO')

% correct for the error
VR.HOA2 = VR.HOA - VR.err;
FR.HOA2 = FR.HOA - FR.err;
%FR.HOA2 = FR.HOA2-min(FR.HOA2(:));

FR.I = FR.HO-FR.HOA2;
VR.I = trueHO-VR.HOA2;
% corrected gallistel
figure;subplot 221;plot(VR.HO_fx'-VR.err'); title('H_O')
subplot 222;plot(VR.HOA2'); title('H_{O|A}')
subplot 223;plot(VR.I'); title('MI')
suptitle('Gallistel')

%%


%% calculate information

%FR.I = FR.HO-FR.HOA;
%VR.I = VR.HO-VR.HOA;
%
% FR.I = FR.HO_fx-FR.HOA;
% VR.I = VR.HO_fx-VR.HOA;

%% plots
figure; hold on;
subplot 221; hold on;
plot(la,VR.lambda_o,'LineWidth', 1.5);
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
plot(la',VR.HOA2','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('conditional entropy H_{\lambda_o|\lambda_a}')
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
ylabel('I(\lambda_o;\lambda_a)')
axis square
prettyplot

% VR schedules, A-O contingency is strong, because the faster they respond,
% the sooner on average a reinforcer is delivered.
subplot 224
C = VR.I./trueHO;
plot(la',C','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('A-O rate contingency')
axis([0 1 0 1])
axis square
prettyplot

suptitle('VR')
%% FR
figure; hold on;
subplot 221; hold on;
plot(la,FR.lambda_o,'LineWidth', 1.5);
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
ylabel('conditional entropy H_{\lambda_o|\lambda_a}')
legend('FR1','FR2','FR5','FR10','FR20','Location','southeast')
legend('boxoff')
axis([0 1 0 5])
axis square
prettyplot

% mutual information between each action and outcome grows as a function of
% the agent's action rate. this is because doing more actions per unit time
% gives you more information about when outcome will appear based on ratio
% parameter
subplot 223
plot(la',FR.MI','LineWidth', 1.5)
axis([0 1 0 4])
xlabel('action rate \lambda_a')
ylabel('I(\lambda_o;\lambda_a)')
axis square
prettyplot

% VR schedules, A-O contingency is strong, because the faster they respond,
% the sooner on average a reinforcer is delivered.
subplot 224
C = FR.I./FR.HO;
plot(la',C','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('A-O rate contingency')
axis([0 1 0 1])
axis square
prettyplot

suptitle('FR')

%% reward complexity stuff 
figure; plot(la,VR.I);
figure;plot(VR.I', VR.lambda_o','.','MarkerSize',30) % avg reward vs. complexity
xlabel('policy complexity I(\lambda_o,\lambda_a)'); ylabel('average reward V^\pi'); prettyplot
figure;plot(mean(VR.I,2)', mean(VR.lambda_o,2)','k.','MarkerSize',30) % avg reward vs. complexity (avged over all action rates)

%maximum reward for smallest policy
figure;hold on;
for i = 1:4
plot(VR.I(i,end), VR.lambda_o(i,end),'.','MarkerSize',50) % avg reward vs. complexity (avged over all action rates)
end
% 
%% beta? (use a smaller step size)
%smoothI = movmean(VR.I,2);
%smoothLO = movmean(VR.lambda_o,2);
for i = 1:size(VR.I,1)
    VR.dIdA(i,:) = gradient(VR.I(i,:));
    VR.dOdA(i,:) = gradient(VR.lambda_o(i,:)); %this is already with respect to lambda a
end
VR.beta = VR.dIdA.*(1./VR.dOdA);

figure; plot(la,VR.dIdA'); figure;plot(la,VR.dOdA');
figure; plot(la,VR.beta');
figure; bar(mean(VR.beta,2))

end