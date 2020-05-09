function contingency_agent2
% this function is adaptive/learning agent instead of

%% todo
% learning steps:
% (1) agent samples an action from it's policy (which is a rate parameter)
% (2) does that action, collects reward

%beta = 1; % do we just sample a beta?
%pi = exp(beta*Q + logPa); % ask sam why he had a sticky parameter and what is logsumexp

%data.action(t) = fastrandsample(pi(t,:));


%% instantiate value parameters
Q = zeros(1,10); % outcome rates and action rates

%% VI schedule
% first, compute a vector of required outcome times (not contingent on actions)
int = 15; % avg required interval (VI)
[VI.I, VI.intTimes, VI.waitTimes] = contingency_generateVI(int);

figure(1); subplot 121;
histogram(VI.waitTimes,30)
title('interval wait times (VI)') % mean(VI.waitTimes)
prettyplot
% still need to calculate when rewards are actually observed by the
% agent (this is within the loop)

%% VR schedule
ratio = 15; % avg required number of presses before reward
[VR.numActions] = contingency_generateVR(ratio);

figure(1); subplot 122;
histogram(VR.numActions,30);
title('num actions required (VR)')
prettyplot
equalabscissa(1,2)

%% initialize stuff
beta = linspace(0.1,15,50); % various Beta values
alphaw = 0.1;
alphaQ = 0.1;
alphaO = 0.1;
c = 0.1;
% descretize action and outcome rates
lambda_a = 0:0.1:1;
lambda_o = 0:0.05:1;
Q = zeros(length(lambda_o), length(lambda_a));
w = zeros(1,length(lambda_o)); % value weights
R = 0;

T = 7200; % total timesteps in seconds (2 hours)
int_ratio = 10;

% Gaussian rbf
sigma = 2;


% generate schedules
[VI.I, VI.intTimes, VI.waitTimes] = contingency_generateVI(int_ratio);
[VR.numActions] = contingency_generateVR(int_ratio);


Tseed = 200; %how many seconds of experience to seed agent with

for b = 1:length(beta)
    t = 1;
    VI_k = 1; VR_k = 1; % keep track of the actions/time interval requirement at any given time
    agent.O_VR = zeros(1,T); agent.O_VR(1) = 1; % beginning reference for reward
    agent.O_VI = zeros(1,T); agent.O_VI(1) = 1; % beginning reference for reward
    
    Po = zeros(1,length(lambda_o)); %marginal probability of observing some outcome, will need to learn
    P = zeros(1,length(lambda_a));
    Pa = zeros(1,length(lambda_a));
    
    % seed agent with some experience
    randrate = randsample(lambda_a,1)
    A = binornd(1,randrate*ones(1,Tseed)); % generate vector of actions
    aWaitTimes = diff(find(A==1)); % agent's action wait times
   
    while t<Tseed
        [agent.O_VR, VR_k] = contingency_VRworld(A, agent.O_VR, t, VR_k, VR.numActions);
        [agent.O_VI, VI_k] = contingency_VIworld(A, agent.O_VI, t, VI_k, VI.waitTimes);
        t = t+1;
    end
    agent.O_VR(1:Tseed);
    agent.O_VI(1:Tseed);
    
    e = sum(agent.O_VR(1:Tseed))-1; % number of inter-reward intervals
    %e = sum(agent.O_VI(1:Tseed));
    
    % initial estimate of rates
    % find estimated lambda_os: for each 1, calculate how many indices back the last 1
    % was. then lambda_o = 1/T
    VR.oTimes = find(agent.O_VR(1:Tseed)==1);
    VR.outRate = 1./diff(find(agent.O_VR(1:Tseed)==1));
    %VI.outRate = 1./diff(find(agent.O_VI(1:Tseed)==1));
   
    
    % put it into the RBF to calculate the closest descritized lambda_o
    % tally the number of times count vector happened in vector po
    sig = .5; % peakiness of gaussian
    for i = 1:e
        rbf_o = exp(-sig*(VR.outRate(i)-lambda_o').^2);
        
        numAct = sum(A(VR.oTimes(e):VR.oTimes(e+1)));%number of actions that happened in that time period
        aRate = 1/numAct;
        rbf_a = exp(-sig*(aRate-lambda_a').^2);
        find(rbf_o==max(rbf_o));
        find(rbf_a==max(rbf_a));
        Q(find(rbf_o==max(rbf_o)),find(rbf_a==max(rbf_a))) = sum(w'.*rbf_o);
        
        Po(find(rbf_o==max(rbf_o))) = Po(find(rbf_o==max(rbf_o))) + 1; % count visitations
        
        R=1;
        w = w + alphaw.*rbf_o';
        Q(find(rbf_o==max(rbf_o)),find(rbf_a==max(rbf_a))) = Q(find(rbf_o==max(rbf_o)),find(rbf_a==max(rbf_a)))+alphaQ*(R-Q(find(rbf_o==max(rbf_o)),find(rbf_a==max(rbf_a))) );
       % figure(100);imagesc(w);pause(0.1)
        
      %  choose new action
        Pa(find(rbf_a==max(rbf_a))) = Pa(find(rbf_a==max(rbf_a))) + 1; % count visitations
        
        logPa = log(Pa./sum(Pa)+0.01); % moving avg of action rates
        %logPa = sum(Po'.*P); % sum over outcome rates
        P = exp(beta(b)*Q + logPa);
        %logPa = sum(Po.*P);
        
        P = P./sum(P(:)); % normalize
        rate = fastrandsample(P(:)');
        r = floor(rate/size(P,2));
        c = mod(rate,size(P,2));
        if c ==0
            c = 11;
        end
        %rate = fastrandsample(P(find(rbf_o==max(rbf_o)),:))
                
        % select actions
        agent.action(i) = lambda_a(c);
        A = binornd(1,agent.action(i)*ones(1,Tseed)); % generate vector of actions
        aWaitTimes = diff(find(A==1)); % agent's action wait times
   
        
        %fastrandsample(P(find(rbf_o==max(rbf_o)),:))
    end
    % compute new policy 
    
    while t<T
        
        
        % action selection
        logPa = sum(Po.*P);
        P = exp(beta(b)*Q + logPa);
        
        P = P./sum(P(:)); % normalize
        
         % instantiate variables
        A = binornd(1,p*ones(1,T)); % generate vector of actions
        aWaitTimes = diff(find(A==1)); % agent's action wait times
        
        for n = 1:N
            data(s).action(n) = fastrandsample(P(n,:));
        end
        
        [agent.O_VR, VR_k] = contingency_VRworld(A, agent.O_VR, t, VR_k, VR.numActions);
        [agent.O_VI, VI_k] = contingency_VIworld(A, agent.O_VI, t, VI_k, VI.waitTimes);
        
        %% update parameters
       % p = lambda_a(a); % agent's probability of action at any given time (assuming poisson responding)
        
     
        
        %figure(2); hold on; subplot(2,5,a); histogram(aWaitTimes,30); title(strcat('agent wait times, \lambda_a=',num2str(lambda_a(a)))); prettyplot;
        
        
        
        %VR.Q(lambda_a) = alpha * (agent.O_VR(t) - Q(o,a));
        %VI.Q(lambda_a) = alpha * (agent.O_VI(t) - Q(o,a));
        
        t = t+1;
    end
    
    %VR.outTimes = find(agent.O_VR==1);
    %VR.waitTimes = diff(VR.outTimes);
    
    VR.lambda_o(i,a) = sum(agent.O_VR==1)/T; % outcome rate
    VI.lambda_o(i,a) = sum(agent.O_VI==1)/T;
    
    
    
    %histogram of outcomes
    
    VR.oWaitTimes = diff(find(agent.O_VR==1));
    %figure;histogram(VR.oWaitTimes)
    VI.oWaitTimes = diff(find(agent.O_VI==1));
    %  hold on;histogram(VI.oWaitTimes)
    VI.HOA(i,a) = calc_entropy(aWaitTimes);
    VI.HO(i,a) = calc_entropy(VI.oWaitTimes);
    
    VR.HOA(i,a) = calc_entropy(aWaitTimes);
    VR.HO(i,a) = calc_entropy(VR.oWaitTimes);
    
    % disp(strcat('Is mean IPI wait time less than interval/ratio?:',num2str(meanATime(a)< int_ratio(i))))
    %disp(strcat('Mean inter-press interval:',num2str(meanATime(a))))
    % disp(strcat('Mean inter-reward interval:',num2str(int_ratio(i))))
    VI.contingency(i,a) = meanATime(a)< int_ratio(i);
    %mean(VR.oWaitTimes)
    %mean(VI.oWaitTimes)
    
    
end
% lambda_o is simply the outcomes per second calculated from data and DOES
% NOT DESCRIBE the parametr of a poisson-process. the distribution does have a mean of
% 1/lambda_o though but unconditional distribution H(O) has to be
% empirically measured ALWAYS (in VR and VI)

% calculate entropy
%VR.HOA = -lambda_a;
%VR.HO = -VR.lambda_o;
VR.I_OA = VR.HO - VR.HOA;
figure; hold on; subplot 121;plot(lambda_a,VR.I_OA,'o-');
xlabel('\lambda_a'); ylabel('contingency I(o;a)'); prettyplot
legend('ratio=5','ratio=10','ratio=15','ratio=25')


VI.I_OA = VI.HO - VI.HOA;
hold on; subplot 122; plot(lambda_a,VI.I_OA,'o-'); prettyplot;
legend('5s','10s','15s','25s')
equalabscissa(1,2)
set(0, 'DefaultLineLineWidth', 1.5);


figure; hold on; subplot 121;plot(1./lambda_a,VR.I_OA,'o-');
xlabel('mean inter-press interval (IPI)'); ylabel('contingency I(o;a)'); prettyplot
legend('ratio=1','ratio=5','ratio=10','ratio=15','ratio=25')
hold on; subplot 122; plot(1./lambda_a,VI.I_OA,'o-'); prettyplot;
legend('5s','10s','15s','25s')
equalabscissa(1,2)
set(0, 'DefaultLineLineWidth', 1.5);

%figure;line([VR.outTimes; VR.outTimes],[zeros(1,length(VR.outTimes)); ones(1,length(VR.outTimes))],'Color',[0 0 0])
%axis([-100 7300 -1 2])
figure; hold on;

subplot 121; hold on; plot(lambda_a, VR.lambda_o); %plot(lambda_a, lambda_a./int_ratio','r')
xlabel('action rate \lambda_a'); ylabel('outcome rate \lambda_o');
title('Variable Ratio (VR)')
legend('ratio=5, p=0.2','ratio=10, p=0.1','ratio=15, p=0.06','ratio=20, p=0.05')
prettyplot;
%legend('simulated','theoretical (equation)')


subplot 122; hold on; plot(lambda_a, VI.lambda_o);  %plot(lambda_a, lambda_a./(int.*lambda_a + 0.8),'r');
title('Variable Interval (VI)')
legend('5s','10s','15s','25s')
prettyplot
equalabscissa(1,2)

%cumulative presses over time
figure;
plot(cumsum(A))
xlabel('time (seconds)')
ylabel('cumulative presses')
prettyplot
end