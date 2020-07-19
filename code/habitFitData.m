function habitFitData
% PURPOSE: Fitting to Eric Garr's data
% REQUIRE: example_rats_cleaned.mat
% WRITTEN BY: Lucy Lai (May 2020)

% Sessions:
% 1 --> Day 2
% 2 --> Day 10
% 3 --> Day 20

habitColors % set color scheme
load('example_rats_cleaned.mat'); % load data

%% instantiation
type = {'FR' 'VR' 'FI' 'VI'}; 
sched.model = 3; % fit the beta
sched.R = 20; % in deciseconds (VR/FR20*10 = 200)
sched.I = 45; % in deciseconds  (VI/FI45*10 = 450)

for ses = 1:3
    figure(1); hold on; 
    subplot(3,1,ses); hold on;
    plot(FR20.session(ses).training.lever_binned);
    plot(VR20.session(ses).training.lever_binned+1);
    plot(FI45.session(ses).training.lever_binned+2);
    plot(VI45.session(ses).training.lever_binned+3);
    
    figure(2); hold on; 
    subplot(3,1,ses); hold on;
    plot(movmean(FR20.session(ses).training.lever_binned,100));
    plot(movmean(VR20.session(ses).training.lever_binned,100));
    plot(movmean(FI45.session(ses).training.lever_binned,100));
    plot(movmean(VI45.session(ses).training.lever_binned,100));
end
figure(1); hold on; xlabel('time(s)'); ylabel('action');subprettyplot(3,1)
figure(2); hold on; xlabel('time(s)'); ylabel('action rate (presses/sec)');subprettyplot(3,1)
legend([type],'FontSize',15); legend('boxoff')

% a loop to go through each schedule separately
% a loop to go through each session separately
for typ = 1:length(type)
    for ses = 1:3
        sched.type = type{typ};
        sched.timeSteps = timeSteps(typ,ses);
        sched.devalTime = sched.timeSteps;
        
        %% for FR only
        if sched.type == 'FR'
            input.data = [FR20.session(ses).training.lever_binned; FR20.session(ses).training.reward_binned];
            %figure;hold on; plot(input.data(1,:));plot(input.data(2,:),'LineWidth',3);
        end
        
        %% for VR only
        if sched.type == 'VR'
            sched.actions = [VR20.session(ses).actions];
            input.data = [VR20.session(ses).training.lever_binned; VR20.session(ses).training.reward_binned];
        end
        
        %% for FI only
        if sched.type == 'FI'
            input.data = [FI45.session(ses).training.lever_binned; FI45.session(ses).training.reward_binned];
        end
        
        %% for VI only
        if sched.type == 'VI'
            sched.times = [VI45.session(ses).times];
            input.data = [FI45.session(ses).training.lever_binned; FI45.session(ses).training.reward_binned];
        end
        
        input.sched = sched;
        
        
        %% fitting (with BADS)
        opts = optimset('MaxFunEvals',10, 'MaxIter',10);
        for fit_i = 1:5 % fit every subject 5 times w/ random x0s
            
            % learning rates, action cost, beta, cmax (last 2 only relevant for model 4)
            
            
            %% fitting beta
            if sched.model == 3 % [alpha_w, alpha_t, alpha_r, beta, acost]
                x0 = [0.03+rand*0.1  0.03+rand*0.1 0.03+rand*0.1  1+rand*0.1  0.03+rand*0.1];
                %x0 = [0.1 0.2 0.1 3 0.2]; % debugging
                %x0 = [0.1158   0.8807   0.0154   1.8724   0.4138];
                [params(fit_i,:),error(fit_i)] = bads(@model,x0,[0.01 0.01 0.01 0.01 0.01],[.99 .99 .99 15 0.99],[0.01 0.01 0.01 0.01 0.01],[.8 .8 .8 3 .8],[],[],input);
            end
            %% fitting the beta learning rate
            if sched.model == 4 % [alpha_w, alpha_t, alpha_r, alpha_b, acost]
                x0 = [0.03+rand*0.1  0.03+rand*0.1  0.03+rand*0.1  rand*5 0.03+rand*0.1];
                [params(fit_i,:),error(fit_i)] = bads(@model,x0,[0.01 0.01 0.01 0.01 0.01],[.99 .99 .99 5 .99],[0.01 0.01 0.01 0.01 0.01],[.8 .8 .8 5 .8],[],[],input);
            end
            
        end
        % params for FR20(1)= [0.0714    0.9898    0.9202    1.8864]
        % error for FR20 = 4582.7
        save(strcat(sched.type,'_params_s',num2str(ses),'.mat'),'params','error');
        
    end
end
end

function nLL = model(params,input)
% INPUT: params
%        data:
%
% use the data to fix:
% (1) total timesteps
% (2) the reward times / number of actions until reward
% (3) reward

%  each session separately (bc diff rats)
% make sure to record: the action probabilities for that timeStep, compare
% to whether there was
%
% parameters to be VIt:
% (1) learning rates

maxiter = 20;
sched = input.sched;

%% initialize
agent.alpha_w = params(1);
agent.alpha_t = params(2);
agent.alpha_r = params(3);  % only relevant for model 3 and 4
agent.alpha_b = 0;          % only relevant for model 3 and 4
sched.beta = params(4);     % only relevant for model 3 and 4
sched.acost = params(5);    % action cost
% sched.beta = 1;     % starting beta; high beta = low cost. beta should increase for high contingency
sched.cmax = 100;     % max complexity
sched.k = 1;

% run many times to get an avg est
for i = 1:maxiter
    results(i) = habitAgent(sched, agent, input.data(1,:));
end

pmean = mean(cat(3,results.pi_as),3);
pstd = std(cat(3,results.pi_as),[],3);
pm = pmean(:,2)';
ps = pstd(:,2)';

% bernoulli log liklihood for each second
% sum up nLL for each decisecond bin

% Y is the action obs [1 or 0]
% n is number of obs (1)
% p(i) is probability of action (p_as) (changing from timestep to timestep)

p = pm;
p(p<0.01) = 0.01;
p(p>0.99) = 0.99;
Y = input.data(1,:);

%% sanity checks
% figure; hold on;shadedErrorBar(1:sched.timeSteps,pm',ps');
% A = binornd(ones(1,sched.timeSteps),pm);  figure; hold on;subplot 311; hold on;plot(movmean(A,200));plot(movmean(Y,200)); title('actions');prettyplot; subplot 313;hold on; plot(results(1).x,'LineWidth',3);plot(input.data(2,:),'LineWidth',3); title('rewards');prettyplot;subplot 312;hold on; plot(results(1).a,'LineWidth',3);plot(input.data(1,:),'LineWidth',3); title('actions');prettyplot; % were rewards delivered at the same time?

n = ones(1,length(Y));
nLL = -sum(Y.*log(p) + (n-Y).*log(1-p));

% modifications: need to fix the exact reward delivery schedule to that of
% the thing?

end
