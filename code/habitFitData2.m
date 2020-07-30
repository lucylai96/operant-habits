function habitFitData2
% PURPOSE: Fitting to Eric Garr's data, only FI and VI
% REQUIRE: example_rats_cleaned.mat
% WRITTEN BY: Lucy Lai (May 2020)

% Sessions:
% 1 --> Day 2
% 2 --> Day 10
% 3 --> Day 20

addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools');                  % various plot tools
addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools/bads-master');      % opt toolbox

habitColors % set color scheme
load('all_data_cleaned.mat'); % load data
%load('all_data.mat'); % load data
%% instantiation
type = {'FI' 'VI'};
sched.model = 4; % fit the model with fixed beta (assuming learning has converged in prev sessions)s
sched.R = 20;
sched.I = 45;

% group all sessions into one giant bin
r = 1; % take avg over all rats
for r = 1:8
    for ses = 1:20 % loop through each session separately
    FI45.lev(r,:,ses) = FI45n(r).session(ses).training.lever_binned;
    VI45.lev(r,:,ses) = VI45n(r).session(ses).training.lever_binned;
    FI45.rew(r,:,ses) = FI45n(r).session(ses).training.reward_binned;
    VI45.rew(r,:,ses) = VI45n(r).session(ses).training.reward_binned;
    end
end


for typ = 1:length(type) % loop through each schedule separately
    for r = 1:8
        sched.type = type{typ};
        sched.timeSteps = 2290; % hardcoded for now
        sched.devalTime = sched.timeSteps;
        
        %         %% for FR only
        %         if sched.type == 'FR'
        %             input.data = [FR20.session(ses).training.lever_binned; FR20.session(ses).training.reward_binned];
        %         end
        %
        %         %% for VR only
        %         if sched.type == 'VR'
        %             sched.actions = [VR20.session(ses).actions];
        %             input.data = [VR20.session(ses).training.lever_binned; VR20.session(ses).training.reward_binned];
        %         end
        
        %% for FI only
        if sched.type == 'FI'
            input.data = [FI45.lev(r,:,:); FI45.rew(r,:,:)]; % 2 x timesteps x sessions
        end
        
        %% for VI only
        if sched.type == 'VI'
            sched.times = [VI45n(r).times]';
            input.data = [VI45.lev(r,:,:); VI45.rew(r,:,:)];
        end
        
        input.sched = sched;
        
        %% fitting (with BADS)
        opts = optimset('MaxFunEvals',10, 'MaxIter',10);
        
        for fit_i = 1:5 % fit every subject 5 times w/ random x0s
            
            %% fitting fixed beta
            if sched.model == 3 % params = [alpha_w, alpha_t, alpha_r, beta, acost]
                x0 = [0.03+rand*0.1  0.03+rand*0.1 0.03+rand*0.1  1+rand*0.1  0.03+rand*0.1];
                % x0 = [0.1 0.2 0.1 3 0.2]; % debugging
                % x0 = [0.1158   0.8807   0.0154   1.8724   0.4138];
                [params(fit_i,:),error(fit_i)] = bads(@model,x0,[0.01 0.01 0.01 0.01 0.01],[.99 .99 .99 20 0.99],[0.01 0.01 0.01 0.01 0.01],[.8 .8 .8 10 .8],[],[],input);
            end
            
            %% fitting the beta learning rate
            if sched.model == 4 % params = [alpha_w, alpha_t, alpha_r, alpha_b, acost]
                x0 = [0.03+rand*0.1  0.03+rand*0.1  0.03+rand*0.1  0.03+rand*0.1 0.03+rand*0.1];
                x0 = [0.0407    0.0314    0.2000    0.0241    0.0100]
                %x0 = [0.0519    0.0330    0.0578    0.0175    0.0100]
                [params(fit_i,:),error(fit_i)] = bads(@model,x0,[0.01 0.01 0.01 0.01 0.01],[.2 .2 .2 .99 .99],[0.01 0.01 0.01 0.01 0.01],[.2 .2 .2 .2 .2],[],[],input);
            end
            
        end
        
        save(strcat(sched.type,'_params_s',num2str(r),'.mat'),'params','error');
        
end
end

end

function nLL = model(params,input)
% INPUT: params - model parameters
%        input - data
%
% use the data to fix:
% (1) total timesteps
% (2) the reward times / number of actions until reward
% (3) the lever-press and reward times
% each session separately (bc diff rats)

sched = input.sched;

%% initialize
agent.alpha_w = params(1);
agent.alpha_t = params(2);
agent.alpha_r = params(3);  % only relevant for model 3 and 4
if sched.model == 4
    agent.alpha_b = params(4);          % only relevant for model 4
    sched.beta = 1;     % only relevant for model 3 and 4
else
    agent.alpha_b = 0;          % only relevant for model 4
    sched.beta = params(4);     % only relevant for model 3 and 4
end
agent.acost = params(5);    % action cost
% sched.beta = 1;     % starting beta; high beta = low cost. beta should increase for high contingency
sched.cmax = 100;     % max complexity
sched.k = 1;
sched.sessnum = 20;
sched.deval = 1;
sched.devalsess = [2 10 20];
results = habitAgent(sched, agent, input.data);

%pmean = mean(cat(3,results.pi_as),3);
%pstd = std(cat(3,results.pi_as),[],3);
%pm = pmean(:,2)'; % mean of action probabilities across maxiter simulations
%ps = pstd(:,2)'; % std of multiple simulations

% bernoulli log liklihood for each second
% sum up nLL for each decisecond bin

% Y is the action obs [1 or 0]
% n is number of obs (1)
% p(i) is probability of action (p_as) (changing from timestep to timestep)

for s = 1:sched.sessnum
    if ismember(s, sched.devalsess)
        p(s,:) = results(s).pi_as(1:end-300,2)';
    else
        p(s,:) = results(s).pi_as(:,2)';
    end
end

p(p<0.01) = 0.01;
p(p>0.99) = 0.99;
Y = squeeze(input.data(1,:,:))'; % action sequence (binned in seconds)
n = ones(size(Y));
nLL = -sum(sum(Y.*log(p) + (n-Y).*log(1-p)));

%% sanity checks
% figure; hold on;plot(1:sched.timeSteps,p'); x = ones(1,sched.timeSteps); A = binornd(repmat(x,[size(p,1) 1]),p); figure; hold on;subplot 311; hold on;shadedErrorBar(1:sched.timeSteps,mean(movmean(A',200)'),std(movmean(A',200)'));plot(movmean(Y,200));plot(movmean(Y,200)); title('actions');prettyplot; subplot 313;hold on; plot(results(1).x,'LineWidth',3);plot(input.data(2,:),'LineWidth',3); title('rewards');prettyplot;subplot 312;hold on; plot(results(1).a,'LineWidth',3);plot(input.data(1,:),'LineWidth',3); title('actions');prettyplot; % were rewards delivered at the same time?

%figure;subplot 211;plot(movmean(A',500));subplot 212;plot(movmean(Y',500));
% figure;hold on;plot(mean(A,2),'r'); plot(mean(Y,2),'k')
end
