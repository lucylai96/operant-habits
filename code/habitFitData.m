function habitFitData
% PURPOSE: Fitting to Eric Garr's data
% REQUIRE: example_rats_cleaned.mat
% WRITTEN BY: Lucy Lai (May 2020)

load('example_rats_cleaned.mat'); % load data
%% timesteps
sched.model = 2; % fit the beta
sched.R = 20; % in deciseconds (VR/FR20*10 = 200)
sched.I = 45; % in deciseconds  (VI/FI45*10 = 450)

% a loop to go through each schedule separately
type = {'FR' 'VR' 'FI' 'VI'}; % even though it's FR, there are still variable times

% a loop to go through each session separately
for typ = 4:length(type)   
    for ses = 1:3
        sched.type = type{typ};
        sched.timeSteps = timeSteps(typ,ses);
        sched.devalTime = sched.timeSteps;
        
        %% for FR only
        if sched.type == 'FR'
            input.data = FR20.session(ses).training.lever_binned;
        end
        
        %% for VR only
        if sched.type == 'VR'
            sched.actions = [VR20.session(ses).actions];
            input.data = VR20.session(ses).training.lever_binned;
        end
        
        %% for FI only
        if sched.type == 'FI'
            input.data = FI45.session(ses).training.lever_binned;
        end
        
        %% for VI only
        if sched.type == 'VI'
            sched.times = [VI45.session(ses).times];
            input.data = VI45.session(ses).training.lever_binned;
        end
        
        input.sched = sched;
        
        
        %% fitting (with BADS)
        opts = optimset('MaxFunEvals',10, 'MaxIter',10);
        for fit_i = 1 % fit every subject 5 times w/ random x0s
            
            % learning rates, action cost, beta, cmax (last 2 only relevant for model 4)
            % parameters: alpha_w = alpha_t = alpha_b, acost, beta, cmax)
           
            % x0 = [0.03+rand*0.1  0.03+rand*0.1 0.03+rand*0.1  10];
           % [params(fit_i,:),error(fit_i)] = bads(@model,x0,[0.01 0.01 0.01 1],[.99 .99 .99 100],[0.01 0.01 0.01 1],[.8 .8 .8 100],[],[],input);
            
            x0 = [0.03+rand*0.1  0.03+rand*0.1   0.03+rand*0.1 10];
            [params(fit_i,:),error(fit_i)] = bads(@model,x0,[0.01 0.01 0.01 1],[.99 .99 .99 100],[0.01 0.01 0.01 1],[.8 .8 .8 100],[],[],input);
        
        
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

sched = input.sched;

%% initialize
agent.alpha_w = params(1);
agent.alpha_t = params(2);
agent.alpha_b = params(1);  % only relevant for model 3 and 4

sched.acost = params(3);    % action cost

sched.beta = params(4);     % starting beta; high beta = low cost. beta should increase for high contingency
%sched.cmax = params(6);     % max complexity
sched.k = 1;

[O,T] =  habitSchedule(sched);
results = habitAgent(O,T, sched, agent);


% bernoulli log liklihood for each second
% sum up nLL for each decisecond bin

% Y is the action obs [1 or 0]
% n is number of obs (1)
% p(i) is probability of action (p_as) (changing from timestep to timestep)

p = results.pa;
p(p<0.01) = 0.01;
p(p>0.99) = 0.99;
Y = input.data;
n = ones(1,length(Y));
nLL = -sum(Y.*log(p) + (n-Y).*log(1-p));

% modifications: need to fix the exact reward delivery schedule and the

end
