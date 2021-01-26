function habitFitData2
% PURPOSE: Fitting to Eric Garr's data, all rats (instead of just 1)
% REQUIRE: all_data_cleaned.mat
% WRITTEN BY: Lucy Lai (May 2020)

% Sessions:
% 1 --> Day 2
% 2 --> Day 10
% 3 --> Day 20

addpath('/params');
addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools');                  % various plot tools
addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools/bads-master');      % opt toolbox

habitColors % set color scheme
load('all_data_cleaned.mat'); % load data

% FI45n - the n indicates individual rat data
% FI45 - summary data and reformatted action rates etc.

%% instantiation
type = {'FR' 'VR' 'FI' 'VI'};
sched.model = 4; 
sched.R = 20;
sched.I = 45;

for r = 1:8
    for typ = 1:length(type) % loop through each schedule separately
        
        sched.type = type{typ};
        if sched.type == 'FR'
            data = FR20n(r);
        elseif sched.type == 'VR'
            data = VR20n(r);
        elseif sched.type == 'FI'
            data = FI45n(r);
        elseif sched.type == 'VI'
            data = VI45n(r);
        end
        input.data = data;
        input.sched = sched;
        
        %% fitting (with BADS)
        opts = optimset('MaxFunEvals',10, 'MaxIter',10);
        
        for fit_i = 1 % fit every subject 5 times w/ random x0s
            
            %% no beta
            if sched.model == 2 % params = [alpha_w, alpha_t, acost]
                x0 = [0.001+rand*0.05  0.001+rand*0.05   0.03+rand*0.1];
                x0 =[0.0238    0.0034    0.0595];
                [params(fit_i,:),error(fit_i)] = bads(@model,x0,[0.001 0.001 0.01],[.3 .3 .5],[0.001 0.001 0.01],[.3 .3 .3],[],[],input);
            end
            %% fitting fixed beta
            if sched.model == 3 % params = [alpha_w, alpha_t, beta, acost]
                x0 = [0.001+rand*0.05  0.001+rand*0.05  1+rand*9  0.03+rand*0.1];
                [params(fit_i,:),error(fit_i)] = bads(@model,x0,[0.001 0.001 0.01 0.01],[.3 .3 10 .5],[0.001 0.001 0.05 0.01],[.3 .3 2 .3],[],[],input);
            end
            
            %% fitting the beta learning rate
            if sched.model == 4 % params = [alpha_w, alpha_t, acost, cmax] we fix the avg reward and avg expected policy cost term
                %x0 = [0.0001+rand*0.005  0.0001+rand*0.005 0.03+rand*0.1 0.03+rand*0.1];
                %[params(fit_i,:),error(fit_i)] = bads(@model, x0, [0.0001 0.0001 0.001 0.001],[.01 .01 .5 .5],[0.0001 0.0001 0.001 0.001],[.001 .001 .3 .3],[],[],input);
                %[params(fit_i,:,typ),error(fit_i,typ)] = bads(@model,x0,[0.001 0.001 0.001 0.01],[.1 .1 .1 .5],[0.001 0.001 0.001 0.01],[.1 .1 .1 .1],[],[],input);
                %[params(fit_i,:),error(fit_i)] = bads(@model,x0,[0.01 0.01 0.01 0.01 0.01],[.2 .2 .2 .99 .99],[0.01 0.01 0.01 0.01 0.01],[.2 .2 .2 .2 .2],[],[],input);
           
                % params = [alpha_w, alpha_t, acost, alpha_b] we fix the avg reward and avg expected policy cost term
                x0 = [0.0001+rand*0.005  0.0001+rand*0.005 0.03+rand*0.1 0.03+rand*0.1];
                [params(fit_i,:),error(fit_i)] = bads(@model, x0, [0.0001 0.0001 0.001 0.001],[.01 .01 .5 .5],[0.0001 0.0001 0.001 0.001],[.001 .001 .3 .3],[],[],input);
                %[params(fit_i,:,typ),error(fit_i,typ)] = bads(@model,x0,[0.001 0.001 0.001 0.01],[.1 .1 .1 .5],[0.001 0.001 0.001 0.01],[.1 .1 .1 .1],[],[],input);
                %[params(fit_i,:),error(fit_i)] = bads(@model,x0,[0.01 0.01 0.01 0.01 0.01],[.2 .2 .2 .99 .99],[0.01 0.01 0.01 0.01 0.01],[.2 .2 .2 .2 .2],[],[],input);
            
            
            end
        end   % number of fit iterations
        save(strcat(sched.type,'_params_s',num2str(r),'_m',num2str(sched.model),'.mat'),'params','error'); % save by rat #
    end % type
    
end % num rats

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
% set learning rates
agent.alpha_w = params(1);
agent.alpha_t = params(2);
agent.acost = params(3);           % action cost
%agent.cmax = params(4);            % max complexity
agent.cmax = 1;

agent.alpha_pi = 0.01;
agent.alpha_b = params(4);     
agent.alpha_r = 0.001;
agent.alpha_e = 0.001;


if sched.model == 4
    sched.beta = 0.1; % starting value of beta
    agent.acost = params(3);           % action cost
    agent.cmax = params(4);            % max complexity
elseif sched.model == 3
    sched.beta = params(3);     % only relevant for model 3 and 4
    agent.acost = params(4);    % action cost
elseif sched.model == 2
    sched.beta = 1;     % only relevant for model 3 and 4
    agent.acost = params(3);    % action cost
end

sched.k = 1;
sched.sessnum = 20;
sched.deval = 0;
sched.devalsess = [2 10 20];
sched.devalWin = 0;         % satiation window
sched.testWin = 0;
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

A = [];
YY = [];
for s = 1:sched.sessnum
    if ismember(s, sched.devalsess)
        p = results(s).pi_as(1:input.data.timeSteps(s),2)';
        p(p<0.01) = 0.01;
        p(p>0.99) = 0.99;
        Y = input.data.session(s).training.lever_binned;       % action sequence (binned in seconds)
        n = ones(size(Y));
        nLL(s) = sum(Y.*log(p) + (n-Y).*log(1-p));
        
    else
        p = results(s).pi_as(:,2)';
        p(p<0.01) = 0.01;
        p(p>0.99) = 0.99;
        Y = input.data.session(s).training.lever_binned;       % action sequence (binned in seconds)
        n = ones(size(Y));
        nLL(s) = sum(Y.*log(p) + (n-Y).*log(1-p));
    end
    x = ones(1,input.data.timeSteps(s));
    A = [A binornd(repmat(x,[size(p,1) 1]),p)];
    YY = [YY input.data.session(s).training.lever_binned];
    if 0
        figure; hold on;subplot 411; hold on;plot(movmean(A',200),'r');plot(movmean(YY,200),'k'); ylabel('p(a=press)'); legend('simulation','data');
        subplot 412;hold on; plot(movmean([results(:).a]-1,200),'r');plot(movmean(YY,200),'k'); ylabel('actual action');% were rewards delivered at the same time?
        subplot 413; hold on; plot([results.ecost],'b');ylabel('policy cost'); 
        subplot 414; plot([results.beta]);ylabel('beta'); subprettyplot(4,1)
    end
end
% pa always the same, its pi_as that changes
%figure; hold on;plot(results(s).cost);
nLL = -sum(nLL);

end
