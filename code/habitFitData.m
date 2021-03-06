function habitFitData
% PURPOSE: Fitting to Eric Garr's data, all rats (instead of just 1)
% REQUIRE: all_data_cleaned.mat
% WRITTEN BY: Lucy Lai (May 2020)
% LAST UPDATE: Lucy Lai (Mar 2021)

% Sessions:
% 1 --> Day 2
% 2 --> Day 10
% 3 --> Day 20

addpath('/params');
addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools');                  % various plot tools
addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools/bads-master');      % opt toolbox

map = habitColors;               % set color scheme
load('all_data_cleaned.mat');    % load data
load('data.mat')
% FI45n - the n indicates individual rat data
% FI45 - summary data and reformatted action rates etc.

%% inputs
% specify:
% (1) schedule
% (2) arming parameter
% (3) rat #

% loop over all schedules and all rats (fit 1 session all together)
% init 4 parameters being fit

%% instantiation
type = {'FR' 'VR' 'FI' 'VI'};

for sch = 4:length(type)
    for r = 1:length(schedule(sch).rat)
        data = schedule(sch).rat(r);
        a = []; x = []; % unroll all data
        
        for s = 1:20 % 20 sessions
            a = [a data.session(s).training.lever_binned];
            x = [x data.session(s).training.reward_binned];
        end
        
        sched.type = type{sch};
        sched.R = 20; sched.I = 45;
        
        input.sched = sched;
        input.timeSteps = cumsum(data.timeSteps);
        input.a = a;
        input.x = x;
        
        %% fitting
        opts = optimset('MaxFunEvals',50, 'MaxIter',100);
        
        % fit 1 beta for all days
        for f = 1 % fit every subject 5 times w/ random x0s
            % fit static beta
            %x0 = [rand*0.05  rand*0.05 rand*0.05 1+rand*10];
            %[params(f,:),error(f)] = bads(@habitAgentFit, x0, [0.0001 0.0001 0.001 1],[.05 .05 .5 15],[0.001 0.0001 0.001 1],[.05 .05 .3 10],[],opts,input);
            
            % changing beta
            x0 = [0.1+rand*1 rand*0.01  rand*0.01 rand*0.01 rand*0.01];
            [params(f,:),error(f)] = bads(@habitAgentFit, x0, [0.1 0.001 0.001 0.001 0.001],[10 .01 .01 .01 .01],[0.1 0.001 0.001 0.001 0.001],[10 .01 .01 .01 .01],[],opts,input);
       
        end  % number of fit iterations
        
        %   x0 = 0.0002    0.0322    0.0176    3.5013
        
        % fit different betas for each day but also yoke the learning
        % rates acros all
        %           for s = 1:20
        %             if s > 1 % only init params for first session
        %                 x0 = [0.01+rand*0.05  0.01+rand*0.05 0.03+rand*0.1 0.03+rand*0.1];
        %             end
        %             for f = 1 % fit every subject 5 times w/ random x0s
        %                 x0 = [0.01+rand*0.05  0.01+rand*0.05 0.03+rand*0.1 0.03+rand*0.1];
        %                 [params(f,:),error(f)] = bads(@model, x0, [0.0001 0.0001 0.001 0.001],[.01 .01 .5 .5],[0.0001 0.0001 0.001 0.001],[.001 .001 .3 .3],[],[],input);
        %             end  % number of fit iterations
        %         end
        save(strcat('sch',num2str(sch),'_r',num2str(r),'.mat'),'params','error'); % save by rat #
        
    end % each rat
    
end % each schedule


end % habitFitData
