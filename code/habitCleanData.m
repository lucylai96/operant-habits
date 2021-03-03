function habitCleanData
% PURPOSE: clean Eric Garr's dataset to be in a format compatible for model
% fitting

% WRITTEN BY: Lucy Lai (May 2020)

%% some notes about the dataset:

% FR-20: 16 rats trained for 30 daily sessions (sessions terminated when 50 rewards were earned or 1 hour elapsed, whichever occurred VIrst). Each rat was administered a speciVIc satiety devaluation test after sessions 2, 10, 20, and 30.
% RR-20: 15 rats trained and tested in an identical manner to the FR-20 rats.

% VI-45: 8 rats trained for 20 daily sessions (sessions lasted a fixed 38 minutes). Each rat was administered a speciVIc satiety devaluation test after sessions 2, 10, and 20.
% RI-45: 39 rats. Think of these rats as belonging to 3 different cohorts. Cohort 1 (n = 8) was trained for 20 daily sessions and tested after sessions 2, 10, and 20. Cohort 2 (n = 16) was trained for 2 daily sessions and only tested after session 2. Cohort 3 (n = 15) was trained for 20 daily sessions and only tested after session 20. Sessions always lasted a VIxed 38 minutes.
% Interval schedules from Experiment 4b in paper

% The correspondence between "session" and training days is as follows:
% 1 --> Day 2
% 2 --> Day 10
% 3 --> Day 20
% 4 --> Day 30

% VI45(1).session(1).test_lever_rates; % mean lever presses per min


% to convert from mins to deciseconds:
% 5 mins * 600 decisecs per min = 300 "timesteps"
% 27361 decisecs / 600 decisecs per min ~= 45 mins = 2736 seconds

clear all
close all
addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools/bads-master')
habitColors;

examp = 0; % loading example rat or whole dataset?
run = 1;

if examp == 1 && run == 1
    example
elseif run == 1
    whole_dataset
end

end

function whole_dataset

%% unpack data
load('complete_data.mat')

% assign diff variables to diff structures
FR20 = schedule.FR;
VR20 = schedule.RR;
FI45 = schedule.FI;
VI45 = schedule.RI;
numsess = 20;
M2T = 0;
max2t = 100;

%% FR20
for r = 1:length(schedule.FR.training.session(1).lever_time_stamps)
    for i = 1:numsess % for all sessions, convert timestamps to secs
        FR20.training.session(i).lever_time_stamps{r}(FR20.training.session(i).lever_time_stamps{r}==0)=[];
        FR20.training.session(i).lever_time_stamps{r} = ceil(FR20.training.session(i).lever_time_stamps{r}./100); % centisecs --> millisecond
        FR20.training.session(i).reward_time_stamps{r}(FR20.training.session(i).reward_time_stamps{r}==0) =[];
        FR20.training.session(i).reward_time_stamps{r} = ceil(FR20.training.session(i).reward_time_stamps{r}./100);
        m2t = 1; xidx = 1;
        %length(FR20.training.session(i).lever_time_stamps{r})
        
        while m2t>max2t && min(FR20.training.session(i).lever_time_stamps{r}) >1
            xidx = find(diff(FR20.training.session(i).lever_time_stamps{r})==0);
            FR20.training.session(i).lever_time_stamps{r}(xidx) = FR20.training.session(i).lever_time_stamps{r}(xidx)-1;
            m2t = length(find(diff(FR20.training.session(i).lever_time_stamps{r})==0));      % how many more than 2 taps in a decisecond
        end
        if m2t > max2t
            M2T = M2T + 1;
        end
        
        FR20.training.session(i).lever_time_stamps{r} = unique(FR20.training.session(i).lever_time_stamps{r});
        
        
        % reorganize data into: [0 1] depending on if reward was observed
        FR20n(r).session(i).training.lever_binned = zeros(1,max(FR20.training.session(i).lever_time_stamps{r}));
        FR20n(r).session(i).training.lever_binned(FR20.training.session(i).lever_time_stamps{r}) = 1;
        FR20n(r).session(i).training.reward_binned = zeros(1,max(FR20.training.session(i).lever_time_stamps{r}));
        FR20n(r).session(i).training.reward_binned(FR20.training.session(i).reward_time_stamps{r}) = 1;
        FR20n(r).timeSteps(i) = max(FR20.training.session(i).lever_time_stamps{r});
        k = 1;outTimes = [1; FR20.training.session(i).reward_time_stamps{r}];
        for t = 1:length(outTimes)-1
            if outTimes(t+1)> length(FR20n(r).session(i).training.lever_binned)
                outTimes(t+1) = length(FR20n(r).session(i).training.lever_binned);
            end
            numAct(k) = sum(FR20n(r).session(i).training.lever_binned(outTimes(t):outTimes(t+1))); % sum up the number of actions between outTimes
            k = k+1;
        end
        FR20n(r).session(i).actions = numAct;
        
        FR.actRate(r,i) = sum(FR20n(r).session(i).training.lever_binned)./FR20n(r).timeSteps(i);
    end % numsess
end % for each rat


%% VR20
for r = 1:length(schedule.RR.training.session(1).lever_time_stamps)
    for i = 1:numsess % for all sessions, convert timestamps to secs
        
        VR20.training.session(i).lever_time_stamps{r}(VR20.training.session(i).lever_time_stamps{r}==0)=[];
        VR20.training.session(i).lever_time_stamps{r} = ceil(VR20.training.session(i).lever_time_stamps{r}./100); % centisecs --> millisecond
        VR20.training.session(i).reward_time_stamps{r}(VR20.training.session(i).reward_time_stamps{r}==0) =[];
        VR20.training.session(i).reward_time_stamps{r} = ceil(VR20.training.session(i).reward_time_stamps{r}./100);
        m2t = 1;  xidx = 1;
        %length(VR20.training.session(i).lever_time_stamps{r})
        
        while m2t>0 && min(VR20.training.session(i).lever_time_stamps{r}) >1
            xidx = find(diff(VR20.training.session(i).lever_time_stamps{r})==0);
            VR20.training.session(i).lever_time_stamps{r}(xidx) = VR20.training.session(i).lever_time_stamps{r}(xidx)-1;
            m2t = length(find(diff(VR20.training.session(i).lever_time_stamps{r})==0));      % how many more than 2 taps in a decisecond
        end
        if m2t > 0
            M2T = M2T + 1;
        end
        
        VR20.training.session(i).lever_time_stamps{r} = unique(VR20.training.session(i).lever_time_stamps{r});
        
        % reorganize data into: [0 1] depending on if reward was observed
        VR20n(r).session(i).training.lever_binned = zeros(1,max(VR20.training.session(i).lever_time_stamps{r}));
        VR20n(r).session(i).training.lever_binned(VR20.training.session(i).lever_time_stamps{r}) = 1;
        VR20n(r).session(i).training.reward_binned = zeros(1,max(VR20.training.session(i).lever_time_stamps{r}));
        VR20n(r).session(i).training.reward_binned(VR20.training.session(i).reward_time_stamps{r}) = 1;
        VR20n(r).timeSteps(i) = max(VR20.training.session(i).lever_time_stamps{r});
        
        clear numAct
        k = 1;
        outTimes = [1; VR20.training.session(i).reward_time_stamps{r}];
        for t = 1:length(outTimes)-1
            if outTimes(t+1)> length(VR20n(r).session(i).training.lever_binned)
                outTimes(t+1) = length(VR20n(r).session(i).training.lever_binned);
            end
            numAct(k) = sum(VR20n(r).session(i).training.lever_binned(outTimes(t):outTimes(t+1))); % sum up the number of actions between outTimes
            k = k+1;
        end
        VR20n(r).session(i).actions = numAct;
        VR.actRate(r,i) = sum(VR20n(r).session(i).training.lever_binned)./VR20n(r).timeSteps(i);
    end % numsess
end % for all rats


%% FI45
for r = 1:length(schedule.FI.training.session(1).lever_time_stamps)
    for i = 1:numsess % for all sessions, convert timestamps to secs
        
        FI45.training.session(i).lever_time_stamps{r}(FI45.training.session(i).lever_time_stamps{r}==0)=[];
        FI45.training.session(i).lever_time_stamps{r} = ceil(FI45.training.session(i).lever_time_stamps{r}./100); % centisecs --> millisecond
        FI45.training.session(i).reward_time_stamps{r}(FI45.training.session(i).reward_time_stamps{r}==0) =[];
        FI45.training.session(i).reward_time_stamps{r} = ceil(FI45.training.session(i).reward_time_stamps{r}./100);
        m2t = 1;  xidx = 1;
        %length(FI45.training.session(i).lever_time_stamps{r})
        
        while m2t>max2t && min(FI45.training.session(i).lever_time_stamps{r}) >1
            xidx = find(diff(FI45.training.session(i).lever_time_stamps{r})==0);
            FI45.training.session(i).lever_time_stamps{r}(xidx) = FI45.training.session(i).lever_time_stamps{r}(xidx)-1;
            m2t = length(find(diff(FI45.training.session(i).lever_time_stamps{r})==0));      % how many more than 2 taps in a decisecond
        end
        if m2t > max2t
            M2T = M2T + 1;
        end
        FI45.training.session(i).lever_time_stamps{r} = unique(FI45.training.session(i).lever_time_stamps{r});
        
        % reorganize data into: [0 1] depending on if reward was observed
        FI45n(r).session(i).training.lever_binned = zeros(1, 2290);
        FI45n(r).session(i).training.lever_binned(FI45.training.session(i).lever_time_stamps{r}) = 1;
        FI45n(r).session(i).training.reward_binned = zeros(1, 2290);
        FI45n(r).session(i).training.reward_binned(FI45.training.session(i).reward_time_stamps{r}) = 1;
        FI45n(r).timeSteps(i) = 2290;
        FI45n(r).session(i).times = [diff([1; FI45.training.session(i).reward_time_stamps{r}])];
        
        
        FI.actRate(r,i) = sum(FI45n(r).session(i).training.lever_binned)./FI45n(r).timeSteps(i);
    end % numsess
end % numrats


%% VI45
for r = 1:length(schedule.RI.training.session(1).lever_time_stamps)
    for i = 1:numsess % for all sessions, convert timestamps to secs
        VI45.training.session(i).lever_time_stamps{r}(VI45.training.session(i).lever_time_stamps{r}==0) =[];
        VI45.training.session(i).lever_time_stamps{r} = ceil(VI45.training.session(i).lever_time_stamps{r}./100); % centisecs --> decisecond
        VI45.training.session(i).reward_time_stamps{r}(VI45.training.session(i).reward_time_stamps{r}==0) =[];
        VI45.training.session(i).reward_time_stamps{r} = ceil(VI45.training.session(i).reward_time_stamps{r}./100);
        m2t = 1;  xidx = 1;
        %length(VI45.training.session(i).lever_time_stamps{r})
        
        while m2t>max2t && min(VI45.training.session(i).lever_time_stamps{r})>1
            xidx = find(diff(VI45.training.session(i).lever_time_stamps{r})==0);
            VI45.training.session(i).lever_time_stamps{r}(xidx) = VI45.training.session(i).lever_time_stamps{r}(xidx)-1;
            m2t = length(find(diff(VI45.training.session(i).lever_time_stamps{r})==0));      % how many more than 2 taps in a decisecond
        end
        if m2t > max2t
            M2T = M2T + 1;
        end
        VI45.training.session(i).lever_time_stamps{r} = unique(VI45.training.session(i).lever_time_stamps{r});
        
        % reorganize data into: [0 1] depending on if reward was observed
        VI45n(r).session(i).training.lever_binned = zeros(1, 2290);
        VI45n(r).session(i).training.lever_binned(VI45.training.session(i).lever_time_stamps{r})= 1;
        VI45n(r).session(i).training.reward_binned = zeros(1, 2290);
        VI45n(r).session(i).training.reward_binned(VI45.training.session(i).reward_time_stamps{r})= 1;
        VI45n(r).timeSteps(i) = 2290;
        VI45n(r).session(i).times = [diff([1; VI45.training.session(i).reward_time_stamps{r}])];
        
        VI.actRate(r,i) = sum(VI45n(r).session(i).training.lever_binned)./VI45n(r).timeSteps(i);
        
    end % numsess
end % numrats


for d = 1:3 % 3 deval sessions
    FR.test(:,:,d) = [FR20.test.cycle(d).lever_presses.valued FR20.test.cycle(d).lever_presses.devalued]./60;
    VR.test(:,:,d) = [VR20.test.cycle(d).lever_presses.valued VR20.test.cycle(d).lever_presses.devalued]./60;
    
    FI.test(:,:,d) = [FI45.test.cycle(d).lever_presses.valued FI45.test.cycle(d).lever_presses.devalued]./60;
    VI.test(:,:,d) = [VI45.test.cycle(d).lever_presses.valued VI45.test.cycle(d).lever_presses.devalued]./60;
    
    FR.delta(:,d) = FR.test(:,1,d) - FR.test(:,2,d);
    VR.delta(:,d) = VR.test(:,1,d) - VR.test(:,2,d);
    FI.delta(:,d) = FI.test(:,1,d) - FI.test(:,2,d);
    VI.delta(:,d) = VI.test(:,1,d) - VI.test(:,2,d);
end
FR20 = FR;
VR20 = VR;
FI45 = FI;
VI45 = VI;

save('all_data_cleaned.mat','FI45','VI45','FI45n','VI45n','FR20','VR20','FR20n','VR20n')

end
