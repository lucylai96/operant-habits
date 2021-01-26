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
load('complete_data (2).mat')

% assign diff variables to diff structures
FR20 = schedule.FR;
VR20 = schedule.RR;
FI45 = schedule.FI;
VI45 = schedule.RI;
numsess = 20;
numrats = 8;
M2T = 0;
max2t = 100;
for r = 1:numrats
    for i = 1:numsess % for all sessions, convert timestamps to secs
        FR20.training.session(i).lever_time_stamps{r}(FR20.training.session(i).lever_time_stamps{r}==0)=[];
        FR20.training.session(i).lever_time_stamps{r} = ceil(FR20.training.session(i).lever_time_stamps{r}./100); % centisecs --> millisecond
        FR20.training.session(i).reward_time_stamps{r}(FR20.training.session(i).reward_time_stamps{r}==0) =[];
        FR20.training.session(i).reward_time_stamps{r} = ceil(FR20.training.session(i).reward_time_stamps{r}./100);
        m2t = 1;xidx = 1;
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
        
    end
end
M2T
%% reorganize data into: [0 1] depending on if reward was observed
for r = 1:numrats
    for i = 1:numsess % for all sessions, convert timestamps to secs
        
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
        
        
        FI45n(r).session(i).training.lever_binned = zeros(1, 2290);
        FI45n(r).session(i).training.lever_binned(FI45.training.session(i).lever_time_stamps{r}) = 1;
        FI45n(r).session(i).training.reward_binned = zeros(1, 2290);
        FI45n(r).session(i).training.reward_binned(FI45.training.session(i).reward_time_stamps{r}) = 1;
        FI45n(r).timeSteps(i) = 2290;
        FI45n(r).session(i).times = [diff([1; FI45.training.session(i).reward_time_stamps{r}])];
        
        VI45n(r).session(i).training.lever_binned = zeros(1, 2290);
        VI45n(r).session(i).training.lever_binned(VI45.training.session(i).lever_time_stamps{r})= 1;
        VI45n(r).session(i).training.reward_binned = zeros(1, 2290);
        VI45n(r).session(i).training.reward_binned(VI45.training.session(i).reward_time_stamps{r})= 1;
        VI45n(r).timeSteps(i) = 2290;
        VI45n(r).session(i).times = [diff([1; VI45.training.session(i).reward_time_stamps{r}])];
        
        FR.actRate(r,i) = sum(FR20n(r).session(i).training.lever_binned)./FR20n(r).timeSteps(i);
        VR.actRate(r,i) = sum(VR20n(r).session(i).training.lever_binned)./VR20n(r).timeSteps(i);
        FI.actRate(r,i) = sum(FI45n(r).session(i).training.lever_binned)./FI45n(r).timeSteps(i);
        VI.actRate(r,i) = sum(VI45n(r).session(i).training.lever_binned)./VI45n(r).timeSteps(i);
        
    end
end

for d = 1:3 % 3 deval sessions
    FR.test(:,:,d) = [FR20.test.cycle(d).lever_presses.valued FR20.test.cycle(d).lever_presses.devalued];
    VR.test(:,:,d) = [VR20.test.cycle(d).lever_presses.valued VR20.test.cycle(d).lever_presses.devalued];
    
    FI.test(:,:,d) = [FI45.test.cycle(d).lever_presses.valued FI45.test.cycle(d).lever_presses.devalued];
    VI.test(:,:,d) = [VI45.test.cycle(d).lever_presses.valued VI45.test.cycle(d).lever_presses.devalued];
end
FR20 = FR;
VR20 = VR;
FI45 = FI;
VI45 = VI;

save('all_data_cleaned.mat','FI45','VI45','FI45n','VI45n','FR20','VR20','FR20n','VR20n')

end

function example

%% unpack data
load('example_rats.mat')

% assign diff variables to diff structures
FR20 = schedule.FR;
VR20 = schedule.RR;
FI45 = schedule.FI;
VI45 = schedule.RI;

% timesteps are in centiseconds
for i = 1:4 % for all sessions, convert timestamps to secs, get rid of zeros
    FR20.session(i).training.lever_time_stamps(FR20.session(i).training.lever_time_stamps==0) =[];
    FR20.session(i).training.lever_time_stamps = ceil(FR20.session(i).training.lever_time_stamps./100); % centisecs --> decisecond
    FR20.session(i).training.reward_time_stamps(FR20.session(i).training.reward_time_stamps==0) =[];
    FR20.session(i).training.reward_time_stamps = ceil(FR20.session(i).training.reward_time_stamps./100);
    
    FR20.more2taps(i) = size(find(diff(FR20.session(i).training.lever_time_stamps)==0),1); % how many more than 2 taps in a decisecond
    FR20.session(i).training.lever_time_stamps = unique(FR20.session(i).training.lever_time_stamps);
    
    VR20.session(i).training.lever_time_stamps(VR20.session(i).training.lever_time_stamps==0) =[];
    VR20.session(i).training.lever_time_stamps = ceil(VR20.session(i).training.lever_time_stamps./100); % centisecs --> deciseconds
    VR20.session(i).training.reward_time_stamps(VR20.session(i).training.reward_time_stamps==0) =[];
    VR20.session(i).training.reward_time_stamps = ceil(VR20.session(i).training.reward_time_stamps./100);
    
    VR20.more2taps(i) = size(find(diff(VR20.session(i).training.lever_time_stamps)==0),1); % how many more than 2 taps in a decisecond
    VR20.session(i).training.lever_time_stamps = unique(VR20.session(i).training.lever_time_stamps);
    
end

for i = 1:3 % for all sessions, convert timestamps to secs
    FI45.session(i).training.lever_time_stamps(FI45.session(i).training.lever_time_stamps==0) =[];
    FI45.session(i).training.lever_time_stamps = ceil(FI45.session(i).training.lever_time_stamps./100); % centisecs --> decisecond
    FI45.session(i).training.reward_time_stamps(FI45.session(i).training.reward_time_stamps==0) =[];
    FI45.session(i).training.reward_time_stamps = ceil(FI45.session(i).training.reward_time_stamps./100);
    
    FI45.more2taps(i) = size(find(diff(FI45.session(i).training.lever_time_stamps)==0),1); % how many more than 2 taps in a decisecond
    FI45.session(i).training.lever_time_stamps = unique(FI45.session(i).training.lever_time_stamps);
    
    
    VI45.session(i).training.lever_time_stamps(VI45.session(i).training.lever_time_stamps==0) =[];
    VI45.session(i).training.lever_time_stamps = ceil(VI45.session(i).training.lever_time_stamps./100); % centisecs --> decisecond
    VI45.session(i).training.reward_time_stamps(VI45.session(i).training.reward_time_stamps==0) =[];
    VI45.session(i).training.reward_time_stamps = ceil(VI45.session(i).training.reward_time_stamps./100);
    
    VI45.more2taps(i) = size(find(diff(VI45.session(i).training.lever_time_stamps)==0),1); % how many more than 2 taps in a decisecond
    VI45.session(i).training.lever_time_stamps = unique(VI45.session(i).training.lever_time_stamps);
    
end

%% reorganize data into: [0 1] depending on if reward was observed

for i = 1:4
    FR20.session(i).training.lever_binned = zeros(1, max(FR20.session(i).training.lever_time_stamps));
    FR20.session(i).training.lever_binned(FR20.session(i).training.lever_time_stamps) = 1;
    FR20.session(i).training.reward_binned = zeros(1, max(FR20.session(i).training.lever_time_stamps));
    FR20.session(i).training.reward_binned(FR20.session(i).training.reward_time_stamps) = 1;
    FR20.timeSteps(i) = length(FR20.session(i).training.lever_binned);
    
    k = 1;outTimes = [1 FR20.session(i).training.reward_time_stamps];
    for t = 1:length(outTimes)-1
        numAct(k) = sum(FR20.session(i).training.lever_binned(outTimes(t):outTimes(t+1))); % sum up the number of actions between outTimes
        k = k+1;
    end
    
    VR20.session(i).training.lever_binned = zeros(1, max(VR20.session(i).training.lever_time_stamps));
    VR20.session(i).training.lever_binned(VR20.session(i).training.lever_time_stamps) = 1;
    VR20.session(i).training.reward_binned = zeros(1, max(VR20.session(i).training.lever_time_stamps));
    VR20.session(i).training.reward_binned(VR20.session(i).training.reward_time_stamps) = 1;
    VR20.timeSteps(i) = length(VR20.session(i).training.lever_binned);
    k = 1; outTimes = [1; VR20.session(i).training.reward_time_stamps];
    for t = 1:length(outTimes)-1
        numAct(k) = sum(VR20.session(i).training.lever_binned(outTimes(t):outTimes(t+1))); % sum up the number of actions between outTimes
        k = k+1;
    end
    VR20.session(i).actions = numAct;
end

for i = 1:3
    FI45.session(i).training.lever_binned = zeros(1, max(FI45.session(i).training.lever_time_stamps));
    FI45.session(i).training.lever_binned(FI45.session(i).training.lever_time_stamps) = 1;
    FI45.session(i).training.reward_binned = zeros(1, max(FI45.session(i).training.lever_time_stamps));
    FI45.session(i).training.reward_binned(FI45.session(i).training.reward_time_stamps) = 1;
    FI45.timeSteps(i) = length(FI45.session(i).training.lever_binned);
    
    VI45.session(i).training.lever_binned = zeros(1, max(VI45.session(i).training.lever_time_stamps));
    VI45.session(i).training.lever_binned(VI45.session(i).training.lever_time_stamps) = 1;
    VI45.session(i).training.reward_binned = zeros(1, max(VI45.session(i).training.lever_time_stamps));
    VI45.session(i).training.reward_binned(VI45.session(i).training.reward_time_stamps) = 1;
    VI45.timeSteps(i) = length(VI45.session(i).training.lever_binned);
    VI45.session(i).times = [diff([1; VI45.session(i).training.reward_time_stamps])];
end

% timeSteps (unrolled)
timeSteps = [FR20.timeSteps(1) FR20.timeSteps(2) FR20.timeSteps(3);
    VR20.timeSteps(1) VR20.timeSteps(2) VR20.timeSteps(3);
    FI45.timeSteps(1) FI45.timeSteps(2) FI45.timeSteps(3);
    VI45.timeSteps(1) VI45.timeSteps(2) VI45.timeSteps(3)];

% mean lever presses per min (unrolled)
test_lever_rates.valued = [FR20.session(1).test_lever_rates.valued FR20.session(2).test_lever_rates.valued FR20.session(3).test_lever_rates.valued;
    VR20.session(1).test_lever_rates.valued VR20.session(2).test_lever_rates.valued VR20.session(3).test_lever_rates.valued;
    FI45.session(1).test_lever_rates.valued FI45.session(2).test_lever_rates.valued FI45.session(3).test_lever_rates.valued;
    VI45.session(1).test_lever_rates.valued VI45.session(2).test_lever_rates.valued VI45.session(3).test_lever_rates.valued];
test_lever_rates.devalued = [FR20.session(1).test_lever_rates.devalued FR20.session(2).test_lever_rates.devalued FR20.session(3).test_lever_rates.devalued;
    VR20.session(1).test_lever_rates.devalued VR20.session(2).test_lever_rates.devalued VR20.session(3).test_lever_rates.devalued;
    FI45.session(1).test_lever_rates.devalued FI45.session(2).test_lever_rates.devalued FI45.session(3).test_lever_rates.devalued;
    VI45.session(1).test_lever_rates.devalued VI45.session(2).test_lever_rates.devalued VI45.session(3).test_lever_rates.devalued];


save('example_rats_cleaned.mat','FR20','VR20','FI45','VI45','timeSteps','test_lever_rates'); % save


end