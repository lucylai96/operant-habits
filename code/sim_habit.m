function results = sim_habit(params,sch,r)
% simulate data with parameters using the actual conditions of the data

if nargin <1
    params = [0.001, 0.001, 0.001, 2];
end

%% instantiation
map = habitColors;               % set color scheme
load('all_data_cleaned.mat');    % load data
load('data.mat')
type = {'FR' 'VR' 'FI' 'VI'};

%for sch = 1:length(type)
%    for r = 1:length(schedule(sch).rat)
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

[nLL,results] = habitAgentFit(params, input);

beta = linspace(0.1,15,50);
%[R,V] = blahut_arimoto(results.normps',results.theta,beta);
%figure; hold on;
%plot(R,V,'b-','LineWidth',3);
%[R,V] = blahut_arimoto(results.normps',results.Q,beta);
%plot(R,V,'r-','LineWidth',3);
%plot(results.mi,results.avgr,'ko'); hold on;
%plot(results.mi(end),results.avgr(end),'r.','MarkerSize',50)
% xlabel('Policy complexity')
%ylabel('Average reward')

%figure; hold on;
%plot(results.ecost,results.avgr,'ko'); hold on;
%plot(results.ecost(end),results.avgr(end),'r.','MarkerSize',50)

%end
%end
end