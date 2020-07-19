function habitColors
% color scheme/aesthetic palette for operant-habit project/paper
addpath '/Users/lucy/Google Drive/Harvard/Projects/mat-tools/'

map = brewermap(4,'*RdBu');
temp = map(3,:);
map(3,:) = map(4,:);
map(4,:) = temp;  % swap colors so darker ones are fixed
set(0, 'DefaultAxesColorOrder', map) % first three rows
set(0, 'DefaultLineLineWidth', 1.5) % set line width
end