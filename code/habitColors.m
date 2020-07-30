function map = habitColors(type)
% color scheme/aesthetic palette for operant-habit project/paper
addpath '/Users/lucy/Google Drive/Harvard/Projects/mat-tools/'

map = brewermap(4,'*RdBu');
temp = map(3,:);
map(3,:) = map(4,:);
map(4,:) = temp;  % swap colors so darker ones are fixed

if nargin > 1
    if type == 'FR'
        set(0, 'DefaultAxesColorOrder', map(1,:))
    elseif type == 'VR'
        set(0, 'DefaultAxesColorOrder', map(2,:))
    elseif type == 'FI'
        set(0, 'DefaultAxesColorOrder', map(3,:))
    elseif type == 'VI'
        set(0, 'DefaultAxesColorOrder', map(4,:))
    end
    
    
else
set(0, 'DefaultAxesColorOrder', map) % first three rows
set(0, 'DefaultLineLineWidth', 1.5) % set line width
end

end