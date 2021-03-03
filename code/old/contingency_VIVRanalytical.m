addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools/');

map = brewermap(4,'*RdBu');
temp = map(3,:);
map(3,:) = map(4,:);
map(4,:) = temp;  % swap colors so darker ones are fixed
set(0, 'DefaultAxesColorOrder', map) % first three rows
set(0, 'DefaultLineLineWidth', 1.5) % first three rows
load('VI_empirical.mat')
load('VR_empirical.mat')
load('FR_empirical.mat')

figure; 
subplot 121; hold on;
la = linspace(0.1,1,20); % action rate (presses per second)

plot(la,FR.I','Color',map(1,:))
plot(la,VR.I','Color',map(2,:))
plot(la,VI.I','Color',map(4,:))
title('mutual information'); prettyplot 

subplot 122; hold on;
plot(la,FR.C','Color',map(1,:))
plot(la,VR.C','Color',map(2,:))
plot(la,VI.C','Color',map(4,:))
legend('FR','VR','VI')
title('contingency'); prettyplot

%% VR schedules
%close all
%clear all;

map = brewermap(9,'Blues');
set(0, 'DefaultAxesColorOrder', map(3:2:9,:)) % first three rows

%precondition is that action rate has to be at least equal to or greater
%than outcome rate (because in VR schedules, you will never have an action
%rate that is less than outcome rate)
r = [2 5 10 20]; % ratio parameter (determines outcome rate)
%r = linspace(1,50,50);; % ratio parameter (determines outcome rate)
k = 1;
m = 1./r;
for i = 1:length(m)
    a(i,:) = linspace(0,1,50);  %lambda_a action rate 
    o(i,:) = m(i).*a(i,:);  %lambda_a action rate 
    beta(i,:) = -r(i)./a(i,:);
    hoa(i,:) = 0;%k-log(a(i,:)/r(i));
    ho(i,:) = k-log(o(i,:));
   
end

figure; hold on;
subplot 131;
plot(a',o','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('outcome rate \lambda_o')
axis square; prettyplot;
subplot 132;
plot(beta',o','LineWidth', 1.5) % for every positive change in avg reward rate, beta becomes larger (pay more of a cost)
xlabel('inverse temperature \beta')
ylabel('outcome rate \lambda_o')
%axis([0 1 0 100])
axis square; prettyplot;
subplot 133;
plot(r',beta') % for every positive change in reward,
ylabel('inverse temperature \beta')
xlabel('ratio parameter R')
axis square; prettyplot;

%at one action rate
%subplot 132;
%plot(beta(:,25)',o(:,25)','LineWidth', 1.5) % for every positive change in avg reward rate, beta becomes larger (pay more of a cost)


figure; hold on;
subplot 121; hold on;
tau = 1./beta;
plot(a',tau','LineWidth', 1.5)
%text(1.1,0, 'paying less for complexity / low temp','FontSize',14)
%text(1.1,0.5, 'paying more for complexity / high temp','FontSize',14)
xlabel('action rate \lambda_a')
ylabel('resource constraint \delta')
axis square
%axis([0 1 0 0.5])
legend('VR2','VR5','VR10','VR20')
legend('boxoff')
prettyplot

subplot 122;  hold on;
plot(a',beta','LineWidth', 1.5)
%text(1.1,0, 'more habit / high temp','FontSize',14) % but even w higher beta, your log P may get more peaky
%text(1.1,90, 'less habit / low temp','FontSize',14)
xlabel('action rate \lambda_a')
ylabel('inv temp \beta')
axis square
%axis([0 1 0 90])
legend('VR2','VR5','VR10','VR20')
legend('boxoff')
prettyplot

figure; hold on;
%hoa = k-log(a);
%ho = k-log(o);
ho = ho.*ones(1,length(a));
hoa = hoa.*ones(1,length(a));
subplot 131; hold on;
plot(a',ho','--','LineWidth', 1.5);
plot(a',hoa','k','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('conditional entropy H_{\lambda_o|\lambda_a}')
legend('VR2','VR5','VR10','VR20','Location','southeast')
legend('boxoff')
axis([0 1 -0.1 7])
axis square
prettyplot

% mutual information between each action and outcome grows as a function of
% the agent's action rate. this is because doing more actions per unit time
% gives you more information about when outcome will appear based on ratio
% parameter
subplot 132
I = ho-hoa;
%I(I<0) = NaN;
%I2 = log(a./o2);
plot(a',I','LineWidth', 1.5)
axis([0 1 0 7])
xlabel('action rate \lambda_a')
ylabel('I(\lambda_o;\lambda_a)')
axis square
prettyplot

% VR schedules, A-O contingency is strong, because the faster they respond,
% the sooner on average a reinforcer is delivered.
subplot 133
C = I./ho;
plot(a',C','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('A-O rate contingency')
axis([0 1 0 1])
axis square
prettyplot

suptitle('Ratio Schedules')


% look at how sam calculated the reward-complexity curve
% for a given constraint C
 
% figure; % i think these are wrong..
% %for a given constraint, what is my return?
% % reward-complexity
% subplot 121; hold on;
% %50 diff action rates, choose one
% plot(I',o'-min(o(:))','LineWidth', 2)
% plot(tau.*I',o','LineWidth', 2)
% 
% plot(I',tau','LineWidth', 2)
% xlabel('policy complexity I(\lambda_o;\lambda_a)')
% ylabel('V^\pi or outcome rate \lambda_o ')
% title('reward-complexity curve')
% axis square
% plot(a',1./beta','LineWidth', 2)
% xlabel('action rate \lambda_a')
% ylabel('inverse temperature \beta')
% text(1.1,0, 'more habit','FontSize',14)
% text(1.1,0.5, 'less habit','FontSize',14)
% legend('VR2','VR5','VR10','VR20','Location','northeast')
% legend('boxoff')
% axis square
% prettyplot

% for a given resource constraint, and equated outcome rates, what is the expected action rate? 
% fb = 50;
% fo = 0.01;
% [~,fixed_beta] = min(abs(beta-fb)');
% [~,fixed_o] = min(abs(o-fo)');
% for i = 1:size(beta,1)
% a(i,fixed_beta(i));
% o(i,fixed_beta(i))
% end


%% VI schedules
clear all
map = brewermap(9,'Reds');
set(0, 'DefaultAxesColorOrder', map(3:2:9,:)) % first three rows

a = linspace(0,1,50);
t = [2 5 10 20];
o = 1./t;
k = 1;
B = .5; % parameter that controls when saturation occurs

hoa = zeros(1,length(a));
for i = 1:length(o)
    m(i,:)= B./(t(i).*a+B).^2;
    vi= a>o(i); % outcome rate o is the lesser of (action rate, vi parameter), the experienced outcome rate
    
    % if action rate < interval arming parameter
    %blah = (-(1./a))-(((k-log(a))./(a+exp(-a)))-((a.*(1-exp(-a)).*(k-log(a)))./((a+exp(-a)).^2))-(1./(a+exp(-a))));
    b1 = -(1./m(i,:)).*(1./a);
    beta(i,vi==0) = b1(vi==0);
    
    ho1 = k-log(a);
    ho(i,vi==0) = ho1(vi==0);
    hoa1 = 0;
    hoa(i,vi==0) = 0;
    
    % if action rate > interval arming parameter
    %b2 = (1./m(i,:)).*((exp(a).*(1+a).*(log(o(i))-k))./((1 + a.*exp(a)).^2));
    b2 = 0;%-(1./m(i,:)).*(1./o(i));
    beta(i,vi==1) = 0;%b2(vi==1);
    
    ho2 = k-log(o(i))*ones(1,length(a)); % governed by outcome rate
    ho(i,vi==1) = ho2(vi==1);
    hoa2 = k-log(o(i))*ones(1,length(a))-0.05;%(a.*(k-log(o(i))))./(a+exp(-a)); % action rate > interval arming parameter
    hoa(i,vi==1) = hoa2(vi==1);
    
   
 %   hoa1(i,:) = (o(i).*(k-log(o(i))))./(o(i)+exp(-o(i)));
  %  hoa(i,:) = (a.*(k-log(v)))./(a+exp(-a));
    
  %  v = min(o(i),a);
  %  ho(i,:) = k-log(o(i))*ones(1,length(a));
end


t = t'.*ones(1,length(a));  %lambda_a action rate 
a = [a;a;a;a];
obs_o = a./(t.*a+B);
figure; hold on;
subplot 131;
plot(a',obs_o','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('outcome rate \lambda_o')
axis square; prettyplot;
subplot 132;
plot(obs_o',beta','LineWidth', 1.5) % for every positive change in avg reward rate, beta becomes larger (pay more of a cost)
ylabel('resource constraint \delta')
xlabel('outcome rate \lambda_o')
axis([0 1 0 100])
axis square; prettyplot;
subplot 133;
plot(t',beta','.') % for every positive change in reward,
ylabel('resource constraint \delta')
xlabel('interval parameter T')
axis square; prettyplot;


figure; hold on;
subplot 121; hold on;
tau = 1./beta;
plot(a',tau','LineWidth', 1.5)
%text(1.1,0, 'paying less for complexity / low temp','FontSize',14)
%text(1.1,0.5, 'paying more for complexity / high temp','FontSize',14)
xlabel('action rate \lambda_a')
ylabel('resource constraint \delta')
axis square
%axis([0 1 0 0.5])
legend('VI2','VI5','VI10','VI20')
legend('boxoff')
prettyplot

subplot 122;  hold on;
plot(a',beta','LineWidth', 1.5)
%text(1.1,0, 'more habit / high temp','FontSize',14) % but even w higher beta, your log P may get more peaky
%text(1.1,90, 'less habit / low temp','FontSize',14)
xlabel('action rate \lambda_a')
ylabel('inv temp \beta')
axis square
%axis([0 1 0 90])
legend('VI2','VI5','VI10','VI20')
legend('boxoff')
prettyplot

figure;hold on
subplot 131; hold on;
plot(a',ho','--','LineWidth', 1.5)
plot(a',hoa','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('conditional entropy H_{\lambda_o|\lambda_a}')
axis([0 1 0 7])
axis square
legend('VI2','VI5','VI10','VI20','Location','southeast')
legend('boxoff')
prettyplot

subplot 132
I = ho-hoa;
%I(I<-0.05) = NaN;
plot(a',I','LineWidth', 1.5)
axis([0 1 0 7])
xlabel('action rate \lambda_a')
ylabel('I(\lambda_o;\lambda_a)')
axis square
prettyplot

% 
subplot 133
C = I./ho;
plot(a',C','LineWidth', 1.5)
xlabel('action rate \lambda_a')
ylabel('A-O rate contingency')
axis([0 1 0 1])
axis square
prettyplot

suptitle('Interval Schedules')

figure;
%for a given constraint, what is my return?
% reward-complexity
I(:,1) = 10;
subplot 121; hold on;
%new_o = (-1./beta.*obs_o);
%new_o = new_o-min(new_o(:));
plot(I',obs_o','LineWidth', 2)
xlabel('policy complexity I(\lambda_o;\lambda_a)')
ylabel('V^\pi or outcome rate \lambda_o ')
title('reward-complexity curve')
axis square
prettyplot

subplot 122;
plot(a',1./beta','LineWidth', 2)
xlabel('action rate \lambda_a')
ylabel('inverse temperature \beta')
text(1.1,0, 'more habit','FontSize',14)
text(1.1,0.5, 'less habit','FontSize',14)
legend('VI2','VI5','VI10','VI20','Location','northeast')
legend('boxoff')
axis([0 1 0 0.5])
axis square
prettyplot