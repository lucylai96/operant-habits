function plot_data(fig)
%% basic data analysis and plotting (Eric Garr - JHU)
% plot settings
% Written by: Lucy Lai, Last updated: Mar 2021

addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools'); % various plotting tools
addpath('./params'); % parameters

map = habitColors; % colormap

type = {'FR' 'VR' 'FI' 'VI'};
load('all_data_cleaned.mat')
load('data.mat') % more compact form

switch fig
    case 'lever_sessions'
        %% 4 schedules action rate over time
        % ratio plots
        figure; hold on;
        h(1,:) = errorbar(mean(FR20.actRate),sem(FR20.actRate,1),'LineWidth',2,'Color',map(1,:)); %plot(mean(FR20.actRate));
        h(2,:) = errorbar(mean(VR20.actRate),sem(VR20.actRate,1),'LineWidth',2,'Color',map(2,:)); %plot(mean(VR20.actRate));
        
        % interval plots
        h(3,:) = errorbar(mean(FI45.actRate),sem(FI45.actRate,1),'LineWidth',2,'Color',map(3,:)); %plot(mean(FI45.actRate));
        h(4,:) = errorbar(mean(VI45.actRate),sem(VI45.actRate,1),'LineWidth',2,'Color',map(4,:)); %plot(mean(VI45.actRate));
        plot([2.5 10.5 20.5;2.5 10.5 20.5],[ylim' ylim' ylim'],'k--') %
        legend(h,'FR20','VR20','FI45','VI45','deval'); legend('boxoff');
        ylabel('lever presses/sec'); xlabel('sessions'); prettyplot;
        
    case 'deval_group'
        %% devalulation (GROUP)
        figure; hold on;
        for d = 1:3
            subplot 221; hold on;
            [b e] = barwitherr(sem(FR20.test(:,:,d),1),d, mean(FR20.test(:,:,d)),'FaceColor',map(1,:));
            b(2).FaceColor = [1 1 1];
            b(2).EdgeColor = map(1,:);
            b(2).LineWidth = 2;
            e.Color = map(1,:);
            e.LineWidth = 2;
            title('FR20')
            
            subplot 222; hold on;
            [b e] = barwitherr(sem(VR20.test(:,:,d),1),d, mean(VR20.test(:,:,d)),'FaceColor',map(2,:));
            b(2).FaceColor = [1 1 1];
            b(2).EdgeColor = map(2,:);
            b(2).LineWidth = 2;
            e.Color = map(2,:);
            e.LineWidth = 2;
            title('VR20')
            
            subplot 223; hold on;
            [b e] = barwitherr(sem(FI45.test(:,:,d),1),d, mean(FI45.test(:,:,d)),'FaceColor',map(3,:));
            b(2).FaceColor = [1 1 1];
            b(2).EdgeColor = map(3,:);
            b(2).LineWidth = 2;
            e.Color = map(3,:);
            e.LineWidth = 2;
            title('FI45')
            
            subplot 224; hold on;
            [b e] = barwitherr(sem(VI45.test(:,:,d),1),d, mean(VI45.test(:,:,d)),'FaceColor',map(4,:));
            b(2).FaceColor = [1 1 1];
            b(2).EdgeColor = map(4,:);
            b(2).LineWidth = 2;
            e.Color = map(4,:);
            e.LineWidth = 2;
            title('VI45')
        end
        subplot 221; hold on;
        ylabel('lever presses/sec'); xlabel('deval session #');
        legend('valued','devalued'); legend('boxoff');
        equalabscissa(2,2)
        subprettyplot(2,2)
        
    case 'deval_indv'
        % individual animals bar plots
        figure; hold on;
        k = [1 5 9]; % subplot indices
        for d = 1:3 % each devaluation day
            subplot(3,4,k(d)); hold on;
            b = bar(FR20.test(:,:,d),'FaceColor',map(1,:));
            b(2).FaceColor = [1 1 1];
            b(2).EdgeColor = map(1,:);
            b(2).LineWidth = 2;
            
            subplot(3,4,k(d)+1); hold on;
            b  = bar(VR20.test(:,:,d),'FaceColor',map(2,:));
            b(2).FaceColor = [1 1 1];
            b(2).EdgeColor = map(2,:);
            b(2).LineWidth = 2;
            
            subplot(3,4,k(d)+2); hold on;
            b = bar(FI45.test(:,:,d),'FaceColor',map(3,:));
            b(2).FaceColor = [1 1 1];
            b(2).EdgeColor = map(3,:);
            b(2).LineWidth = 2;
            
            subplot(3,4,k(d)+3); hold on;
            b = bar(VI45.test(:,:,d),'FaceColor',map(4,:));
            b(2).FaceColor = [1 1 1];
            b(2).EdgeColor = map(4,:);
            b(2).LineWidth = 2;
            
        end
        subplot 341;  title('FR20'); ylabel('session 2');legend('valued','devalued'); legend('boxoff');
        subplot 342;  title('VR20');
        subplot 343;  title('FI45')
        subplot 344;  title('VI45')
        subplot 345; ylabel('session 10');
        subplot 349; ylabel('session 20');
        subplot(3,4,12)
        ylabel('lever presses/sec'); xlabel('rat #')
        subprettyplot(3,4)
        
        % devalued - valued
        figure; hold on;
        k = [1 5 9]; % subplot indices
        for d = 1:3 % each devaluation day
            subplot(3,4,k(d)); hold on;
            bar(FR20.delta(:,d),'FaceColor',map(1,:));
            
            subplot(3,4,k(d)+1); hold on;
            bar(VR20.delta(:,d),'FaceColor',map(2,:));
            
            subplot(3,4,k(d)+2); hold on;
            bar(FI45.delta(:,d),'FaceColor',map(3,:));
            
            subplot(3,4,k(d)+3); hold on;
            bar(VI45.delta(:,d),'FaceColor',map(4,:));
        end
        
        subplot 341;  title('FR20'); ylabel('session 2');legend('valued - devalued'); legend('boxoff');
        subplot 342;  title('VR20');
        subplot 343;  title('FI45')
        subplot 344;  title('VI45')
        subplot 345; ylabel('session 10');
        subplot 349; ylabel('session 20');
        subplot(3,4,12)
        ylabel('lever presses/sec'); xlabel('rat #')
        subprettyplot(3,4)
        sgtitle('\Delta press rate (pre-post)','FontSize',20)
        equalabscissa(3,4)
        why
        
    case 'cumsum'
        r = 1; % which example rat? (but remember not all rats did the same thing)
        s = 5; % which example session?
        num = 1000; % last how many # trials to look at
        
        figure; hold on;
        for i = 1:length(type)
            data = schedule(i).rat(r);
            a = data.session(s).training.lever_binned;
            x = data.session(s).training.reward_binned;
            cs = cumsum(a(end-num:end));
            h(i,:) = plot(cs+5,'Color',map(i,:),'LineWidth',2);
            xplot = find(x(end-num:end)==1);
            line([xplot' xplot'+50]',[cs(xplot)' cs(xplot)']'+5,'LineWidth',2,'Color','k');
        end
        
        legend(h,type);
        legend('boxoff')
        xlabel('time (s)')
        ylabel('cumulative # actions')
        prettyplot
        
    case 'rates'
        %% press and reward rates over time
        r = 1; % which example rat? (but remember not all rats did the same thing)
        sp = [1 2;3 4;5 6;7 8];
        figure; hold on;
        for i = 1:length(type)
            data = schedule(i).rat(r);
            a = []; x = []; % unroll
            
            for s = 1:20 % 20 sessions
                a = [a data.session(s).training.lever_binned];
                x = [x data.session(s).training.reward_binned];
            end
            
            win = 100; % # seconds moving window
            subplot(4,2,sp(i,1)); hold on;
            plot(movmean(a, win,'Endpoints','shrink'),'LineWidth',1.5,'Color',map(i,:))
            plot([cumsum(data.timeSteps);cumsum(data.timeSteps)],repmat(ylim',1,20),'k--')
            prettyplot; axis tight
            
            subplot(4,2,sp(i,2)); hold on;
            plot(movmean(x, win,'Endpoints','shrink'),'LineWidth',1.5,'Color',map(i,:))
            plot([cumsum(data.timeSteps);cumsum(data.timeSteps)],repmat(ylim',1,20),'k--')
            prettyplot; axis tight
            if i==1
                subplot(4,2,sp(i,1)); 
                ylabel('lever press rate (/sec)')
                xlabel('time (s)')
                subplot(4,2,sp(i,2));
                ylabel('reward rate (/sec)')
            end
            
        end % each schedule
end


end