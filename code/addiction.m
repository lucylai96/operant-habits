function addiction

% normal TD example
V(1) = 0; TD(1) = 1;
alpha = 0.1;
r = 1;
for t = 2:40
    TD(t) = (r - V(t-1));
    V(t) = V(t-1) + alpha*TD(t);
end 
figure; hold on;
subplot 211; hold on;
plot(1:40,V,'ro-');
plot(1:40,ones(1,40),'g')
xlabel('Timesteps')
ylabel('V_{CS}')
legend('V_{CS}','r_{US}')

subplot 212;
plot(1:40,TD,'bo-');
ylabel('\delta')
xlabel('Timesteps')
sgtitle('Normal TD learning')

% addiction
V(1,1) = 0; TD(1) = [];
V(2,1) = 0; 
D = [0 10];
alpha = 0.1;
r = 1;
for t = 2:40
    for s = 1:2
    TD(t) = max((r - V(s,t-1) + D(s)), D(s));
    V(s,t) = V(s,t-1) + alpha*TD(t);
    end
end 
figure; hold on;
subplot 211; hold on;
plot(1:40,V(1,:),'ko-');
plot(1:40,V(2,:),'ro-');
plot(1:40,ones(1,40),'g')
xlabel('Timesteps')
ylabel('Value')
legend('V_{food}','V_{cocaine}','r')

subplot 212;
plot(1:40,TD,'bo-');
ylabel('\delta')
xlabel('Timesteps')
sgtitle('Addiction')

end 