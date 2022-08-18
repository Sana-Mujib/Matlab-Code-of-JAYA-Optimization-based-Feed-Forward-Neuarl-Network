% *************************************************************************************************************
%                  Source Code of JAYA Optimization based Feed-Forward
%                  Neural Network By Sana Mujeeb, 14h August, 2022 

% Cite: Wang S, Rao RV, Chen P, Zhang Y, Liu A, Wei L. Abnormal breast detection in 
% mammogram images by feed-forward neural network trained by Jaya algorithm. 
% Fundamenta Informaticae. 2017 Jan 1;151(1-4):191-211.
% *************************************************************************************************************
% Enjoy JAYA-ANN! 
clc;
% Generating random correlated data 
mu = 50;
sigma = 5;
M = mu + sigma * randn(300, 2);
R = [1, 0.75; 0.75, 1];
L = chol(R);
M = M*L;
x = M(:,1);  % Example Inputs, Replace by your data inputs for your own experiments
y = M(:,2); % Example labels, Replace by your data labels for your own experiments
%% JAYA algorithms
%% Problem Definition
pop = 30;               % Population size
% Min-max normalization of data
m = max(x); mn = min(x); mm = m-mn;
X = ((x-mn)/mm); Y = ((x-mn)/mm);
% 90%:10% splitting of data for training and testing 
sz = (ceil(size(X,1))*0.9);
inputs = (X(1:sz))';
targets = (Y(1:sz))';
XTest = (X(sz+1:end))';
YTest = Y(sz+1:end)';
% number of neurons
n = 4;
tic;
% create a neural network
net = feedforwardnet(n);
% configure the neural network for this dataset
net = configure(net, inputs, targets);
% Denormalizaion and Prediction by FNN
FNN_Pred = ((net(XTest))' * mm) + mn;
sz = n^2 + n + n + 1; % Number of design variables i.e., no. of weights in FNN                
maxGen = 30;            % Maximum number of iterations
mini = repmat(-1,1,sz); % Lower Bound of Variables
maxi = ones(1,sz);      % Upper Bound of Variables  
objective = @(x) NMSE(x, net, inputs, targets);      % Cost Function
%% initialize
[row,var] = size(mini);
x = zeros(pop,var);
fnew = zeros(pop,1);
f = zeros(pop,1);
fopt= zeros(pop,1);
xopt=zeros(1,var);
%%  Generation and Initialize the positions 
for i=1:var
    x(:,i) = mini(i)+(maxi(i)-mini(i))*rand(pop,1);
end
for i=1:pop
    f(i) = objective(x(i,:));
end
%%  Main Loop
gen=1;
fprintf('Best Cost per Iteration of JAYA Opimization Algorithm \n');
while(gen <= maxGen)
    [row,col]=size(x);
    [t,tindex]=min(f);
    Best=x(tindex,:);
    [w,windex]=max(f);
    worst=x(windex,:);
    xnew=zeros(row,col);
    for i=1:row
        for j=1:col
            xnew(i,j)=(x(i,j))+rand*(Best(j)-abs(x(i,j))) - (worst(j)-abs(x(i,j)));  % 
        end
    end 
    for i=1:row
        xnew(i,:) = max(min(xnew(i,:),maxi),mini);   
        fnew(i,:) = objective(xnew(i,:));
    end
    for i=1:pop
        if(fnew(i)<f(i))
            x(i,:) = xnew(i,:);
            f(i) = fnew(i);
        end
    end
    fnew = []; xnew = [];
    [fopt(gen),ind] = min(f);
    xopt(gen,:)= x(ind,:);
    gen = gen+1; 
    disp(['Iteration No. = ',num2str(gen-1), ',   Best Cost = ',num2str(min(f))])
end
%% 
[val,ind] = min(fopt);
Fes = pop*ind;
disp(['Optimum value = ',num2str(val,10)])
 figure;
 plot(fopt,'LineWidth', 2);
 xlabel('Itteration');
 ylabel('Best Cost');
 legend('JAYA');
 disp(' ' );
% Setting optimized weights and bias in network
net = setwb(net, Best');
% Denormalizaion and Prediction by JAYA_FNN
JAYA_FNN_Pred = ((net(XTest))' * mm) + mn;
YTest = (YTest * mm) + mn;
JAYA_FNN_Execution_Time_Seconds = toc 
% Plotting prediction results
figure;
plot(YTest,'LineWidth',2, 'Marker','diamond', 'MarkerSize',8);
hold on;
plot(FNN_Pred, 'LineWidth',2, 'Marker','x', 'MarkerSize',8);
plot(JAYA_FNN_Pred, 'LineWidth',2, 'Marker','pentagram', 'MarkerSize',8);
title('JAYA Optimization based Feed-Forward Neural Network');
xlabel('Time Interval');
ylabel('Values');
legend('Actual Values', 'FNN Predictions', 'JAYA-FNN Predictions');
hold off;
% Performance Evaluaion of FNN and JAYA-FNN
fprintf('Performance Evaluaion of FNN and JAYA-FNN using Normalized Root Mean Square Error \n');
NRMSE_FNN = (abs( sqrt( mean(mean((FNN_Pred - YTest).^2) )) )) / (max(YTest)-min(YTest))
NRMSE_JAYA_FNN = (abs( sqrt( mean(mean((JAYA_FNN_Pred - YTest).^2) ) ) )) / (max(YTest)-min(YTest))


% Objective Function for minimizing normalized mean square error of FNN by
% updation of nework's weights and biases
function [f] = NMSE(wb, net, input, target)
% wb is the weights and biases row vector obtained from the genetic algorithm.
% It must be transposed when transferring the weights and biases to the network net.
 net = setwb(net, wb');
% The net output matrix is given by net(input). The corresponding error matrix is given by
 error = target - net(input);
% The mean squared error normalized by the mean target variance is
 f = (mean(error.^2)/mean(var(target',1)));
% It is independent of the scale of the target components and related to the Rsquare statistic via
% Rsquare = 1 - NMSEcalc ( see Wikipedia)
 end