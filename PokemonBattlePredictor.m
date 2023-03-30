% 4H03 Project
% Tony Fan, fant6, 200003466
% Hamdan Basharat, basham1, 400124515
% Julian Morrone, morronej, 400138570
% Pokemon Battle Predictor

clear variables;
close all;
clc;
% load('workspace.mat')

%% Import Data
stats = readtable('pokemon.csv');
battles = readtable('combats.csv');
winnerstats = [];
statDifference_percent = [];

%% split battles into training / testing

% first step is to take all the battles and find the stat differences between the winners and losers
for i=1:size(battles,1)
    pokemon1id = battles(i,1);
    pokemon1id = table2array(pokemon1id);
    pokemon2id = battles(i,2);
    pokemon2id = table2array(pokemon2id);
    winnerid = battles(i,3);
    winnerid = table2array(winnerid);

    if winnerid == pokemon1id
        loserid = pokemon2id;
    else
        loserid = pokemon1id;
    end

    pokemon1stats = stats(pokemon1id,:);
    pokemon2stats = stats(pokemon2id,:);
%     stats(winnerid,5:12) - stats(loserid,5:12)
    winnerstats(i,:) = table2array(stats(winnerid,5:10));
    loserstats(i,:) = table2array(stats(loserid,5:10));
    statDifference(i,:) = table2array(stats(winnerid,5:10))-table2array(stats(loserid,5:10));
    numerator = abs(winnerstats(i,:)-loserstats(i,:));
    denominator = (winnerstats(i,:)+loserstats(i,:))/2;
    statDifference_percent(i,:) = 100*numerator./denominator;
end


%% Find the most important stat based on stat difference
stat_importance = [];
for j = 1:6
    stat_importance(1,j) = mean(statDifference_percent(:,j));
end

% We find that speed has the largest difference by quite a big margin
header = {'HP','Attack','Defense', 'Sp_Atk', 'Sp_Def', 'Speed'};
output = [header; num2cell(stat_importance)];

%% create a model using stats where larger difference from previous section = higher coeff
coefficients = stat_importance/sum(stat_importance);

% Append pokemon id's and winner to input data
statDifference = [statDifference table2array(battles)];

%%
% define your net to be a feedforwardnet with 10 neurons in  % the hidden layer and trainlm as training algorithm
net = feedforwardnet(10, 'trainlm');
%Mdl = fitcnet(statDifference, statDifference(:,9),"Standardize",true);

% train your model using input data statDifference and target data winner of battles
% x = statDifference(:,1:8)';
% t = statDifference(:,9)';
% net = train(net, x, t);

% estimate the target y using input data x
% y = net(x);
label = predict(Mdl,statDifference);
%%
% ynew = predict(net,x)