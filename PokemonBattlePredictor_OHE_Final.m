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
typeDifference=[];
outcome = []; % Value for which pokemon wins, 1 for pokemon 1, 2 for pokemon 2

%% split battles into training / testing

% first step is to take all the battles and find the stat differences between the winners and losers
for i=1:size(battles,1)
    %get the 2 pokemon IDs
    pokemon1id = battles(i,1);
    pokemon1id = table2array(pokemon1id);
    pokemon2id = battles(i,2);
    pokemon2id = table2array(pokemon2id);

    %get the winner pokemon ID
    winnerid = battles(i,3);
    winnerid = table2array(winnerid);
    
    %get the loser pokemon ID
    if winnerid == pokemon1id
        loserid = pokemon2id;
        outcome(i) = 1; % If pokemon 1 wins then outcome is 1
    else
        loserid = pokemon1id;
        outcome(i) = 2; % If pokemon 2 wins then outcome is 2
    end
    
    %pokemon are all in order so get the stats of the two pokemon battling
    pokemon1stats = stats(pokemon1id,:);
    pokemon2stats = stats(pokemon2id,:);

    %get the types of the two pokemon
    pok1t1 = table2cell(stats(pokemon1id,"Type1"));%grab the first type
    pok1t2 = table2cell(stats(pokemon1id,"Type2"));%grab the second type
    
    pok2t1 = table2cell(stats(pokemon2id,"Type1"));%grab the first type
    pok2t2 = table2cell(stats(pokemon2id,"Type2"));%grab the second type
    
    types  = cell2table(cell(0,18), 'VariableNames', {'Normal', 'Fire', 'Water', 'Grass', 'Electric', 'Ice', 'Fighting', 'Poison', 'Ground', ...
        'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', 'Dark', 'Dragon', 'Steel', 'Fairy'}); %make empty table of all types
        
    pok1types = ~cellfun('isempty', regexp(types.Properties.VariableNames, pok1t1+"|"+pok1t2, 'once')) ;
    pok2types = ~cellfun('isempty', regexp(types.Properties.VariableNames, pok2t1+"|"+pok2t2, 'once')) ;
    
    typeDifference(i,:) = pok1types-pok2types;
    
    %get the stats of the winnner and the loser and store them both
    winnerstats(i,:) = table2array(stats(winnerid,5:10));
    loserstats(i,:) = table2array(stats(loserid,5:10));

    % Always find Pokemon 1 stats - Pokemon 2 stats
    statDifference(i,:) = table2array(stats(pokemon1id,5:10))-table2array(stats(pokemon2id,5:10));
end

%% Prep Data for the model
% Stat difference is 1-2 stats, so we append the outcome to be the last column
statDifference = [statDifference typeDifference outcome']; 
X = statDifference(:,1:24);
Y = statDifference(:,25);

% Split data into training and testing sets
cv = cvpartition(size(X,1),'Holdout',0.3);
Xtrain = X(cv.training,:);
Ytrain = Y(cv.training,:);
Xtest = X(cv.test,:);
Ytest = Y(cv.test,:);

%% Cross-validation (model)
cv_model = cvpartition(size(Xtrain,1),'KFold',5);
accuracy = zeros(1,5);
precision = zeros(1,5);
recall = zeros(1,5);
f_score = zeros(1,5);

net = feedforwardnet(10, 'trainlm');

for i = 1:5
    net = train(net,Xtrain(cv_model.training(i),:)',Ytrain(cv_model.training(i))'); % Train aNN model
    Ypred = net(Xtrain(cv_model.test(i),:)');
    Ypred = round(Ypred);
    confusion_matrix = confusionmat(Ytrain(find(cv_model.test(i))),Ypred); % Four cell matrix: True positive, False Positive, False negative, true negative
    accuracy(i) = (confusion_matrix(1,1) + confusion_matrix(2,2))/(confusion_matrix(1,1) + confusion_matrix(2,2) + confusion_matrix(1,2) + confusion_matrix(2,1));
    precision(i) = confusion_matrix(1,1) / (confusion_matrix(1,1) + confusion_matrix(1,2));
    recall(i) = confusion_matrix(1,1) / (confusion_matrix(1,1) + confusion_matrix(2,1));
    f_score(i) = (2*confusion_matrix(1,1)) / (2*confusion_matrix(1,1) + confusion_matrix(1,2) + confusion_matrix(2,1));
end

% Performance metrics
mean_accuracy = mean(accuracy); % checking percentage of the predictions that match test data
mean_precision = mean(precision);
mean_recall = mean(recall);
mean_f_score = mean(f_score);

%% ANN Test
% define your net to be a feedforwardnet with 10 neurons in  % the hidden layer and trainlm as training algorithm
net = feedforwardnet(10, 'trainlm');

% train your model using input data x and target data t
net = train(net, X', Y');

% estimate the target y using input data x
y = net(X');
y = round(y);

% Prediction
Ypred = net(Xtest');
Ypred = round(Ypred);

accuracy_ANN = sum(Ypred' == Ytest)/size(Ytest,1);

% Four cell matrix: True positive, False Positive, False negative, true negative
confusion_matrix = confusionmat(Ytest,Ypred);

%% Function call for live demo to test between two pokemon and say whether it was correct or not
pokemon1 = 208;
pokemon2 = 6;

[winner] = battleSim(battles,stats,pokemon1,pokemon2,net);
fprintf(char(stats{pokemon1,2}) + " vs. " + char(stats{pokemon2,2}) + ": Winner is " + char(stats{winner,2}) + "! \n");

function [prediction_rescale] = battleSim(battles,stats,ID1,ID2,model)
%BATTLESIM Live demo of a battle, pass in two pokemon ID's and the model then return the output
%input: the battles, the stats, the ID of the two pokemon, and the model being tested on
    pokemon1id = ID1;
    pokemon2id = ID2;

    pokemon1stats = stats(pokemon1id,:);
    pokemon2stats = stats(pokemon2id,:);

    BattleStatDifference = table2array(stats(pokemon1id,5:10))-table2array(stats(pokemon2id,5:10));
    
    %get the types of the two pokemon passed in
    pok1t1 = table2cell(stats(pokemon1id,"Type1"));%grab the first type
    pok1t2 = table2cell(stats(pokemon1id,"Type2"));%grab the second type
    
    pok2t1 = table2cell(stats(pokemon2id,"Type1"));%grab the first type
    pok2t2 = table2cell(stats(pokemon2id,"Type2"));%grab the second type
    
    types  = cell2table(cell(0,18), 'VariableNames', {'Normal', 'Fire', 'Water', 'Grass', 'Electric', 'Ice', 'Fighting', 'Poison', 'Ground', ...
        'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', 'Dark', 'Dragon', 'Steel', 'Fairy'}); %make empty table of all types
        
    pok1types = ~cellfun('isempty', regexp(types.Properties.VariableNames, pok1t1+"|"+pok1t2, 'once')) ;
    pok2types = ~cellfun('isempty', regexp(types.Properties.VariableNames, pok2t1+"|"+pok2t2, 'once')) ;
    
    typeDifference = pok1types-pok2types; %calculate the difference

    input = [ BattleStatDifference typeDifference]; %append the stat differences and the type differences as the input to the model
    % Prediction
    prediction = model(input');
    prediction = round(prediction);
    
    % Change prediction back from 1/2 to the actual winner ID
    if prediction == 1
        prediction_rescale = pokemon1id;
    else
        prediction_rescale = pokemon2id;
    end   
end

