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
%     stats(winnerid,5:12) - stats(loserid,5:12)
    winnerstats(i,:) = table2array(stats(winnerid,5:10));
    loserstats(i,:) = table2array(stats(loserid,5:10));

    % Always find Pokemon 1 stats - Pokemon 2 stats
    statDifference(i,:) = table2array(stats(pokemon1id,5:10))-table2array(stats(pokemon2id,5:10));
%     numerator = abs(winnerstats(i,:)-loserstats(i,:));
%     denominator = (winnerstats(i,:)+loserstats(i,:))/2;
%     statDifference_percent(i,:) = 100*numerator./denominator;
end


%% Find the most important stat based on stat difference
% stat_importance = [];
% for j = 1:6
%     stat_importance(1,j) = mean(statDifference_percent(:,j));
% end

% We find that speed has the largest difference by quite a big margin
% header = {'HP','Attack','Defense', 'Sp_Atk', 'Sp_Def', 'Speed'};
% output = [header; num2cell(stat_importance)];

%% Prep Data for the model
% coefficients = stat_importance/sum(stat_importance);

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

% Use cross-validation to select optimal number of neighbors
% outer loop of the cross-validation splits the data into num_folds folds and iterates over each fold as the validation set, while using the remaining folds as the training set
% inner loop iterates over different values of NumNeighbors and trains a kNN model with that number of neighbors on the training set
% Resulting accuracy is then stored in cv_accuracy matrix and we calculate
% the mean for reach model
num_folds = 5; % google recomments 3 to 5 folds for datasets over 1k in size 
num_neighbors = 1:10;
cv_accuracy = zeros(length(num_neighbors),num_folds);
for i = 1:num_folds
    cv = cvpartition(size(Xtrain,1),'KFold',num_folds);
    for j = 1:length(num_neighbors)
        mdl = fitcknn(Xtrain(cv.training(i),:),Ytrain(cv.training(i)),'NumNeighbors',num_neighbors(j));
        Ypred = predict(mdl,Xtrain(cv.test(i),:));
        cv_accuracy(j,i) = sum(Ypred == Ytrain(cv.test(i)))/numel(Ypred);
    end
end
mean_cv_accuracy = mean(cv_accuracy,2);
[best_accuracy,idx] = max(mean_cv_accuracy);
best_num_neighbors = num_neighbors(idx);

%% Create the Model

% Train kNN model with best number of neighbors
mdl = fitcknn(Xtrain,Ytrain,'NumNeighbors',best_num_neighbors);

% Prediction
Ypred = predict(mdl,Xtest);

% Evaluate model performance by checking percentage of the predictions that match test data
accuracy = sum(Ypred == Ytest)/size(Ytest,1);

% Four cell matrix: True positive, False Positive, False negative, true negative
confusion_matrix = confusionmat(Ytest,Ypred);

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
[winner] = battleSim(battles,stats,229,22,mdl)

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
    prediction = predict(model,input);
    
%     for i=1:size(battles,1)
%         if (table2array(battles(i,1)) == ID1 | table2array(battles(i,1)) == ID2) && (table2array(battles(i,2)) == ID2 | table2array(battles(i,2)) == ID1)
%            real_outcome = table2array(battles(i,3));
%            break;
%         end
%     end
    
    % Change prediction back from 1/2 to the actual winner ID
    if prediction == 1
        prediction_rescale = pokemon1id;
    else
        prediction_rescale = pokemon2id;
    end

%     if prediction_rescale == real_outcome
%         correct = true;
%     else
%         correct = false;
%     end
%     
end


