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
outcome = []; % Value for which pokemon wins, 1 for pokemon 1, 2 for pokemon 2

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
        outcome(i) = 1; % If pokemon 1 wins then outcome is 1
    else
        loserid = pokemon1id;
        outcome(i) = 2; % If pokemon 2 wins then outcome is 2
    end

    pokemon1stats = stats(pokemon1id,:);
    pokemon2stats = stats(pokemon2id,:);
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
statDifference = [statDifference outcome']; 
X = statDifference(:,1:6);
Y = statDifference(:,7);

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

%% Function call for live demo to test between two pokemon and say whether it was correct or not
[winner] = battleSim(battles,stats,320,6,mdl) %Wailmer and Charizard
