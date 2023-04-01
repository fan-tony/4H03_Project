function [prediction_rescale] = battleSim(battles,stats,ID1,ID2,model)
%BATTLESIM Live demo of a battle, pass in two pokemon ID's and the model then return the output
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

    input = [BattleStatDifference typeDifference]; %append the stat differences and the type differences as the input to the model

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

