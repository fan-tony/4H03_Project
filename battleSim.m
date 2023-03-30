function [prediction_rescale] = battleSim(battles,stats,ID1,ID2,model)
%BATTLESIM Live demo of a battle, pass in two pokemon ID's and the model then return the output
    pokemon1id = ID1;
    pokemon2id = ID2;

    pokemon1stats = stats(pokemon1id,:);
    pokemon2stats = stats(pokemon2id,:);

    BattleStatDifference = table2array(stats(pokemon1id,5:10))-table2array(stats(pokemon2id,5:10));

    % Prediction
    prediction = predict(model,BattleStatDifference);
    
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

