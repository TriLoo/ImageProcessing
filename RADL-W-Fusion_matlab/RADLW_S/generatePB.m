function [ PB ] = generatePB( PB_A, PB_B )
% ------------------------------
% Author : smh
% Date   : 2017.12.05
% Description:
%   This file including the generation of Prediction Block.
%   
% ------------------------------

lenA = length(PB_A);
lenB = length(PB_B);

if lenA >= lenB
    PB = cell(1, lenA);
    for i = 1 : lenB
        sdA = squareDifference(PB_A{i});
        sdB = squareDifference(PB_B{i});
        if sdA >= sdB
            PB{i} = PB_A{i};
        else
            PB{i} = PB_B{i};
        end
    end
    for i = lenB+1 : lenA
        PB{i} = PB_A{i};
    end
else
    PB = cell(1, lenB);
    for i = 1 : lenA
        sdA = squareDifference(PB_A{i});
        sdB = squareDifference(PB_B{i});
        if sdA >= sdB
            PB{i} = PB_A{i};
        else
            PB{i} = PB_B{i};
        end
    end
    for i = lenA+1 : lenB
        PB{i} = PB_B{i};
    end
end


end

