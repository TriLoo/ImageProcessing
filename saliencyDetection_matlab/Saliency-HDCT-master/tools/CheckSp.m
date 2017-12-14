function [new_Sp N_Sp] = CheckSp(Sp)

new_Sp = double(Sp);
for i = 0 : 1 : max(max(new_Sp))
    if sum(sum(new_Sp==i)) == 0
        while sum(sum(new_Sp==i)) == 0 & sum(sum(new_Sp > i)) ~= 0 
            new_Sp = new_Sp - double(new_Sp > i);
        end
    end
end
new_Sp = new_Sp + 1;
N_Sp = max(max(new_Sp));

end