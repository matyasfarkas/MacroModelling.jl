function [lb, ub] = dynamic_complementarity_conditions(params)
ub = inf(192,1);
lb = -ub;
lb(7)=0;
end
