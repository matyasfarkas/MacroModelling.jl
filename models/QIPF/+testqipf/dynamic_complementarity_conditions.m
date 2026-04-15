function [lb, ub] = dynamic_complementarity_conditions(params)
ub = inf(121,1);
lb = -ub;
lb(5)=0;
end
