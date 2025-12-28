function [T_order, T] = static_g1_tt(y, x, params, T_order, T)
if T_order >= 1
    return
end
[T_order, T] = Smets_Wouters_2007_HLT.static_resid_tt(y, x, params, T_order, T);
T_order = 1;
if size(T, 1) < 73
    T = [T; NaN(73 - size(T, 1), 1)];
end
T(56) = getPowerDeriv(T(3),T(4),1);
T(57) = getPowerDeriv(T(3),T(15),1);
T(58) = 1-params(12)/params(10);
T(59) = getPowerDeriv(y(16),T(36),1);
T(60) = getPowerDeriv(T(9),T(10),1);
T(61) = getPowerDeriv(T(9),T(17),1);
T(62) = getPowerDeriv(y(17),T(32),1);
T(63) = getPowerDeriv(T(20),1+params(34),1);
T(64) = getPowerDeriv(T(20),params(34),1);
T(65) = getPowerDeriv(y(42),params(13),1);
T(66) = (y(42)*T(6)*T(65)-T(5)*T(6))/(y(42)*y(42));
T(67) = T(66)*getPowerDeriv(T(7),T(4),1);
T(68) = getPowerDeriv(y(42),params(14),1);
T(69) = (y(42)*T(12)*T(68)-T(11)*T(12))/(y(42)*y(42));
T(70) = T(69)*getPowerDeriv(T(13),T(10),1);
T(71) = T(66)*getPowerDeriv(T(7),T(15),1);
T(72) = T(69)*getPowerDeriv(T(13),T(17),1);
T(73) = getPowerDeriv(T(43),(1-params(31))*params(32),1);
end
