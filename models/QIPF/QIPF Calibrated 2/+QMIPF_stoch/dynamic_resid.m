function residual = dynamic_resid(T, y, x, params, steady_state, it_, T_flag)
% function residual = dynamic_resid(T, y, x, params, steady_state, it_, T_flag)
%
% File created by Dynare Preprocessor from .mod file
%
% Inputs:
%   T             [#temp variables by 1]     double   vector of temporary terms to be filled by function
%   y             [#dynamic variables by 1]  double   vector of endogenous variables in the order stored
%                                                     in M_.lead_lag_incidence; see the Manual
%   x             [nperiods by M_.exo_nbr]   double   matrix of exogenous variables (in declaration order)
%                                                     for all simulation periods
%   steady_state  [M_.endo_nbr by 1]         double   vector of steady state values
%   params        [M_.param_nbr by 1]        double   vector of parameter values in declaration order
%   it_           scalar                     double   time period for exogenous variables for which
%                                                     to evaluate the model
%   T_flag        boolean                    boolean  flag saying whether or not to calculate temporary terms
%
% Output:
%   residual
%

if T_flag
    T = QMIPF_stoch.dynamic_resid_tt(T, y, x, params, steady_state, it_);
end
residual = zeros(192, 1);
    residual(1) = (y(138)) - (T(2)/(1+y(187)));
    residual(2) = (y(138)) - (T(3)*y(278));
    residual(3) = (y(217)^(1+T(4)*params(7))) - ((1+params(85))*y(246)/y(248)*params(99)/params(101));
    residual(4) = (y(246)) - (T(6)*T(7)/params(99)+T(10)*y(309));
    residual(5) = (y(248)) - (y(152)*y(138)*y(203)*(1-y(190))*y(201)*T(11)/params(101)+T(13)*y(311));
    residual(6) = (y(211)^T(12)) - ((1-params(97))*y(217)^T(12)+params(97)*T(14)^T(12));
    residual(7) = (y(182)) - (T(16)*T(180));
    residual(8) = (y(209)) - ((1-params(97))*(y(217)/y(211))^T(9)+params(97)*y(51)*T(19));
    residual(9) = (y(97)) - (y(93)+params(12)*y(108));
    residual(10) = (y(150)) - (y(211)*T(20)*y(110)*T(21)*T(23));
    residual(11) = (y(234)) - (y(236)*y(162)-y(238)*T(25));
    residual(12) = (y(234)) - (T(28)/y(110)+T(31)*y(303));
    residual(13) = (y(236)) - (T(26)*y(222)*y(138)*y(203)*y(197)/y(110)+T(33)*y(305));
    residual(14) = (y(238)) - (y(222)*y(197)*y(138)*y(203)*params(84)*params(36)/(1+params(36)+params(84)*params(36))/y(110)+params(3)*params(95)*y(290)/y(286)*y(307));
    residual(15) = (y(178)) - (T(34)*y(30)^params(15));
    residual(16) = (y(222)*y(158)+y(161)*y(228)) - (T(35)*(y(152)*y(233))^(1-params(1)));
    residual(17) = (y(158)) - (T(26)/(1+params(36))*T(36)+params(36)/(1+params(36)));
    residual(18) = (y(240)^T(37)) - ((1-params(95))*y(162)^T(37)+params(95)*(T(38)*y(58))^T(37));
    residual(19) = (y(161)) - (T(39)/(1+params(39))*T(40)+params(39)/(1+params(39)));
    residual(20) = (y(256)^T(41)) - ((1-params(93))*y(165)^T(41)+params(93)*(T(42)*y(64))^T(41));
    residual(21) = (y(205)) - (1+params(36)-params(36)*y(242));
    residual(22) = (y(242)) - (y(162)*(1-params(95))+params(95)*y(178)/y(172)*y(60));
    residual(23) = (y(205)) - (y(244));
    residual(24) = (y(244)^T(32)) - ((1-params(95))*y(162)^T(32)+params(95)*(T(38)*y(62))^T(32));
    residual(25) = (y(222)) - (y(93)*T(44)+y(108)*T(46));
    residual(26) = (y(250)) - (y(165)*y(252)-y(254)*T(47));
    residual(27) = (y(250)) - (T(49)/y(110)+T(51)*y(313));
    residual(28) = (y(252)) - (T(39)*y(228)*y(138)*y(203)*y(199)/y(116)*y(184)+T(53)*y(315));
    residual(29) = (y(254)) - (y(184)*T(54)+params(3)*params(93)*y(293)/y(289)*y(317));
    residual(30) = (y(181)) - (T(55)*y(33)^params(17));
    residual(31) = (y(206)) - (1+params(39)-params(39)*y(258));
    residual(32) = (y(258)) - ((1-params(93))*y(165)+params(93)*y(181)/y(177)*y(66));
    residual(33) = (y(206)) - (y(260));
    residual(34) = (y(260)^T(52)) - ((1-params(93))*y(165)^T(52)+params(93)*(T(42)*y(68))^T(52));
    residual(35) = (y(228)) - (T(56)*(T(59)+T(62)));
    residual(36) = (y(110)) - (T(64)^(-params(49)));
    residual(37) = (y(118)) - (T(66)^(-params(53)));
    residual(38) = (y(116)) - (T(68)^(-params(50)));
    residual(39) = (y(124)) - (T(70)^(-params(54)));
    residual(40) = (y(166)) - (y(172)*y(110)/y(15));
    residual(41) = (y(126)/y(19)) - (y(176)/y(172));
    residual(42) = (y(142)) - (T(72));
    residual(43) = (y(146)) - (T(74));
    residual(44) = (y(131)) - (max(params(10),T(75)));
    residual(45) = (y(140)) - (T(76)/(1+y(188)));
    residual(46) = (y(140)) - (T(77)*y(280));
    residual(47) = (y(218)^(1+T(4)*params(7))) - ((1+params(85))*y(247)/y(249)*params(100)/params(102));
    residual(48) = (y(247)) - (T(79)*T(80)/params(100)+T(82)*y(310));
    residual(49) = (y(249)) - (y(154)*y(140)*y(204)*(1-y(191))*y(202)*T(83)/params(102)+T(84)*y(312));
    residual(50) = (y(213)^T(12)) - ((1-params(98))*y(218)^T(12)+params(98)*T(85)^T(12));
    residual(51) = (y(183)) - (T(87)*T(88));
    residual(52) = (y(210)) - ((1-params(98))*(y(218)/y(213))^T(9)+params(98)*y(52)*T(90));
    residual(53) = (y(99)) - (y(95)+params(12)*y(109));
    residual(54) = (y(151)) - (T(20)*y(213)*y(112)*T(91)*T(93));
    residual(55) = (y(235)) - (y(237)*y(163)-y(239)*T(94));
    residual(56) = (y(235)) - (T(97)/y(112)+T(99)*y(304));
    residual(57) = (y(237)) - (T(95)*y(224)*y(140)*y(204)*y(200)/y(112)+T(101)*y(306));
    residual(58) = (y(239)) - (y(224)*y(200)*y(140)*y(204)*params(84)*params(45)/(1+params(45)+params(84)*params(45))/y(112)+params(4)*params(96)*y(291)/y(287)*y(308));
    residual(59) = (y(179)) - (T(102)*y(31)^params(19));
    residual(60) = (y(224)*y(159)+y(160)*y(226)) - (T(103)*(y(154)*y(262))^(1-params(1)));
    residual(61) = (y(159)) - (T(95)/(1+params(45))*T(104)+params(45)/(1+params(45)));
    residual(62) = (y(241)^T(105)) - ((1-params(96))*y(163)^T(105)+params(96)*(T(106)*y(59))^T(105));
    residual(63) = (y(160)) - (T(107)/(1+params(40))*T(108)+params(40)/(1+params(40)));
    residual(64) = (y(257)^T(109)) - ((1-params(94))*y(164)^T(109)+params(94)*(T(110)*y(65))^T(109));
    residual(65) = (y(208)) - (1+params(45)-params(45)*y(243));
    residual(66) = (y(243)) - (y(163)*(1-params(96))+params(96)*y(179)/y(174)*y(61));
    residual(67) = (y(208)) - (y(245));
    residual(68) = (y(245)^T(100)) - ((1-params(96))*y(163)^T(100)+params(96)*(T(106)*y(63))^T(100));
    residual(69) = (y(224)) - (y(95)*T(111)+y(109)*T(112));
    residual(70) = (y(251)) - (y(164)*y(253)-y(255)*T(113));
    residual(71) = (y(251)) - (T(115)/y(112)+T(117)*y(314));
    residual(72) = (y(253)) - (T(107)*T(118)+T(120)*y(316));
    residual(73) = (y(255)) - (T(121)/y(184)+params(4)*params(94)*y(292)/y(288)*y(318));
    residual(74) = (y(180)) - (T(122)*y(32)^params(18));
    residual(75) = (y(207)) - (1+params(40)-params(40)*y(259));
    residual(76) = (y(259)) - ((1-params(94))*y(164)+params(94)*y(180)/y(176)*y(67));
    residual(77) = (y(207)) - (y(261));
    residual(78) = (y(261)^T(119)) - ((1-params(94))*y(164)^T(119)+params(94)*(T(110)*y(69))^T(119));
    residual(79) = (y(226)) - (T(123)*(T(72)+T(74)));
    residual(80) = (y(112)) - (T(125)^(-params(50)));
    residual(81) = (y(120)) - (T(127)^(-params(54)));
    residual(82) = (y(114)) - (T(129)^(-params(49)));
    residual(83) = (y(122)) - (T(131)^(-params(53)));
    residual(84) = (y(168)) - (y(174)*y(112)/y(17));
    residual(85) = (y(128)/y(20)) - (y(177)/y(174));
    residual(86) = (y(144)) - (T(59));
    residual(87) = (y(148)) - (T(62));
    residual(88) = (y(135)) - (max(params(11),T(132)));
    residual(89) = (y(131)*(1-y(189))) - (y(282)*y(135)*y(296)/y(184)/y(284)+y(131)*y(129)*y(87)/(params(194)+params(197)));
    residual(90) = (y(87)) - ((-y(86))-y(90)+y(89));
    residual(91) = (y(86)) - (y(222)+T(135)*y(1)+y(1)*(1-params(29))*(y(25)-y(21))/y(172)+T(138)*((params(35)-params(32))*y(5)-(1-params(32))*y(4))+T(139)*((1-params(32))*y(2)+y(5)*(1-params(35)))+y(228)*T(140)-y(93)*y(110)-y(108)*y(118));
    residual(92) = (y(129)) - (T(141));
    residual(93) = (y(137)) - (y(131)+y(193));
residual(94) = y(193);
    residual(95) = (y(92)) - (y(86)+params(26)*y(302));
    residual(96) = (y(219)) - (y(222)+y(228));
    residual(97) = (y(231)) - (y(224)+y(226));
    residual(98) = (y(194)) - (y(203)*T(142)+params(3)*y(298));
    residual(99) = (y(195)) - (y(204)*T(143)+params(4)*y(299));
    residual(100) = (y(139)) - (T(144)/(1+y(187)));
    residual(101) = (y(139)) - (T(145)*y(279));
    residual(102) = ((1-y(190))*y(212)) - (T(146)/y(139));
    residual(103) = (y(98)) - (params(12)*y(108)+y(94));
    residual(104) = ((1+params(82))/(1+params(84))) - (y(212)*T(23)/(1-params(1))*y(111)*T(147));
    residual(105) = (y(223)+y(229)) - (T(35)*(y(233)*y(153))^(1-params(1)));
    residual(106) = (y(223)) - (y(94)*T(148)+y(108)*T(149));
    residual(107) = (y(229)) - (T(56)*(T(151)+T(153)));
    residual(108) = (y(111)) - (T(154)^(-params(49)));
    residual(109) = (y(119)) - (T(155)^(-params(53)));
    residual(110) = (y(117)) - (T(156)^(-params(50)));
    residual(111) = (y(125)) - (T(157)^(-params(54)));
    residual(112) = (y(167)) - (y(111)/y(16)*y(173));
    residual(113) = (y(143)) - (T(159));
    residual(114) = (y(147)) - (T(161));
    residual(115) = (y(141)) - (T(162)/(1+y(188)));
    residual(116) = (y(141)) - (T(163)*y(281));
    residual(117) = ((1-y(191))*y(214)) - (T(164)/y(141));
    residual(118) = (y(100)) - (params(12)*y(109)+y(96));
    residual(119) = ((1+params(82))/(1+params(84))) - (y(214)*T(93)/(1-params(1))*y(113)*T(165));
    residual(120) = (y(225)+y(227)) - (T(103)*(y(262)*y(155))^(1-params(1)));
    residual(121) = (y(225)) - (y(96)*T(166)+y(109)*T(167));
    residual(122) = (y(227)) - (T(123)*(T(159)+T(161)));
    residual(123) = (y(113)) - (T(168)^(-params(50)));
    residual(124) = (y(121)) - (T(169)^(-params(54)));
    residual(125) = (y(115)) - (T(170)^(-params(49)));
    residual(126) = (y(123)) - (T(171)^(-params(53)));
    residual(127) = (y(169)) - (y(113)/y(18)*y(175));
    residual(128) = (y(145)) - (T(151));
    residual(129) = (y(149)) - (T(153));
    residual(130) = (y(134)*(1-params(169))) - (y(283)*y(136)*y(297)/y(186)/y(285)+y(134)*y(130)*y(88)/(params(194)+params(197)));
    residual(131) = (y(88)) - ((-y(91))-params(110)+params(109));
    residual(132) = (y(91)) - (y(223)+T(174)*y(6)+(y(22)/y(173)-T(176)/y(37))*(params(110)*(params(32)-params(35))-(1-params(32))*params(109))+y(22)*(1-params(6))*params(169)/y(173)*((1-params(32))*y(3)+(1-params(35))*params(110))+y(229)*y(186)*y(111)/y(117)-y(94)*y(111)-y(108)*y(119));
    residual(133) = (y(130)) - (T(141));
    residual(134) = (y(127)) - (y(111)*y(186)/y(113));
    residual(135) = (y(134)) - ((1-params(37))*(params(131)+params(41)*(y(167)-params(154))+params(43)*(y(173)-params(156)))+params(37)*y(22));
    residual(136) = (y(136)) - ((1-params(38))*(params(132)+params(42)*(y(169)-params(155))+params(44)*(y(175)-params(157)))+params(38)*y(24));
    residual(137) = (y(230)) - (y(223)+y(229));
    residual(138) = (y(232)) - (y(225)+y(227));
    residual(139) = (y(203)-params(181)) - (params(68)*(y(49)-params(181))+x(it_, 20)+params(80)*x(it_, 21));
    residual(140) = (y(156)-params(144)) - (params(55)*(y(26)-params(144))+x(it_, 7));
    residual(141) = (y(187)-params(167)) - (params(57)*(y(38)-params(167))+x(it_, 9));
    residual(142) = (y(190)-params(170)) - (params(60)*(y(41)-params(170))+x(it_, 12));
    residual(143) = ((y(108)-params(118))/params(118)) - (params(89)*(y(13)-params(118))/params(118)+1/params(72)*x(it_, 3));
    residual(144) = ((y(233)-params(199))/params(199)) - (params(70)*(y(57)-params(199))/params(199)+x(it_, 22)+params(81)*x(it_, 23));
    residual(145) = (log((1+params(82))/y(197))) - (params(62)*log((1+params(82))/y(43))+x(it_, 14));
    residual(146) = (log((1+params(82))/y(198))) - (params(63)*log((1+params(82))/y(44))+x(it_, 15));
    residual(147) = (log((1+params(83))/y(201))) - (params(66)*log((1+params(83))/y(47))+x(it_, 18)+params(79)*x(it_, 19));
    residual(148) = (y(106)) - (params(51)*y(11)+x(it_, 5)+params(78)*x(it_, 6));
    residual(149) = (y(204)-params(182)) - (x(it_, 21)+params(69)*(y(50)-params(182)));
    residual(150) = (y(157)-params(145)) - (params(56)*(y(27)-params(145))+x(it_, 8));
    residual(151) = (y(188)-params(168)) - (params(58)*(y(39)-params(168))+x(it_, 10));
    residual(152) = (y(191)-params(171)) - (params(61)*(y(42)-params(171))+x(it_, 13));
    residual(153) = ((y(109)-params(119))/params(119)) - (params(90)*(y(14)-params(119))/params(119)+1/params(73)*x(it_, 4));
    residual(154) = ((y(262)-params(228))/params(228)) - (x(it_, 23)+params(71)*(y(70)-params(228))/params(228));
    residual(155) = (log((1+params(82))/y(200))) - (params(65)*log((1+params(82))/y(46))+x(it_, 17));
    residual(156) = (log((1+params(82))/y(199))) - (params(64)*log((1+params(82))/y(45))+x(it_, 16));
    residual(157) = (log((1+params(83))/y(202))) - (x(it_, 19)+params(67)*log((1+params(83))/y(48)));
    residual(158) = (y(107)) - (x(it_, 6)+params(52)*y(12));
    residual(159) = ((y(90)-params(110))/params(193)) - (params(92)*(y(5)-params(110))/params(193)+x(it_, 2));
    residual(160) = ((y(89)-params(109))/params(193)) - ((y(90)-params(110))*params(229)/params(193)-y(193)*params(230)+x(it_, 1));
    residual(161) = (y(189)-params(169)) - (max(0,(-params(231))*(y(86)-params(107))/params(193))+x(it_, 11));
    residual(162) = (y(101)) - (100*log(y(184)/y(73)));
    residual(163) = (y(102)) - (100*log(y(211)/y(76)));
    residual(164) = (y(103)) - (100*log(y(213)/y(79)));
    residual(165) = (y(104)) - (100*log(y(219)/y(82)));
    residual(166) = (y(105)) - (100*log(y(231)/y(85)));
    residual(167) = (y(132)) - (400*(y(131)-params(131)));
    residual(168) = (y(133)) - (400*(y(135)-params(132)));
    residual(169) = (y(170)) - ((y(172)-params(156))*400);
    residual(170) = (y(171)) - ((y(168)-params(155))*400);
    residual(171) = (y(185)) - (100*log(y(184)/params(166)));
    residual(172) = (y(192)) - (100*(T(177)-y(93)*y(110)-y(108)*y(118))/T(177)-100+100*(params(118)+params(112))/params(193));
    residual(173) = (y(196)) - (100*(y(131)*y(129)*y(87)-params(131)*params(130)*params(108))/(params(194)+params(197)));
    residual(174) = (y(215)) - (100*log(y(211)/params(189)));
    residual(175) = (y(216)) - (100*log(y(213)/params(190)));
    residual(176) = (y(220)) - (100*log(y(219)/params(193)));
    residual(177) = (y(221)) - (100*log(y(231)/params(198)));
    residual(178) = (y(263)) - (y(36));
    residual(179) = (y(264)) - (y(71));
    residual(180) = (y(265)) - (y(72));
    residual(181) = (y(266)) - (y(53));
    residual(182) = (y(267)) - (y(74));
    residual(183) = (y(268)) - (y(75));
    residual(184) = (y(269)) - (y(54));
    residual(185) = (y(270)) - (y(77));
    residual(186) = (y(271)) - (y(78));
    residual(187) = (y(272)) - (y(55));
    residual(188) = (y(273)) - (y(80));
    residual(189) = (y(274)) - (y(81));
    residual(190) = (y(275)) - (y(56));
    residual(191) = (y(276)) - (y(83));
    residual(192) = (y(277)) - (y(84));

end
