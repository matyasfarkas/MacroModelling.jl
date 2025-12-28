var 
A C MC M_real N Pi Pi_star Q R S W_real Y Z i_ann log_N log_W_real log_y nu pi_ann r_real_ann realinterest x_aux_1 x_aux_2 ;

varexo 
eps_a eps_nu eps_z ;

parameters 
std_a std_nu std_z alpha beta eta theta rho__a rho__z rho__nu sigma tau varphi phi__y phi__p__i epsilon ;

% Parameter definitions:
	sigma	=	1.0;
	varphi	=	5.0;
	phi__p__i	=	1.5;
	phi__y	=	0.125;
	theta	=	0.75;
	rho__nu	=	0.5;
	rho__z	=	0.5;
	rho__a	=	0.9;
	beta	=	0.99;
	eta	=	3.77;
	alpha	=	0.25;
	epsilon	=	9.0;
	tau	=	0.0;
	std_a	=	0.01;
	std_z	=	0.05;
	std_nu	=	0.0025;

model;
	W_real(0) = C(0) ^ sigma * N(0) ^ varphi;

	Q(0) = ((beta * (C(1) / C(0)) ^ -sigma * Z(1)) / Z(0)) / Pi(1);

	R(0) = 1 / Q(0);

	Y(0) = A(0) * (N(0) / S(0)) ^ (1 - alpha);

	R(0) = Pi(1) * realinterest(0);

	R(0) = (1 / beta) * Pi(0) ^ phi__p__i * (Y(0) / STEADY_STATE(Y)) ^ phi__y * exp(nu(0));

	C(0) = Y(0);

	log(A(0)) = rho__a * log(A(-1)) + std_a * eps_a;

	log(Z(0)) = rho__z * log(Z(-1)) - std_z * eps_z;

	nu(0) = rho__nu * nu(-1) + std_nu * eps_nu;

	MC(0) = W_real(0) / ((S(0) * Y(0) * (1 - alpha)) / N(0));

	1 = theta * Pi(0) ^ (epsilon - 1) + (1 - theta) * Pi_star(0) ^ (1 - epsilon);

	S(0) = (1 - theta) * Pi_star(0) ^ (-epsilon / (1 - alpha)) + theta * Pi(0) ^ (epsilon / (1 - alpha)) * S(-1);

	Pi_star(0) ^ (1 + (epsilon * alpha) / (1 - alpha)) = (((epsilon * x_aux_1(0)) / x_aux_2(0)) * (1 - tau)) / (epsilon - 1);

	x_aux_1(0) = MC(0) * Y(0) * Z(0) * C(0) ^ -sigma + beta * theta * Pi(1) ^ (epsilon + (alpha * epsilon) / (1 - alpha)) * x_aux_1(1);

	x_aux_2(0) = Y(0) * Z(0) * C(0) ^ -sigma + beta * theta * Pi(1) ^ (epsilon - 1) * x_aux_2(1);

	log_y(0) = log(Y(0));

	log_W_real(0) = log(W_real(0));

	log_N(0) = log(N(0));

	pi_ann(0) = 4 * log(Pi(0));

	i_ann(0) = 4 * log(R(0));

	r_real_ann(0) = 4 * log(realinterest(0));

	M_real(0) = Y(0) / R(0) ^ eta;

end;

shocks;
var	eps_a	=	1;
var	eps_nu	=	1;
var	eps_z	=	1;
end;

initval;
	A	=	1.0;
	C	=	0.9505798249541406;
	MC	=	0.8888888888888891;
	M_real	=	0.9152363832868937;
	N	=	0.9346552651840673;
	Pi	=	0.9999999999999992;
	Pi_star	=	0.9999999999999977;
	Q	=	0.9900000000000011;
	R	=	1.010101010101009;
	S	=	1.0;
	W_real	=	0.6780252644037247;
	Y	=	0.9505798249541407;
	Z	=	1.0;
	i_ann	=	0.040201343414001625;
	log_N	=	-0.06757751801802725;
	log_W_real	=	-0.3885707286036575;
	log_y	=	-0.05068313851352055;
	nu	=	0.0;
	pi_ann	=	-3.1086244689504395e-15;
	r_real_ann	=	0.04020134341400514;
	realinterest	=	1.01010101010101;
	x_aux_1	=	3.4519956850053006;
	x_aux_2	=	3.883495145630999;
end;

stoch_simul(order = 1, irf = 40);

% Extended Path (SEP) tests for validation against MacroModelling.jl
% Configure extended_path options for better convergence
options_.ep.stochastic.order = 0;
options_.ep.verbosity = 1;
options_.ep.maxit = 500;
options_.ep.tolerance.f = 1e-5;
options_.ep.tolerance.x = 1e-5;
% Use trust region solver for better convergence
options_.solve_algo = 4;

% Test 1: SEP with order=1 (periods=10, branching length = 1)
disp('================================================================');
disp('DYNARE EXTENDED PATH TEST 1: periods=10, order=1');
disp('================================================================');
extended_path(periods=10, order=1);

% Report key steady state values for comparison
disp(' ');
disp('Key steady state values (SEP order=1):');
Y_idx = strmatch('Y', M_.endo_names, 'exact');
C_idx = strmatch('C', M_.endo_names, 'exact');
Pi_idx = strmatch('Pi', M_.endo_names, 'exact');
R_idx = strmatch('R', M_.endo_names, 'exact');
N_idx = strmatch('N', M_.endo_names, 'exact');

if size(oo_.endo_simul, 2) >= M_.maximum_lag+1
    disp(['  Y  = ' num2str(oo_.endo_simul(Y_idx, M_.maximum_lag+1), '%12.8f')]);
    disp(['  C  = ' num2str(oo_.endo_simul(C_idx, M_.maximum_lag+1), '%12.8f')]);
    disp(['  Pi = ' num2str(oo_.endo_simul(Pi_idx, M_.maximum_lag+1), '%12.8f')]);
    disp(['  R  = ' num2str(oo_.endo_simul(R_idx, M_.maximum_lag+1), '%12.8f')]);
    disp(['  N  = ' num2str(oo_.endo_simul(N_idx, M_.maximum_lag+1), '%12.8f')]);
else
    disp('  Warning: Simulation did not complete successfully');
end

% Test 2: SEP with order=2 (periods=10, branching length = 2)
disp(' ');
disp('================================================================');
disp('DYNARE EXTENDED PATH TEST 2: periods=10, order=2');
disp('================================================================');
extended_path(periods=10, order=2);

% Report key steady state values for comparison
disp(' ');
disp('Key steady state values (SEP order=2):');
if size(oo_.endo_simul, 2) >= M_.maximum_lag+1
    disp(['  Y  = ' num2str(oo_.endo_simul(Y_idx, M_.maximum_lag+1), '%12.8f')]);
    disp(['  C  = ' num2str(oo_.endo_simul(C_idx, M_.maximum_lag+1), '%12.8f')]);
    disp(['  Pi = ' num2str(oo_.endo_simul(Pi_idx, M_.maximum_lag+1), '%12.8f')]);
    disp(['  R  = ' num2str(oo_.endo_simul(R_idx, M_.maximum_lag+1), '%12.8f')]);
    disp(['  N  = ' num2str(oo_.endo_simul(N_idx, M_.maximum_lag+1), '%12.8f')]);
else
    disp('  Warning: Simulation did not complete successfully');
end

disp(' ');
disp('================================================================');
disp('EXTENDED PATH TESTS COMPLETE');
disp('================================================================');
