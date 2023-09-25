var 
A C MC M_real N Pi Pi_star Q R S W_real Y Z i_ann log_A log_N log_W_real log_Z log_y nu pi_ann r_real_ann realinterest x_aux_1 x_aux_2 ;

varexo 
eps_a eps_nu eps_z ;

parameters 
alppha betta epsilon eta phi_pi phi_y rho_a rho_nu rho_z siggma tau theta varphi ;

# Parameter definitions:
	siggma	=	1.0;
	varphi	=	5.0;
	phi_pi	=	1.5;
	phi_y	=	0.125;
	theta	=	0.75;
	rho_nu	=	0.5;
	rho_z	=	0.5;
	rho_a	=	0.9;
	betta	=	0.99;
	eta	=	3.77;
	alppha	=	0.25;
	epsilon	=	9.0;
	tau	=	0.0;

model;
	W_real(0) = C(0) ^ siggma * N(0) ^ varphi;

	Q(0) = ((betta * (C(1) / C(0)) ^ -siggma * Z(1)) / Z(0)) / Pi(1);

	R(0) = 1 / Q(0);

	Y(0) = A(0) * (N(0) / S(0)) ^ (1 - alppha);

	R(0) = Pi(1) * realinterest(0);

	R(0) = (1 / betta) * Pi(0) ^ phi_pi * (Y(0) / STEADY_STATE(Y)) ^ phi_y * exp(nu(0));

	C(0) = Y(0);

	log(A(0)) = rho_a * log(A(-1)) + eps_a;

	log(Z(0)) = rho_z * log(Z(-1)) - eps_z;

	nu(0) = rho_nu * nu(-1) + eps_nu;

	MC(0) = W_real(0) / ((S(0) * Y(0) * (1 - alppha)) / N(0));

	1 = theta * Pi(0) ^ (epsilon - 1) + (1 - theta) * Pi_star(0) ^ (1 - epsilon);

	S(0) = (1 - theta) * Pi_star(0) ^ (-epsilon / (1 - alppha)) + theta * Pi(0) ^ (epsilon / (1 - alppha)) * S(-1);

	Pi_star(0) ^ (1 + (epsilon * alppha) / (1 - alppha)) = (((epsilon * x_aux_1(0)) / x_aux_2(0)) * (1 - tau)) / (epsilon - 1);

	x_aux_1(0) = MC(0) * Y(0) * Z(0) * C(0) ^ -siggma + betta * theta * Pi(1) ^ (epsilon + (alppha * epsilon) / (1 - alppha)) * x_aux_1(1);

	x_aux_2(0) = Y(0) * Z(0) * C(0) ^ -siggma + betta * theta * Pi(1) ^ (epsilon - 1) * x_aux_2(1);

	log_y(0) = log(Y(0));

	log_W_real(0) = log(W_real(0));

	log_N(0) = log(N(0));

	pi_ann(0) = 4 * log(Pi(0));

	i_ann(0) = 4 * log(R(0));

	r_real_ann(0) = 4 * log(realinterest(0));

	M_real(0) = Y(0) / R(0) ^ eta;

	log_A(0) = log(A(0));

	log_Z(0) = log(Z(0));

end;

shocks;
var	eps_a	=	1;
var	eps_nu	=	1;
var	eps_z	=	1;
end;

initval;
	A	=	1.0;
	C	=	0.950579824954767;
	MC	=	0.8888888888935611;
	M_real	=	0.9152363832330537;
	N	=	0.9346552651848885;
	Pi	=	1.0000000000090865;
	Pi_star	=	1.0000000000272595;
	Q	=	0.9899999999923134;
	R	=	1.010101010116947;
	S	=	1.0;
	W_real	=	0.6780252644071318;
	Y	=	0.950579824954767;
	Z	=	1.0;
	i_ann	=	0.04020134347711578;
	log_A	=	0.0;
	log_N	=	-0.0675775180171486;
	log_W_real	=	-0.38857072859863234;
	log_Z	=	0.0;
	log_y	=	-0.050683138512861714;
	nu	=	0.0;
	pi_ann	=	3.634603729080339e-11;
	r_real_ann	=	0.04020134344077005;
	realinterest	=	1.0101010101077688;
	x_aux_1	=	3.451995686325885;
	x_aux_2	=	3.8834951466943237;
end;

stoch_simul(irf=40);
