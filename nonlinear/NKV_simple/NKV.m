%
% Status : main Dynare file 
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

clear all
tic;
global M_ oo_ options_ ys0_ ex0_ estimation_info
options_ = [];
M_.fname = 'NKV';
%
% Some global variables initialization
%
global_initialization;
diary off;
diary('NKV.log');
M_.exo_names = 'EPS_IS';
M_.exo_names_tex = 'EPS\_IS';
M_.exo_names_long = 'EPS_IS';
M_.exo_names = char(M_.exo_names, 'EPS_PC');
M_.exo_names_tex = char(M_.exo_names_tex, 'EPS\_PC');
M_.exo_names_long = char(M_.exo_names_long, 'EPS_PC');
M_.exo_names = char(M_.exo_names, 'EPS_MP');
M_.exo_names_tex = char(M_.exo_names_tex, 'EPS\_MP');
M_.exo_names_long = char(M_.exo_names_long, 'EPS_MP');
M_.endo_names = 'I';
M_.endo_names_tex = 'I';
M_.endo_names_long = 'I';
M_.endo_names = char(M_.endo_names, 'PI');
M_.endo_names_tex = char(M_.endo_names_tex, 'PI');
M_.endo_names_long = char(M_.endo_names_long, 'PI');
M_.endo_names = char(M_.endo_names, 'Y_GAP');
M_.endo_names_tex = char(M_.endo_names_tex, 'Y\_GAP');
M_.endo_names_long = char(M_.endo_names_long, 'Y_GAP');
M_.param_names = 'alpha';
M_.param_names_tex = 'alpha';
M_.param_names_long = 'alpha';
M_.param_names = char(M_.param_names, 'beta');
M_.param_names_tex = char(M_.param_names_tex, 'beta');
M_.param_names_long = char(M_.param_names_long, 'beta');
M_.param_names = char(M_.param_names, 'epsilon');
M_.param_names_tex = char(M_.param_names_tex, 'epsilon');
M_.param_names_long = char(M_.param_names_long, 'epsilon');
M_.param_names = char(M_.param_names, 'gamma_eta');
M_.param_names_tex = char(M_.param_names_tex, 'gamma\_eta');
M_.param_names_long = char(M_.param_names_long, 'gamma_eta');
M_.param_names = char(M_.param_names, 'kappa');
M_.param_names_tex = char(M_.param_names_tex, 'kappa');
M_.param_names_long = char(M_.param_names_long, 'kappa');
M_.param_names = char(M_.param_names, 'lambda');
M_.param_names_tex = char(M_.param_names_tex, 'lambda');
M_.param_names_long = char(M_.param_names_long, 'lambda');
M_.param_names = char(M_.param_names, 'lambda_eta');
M_.param_names_tex = char(M_.param_names_tex, 'lambda\_eta');
M_.param_names_long = char(M_.param_names_long, 'lambda_eta');
M_.param_names = char(M_.param_names, 'lambda_eta_eta');
M_.param_names_tex = char(M_.param_names_tex, 'lambda\_eta\_eta');
M_.param_names_long = char(M_.param_names_long, 'lambda_eta_eta');
M_.param_names = char(M_.param_names, 'omega');
M_.param_names_tex = char(M_.param_names_tex, 'omega');
M_.param_names_long = char(M_.param_names_long, 'omega');
M_.param_names = char(M_.param_names, 'phi');
M_.param_names_tex = char(M_.param_names_tex, 'phi');
M_.param_names_long = char(M_.param_names_long, 'phi');
M_.param_names = char(M_.param_names, 'phi_pi');
M_.param_names_tex = char(M_.param_names_tex, 'phi\_pi');
M_.param_names_long = char(M_.param_names_long, 'phi_pi');
M_.param_names = char(M_.param_names, 'phi_y');
M_.param_names_tex = char(M_.param_names_tex, 'phi\_y');
M_.param_names_long = char(M_.param_names_long, 'phi_y');
M_.param_names = char(M_.param_names, 'sigma');
M_.param_names_tex = char(M_.param_names_tex, 'sigma');
M_.param_names_long = char(M_.param_names_long, 'sigma');
M_.param_names = char(M_.param_names, 'sigma_y');
M_.param_names_tex = char(M_.param_names_tex, 'sigma\_y');
M_.param_names_long = char(M_.param_names_long, 'sigma_y');
M_.param_names = char(M_.param_names, 'theta');
M_.param_names_tex = char(M_.param_names_tex, 'theta');
M_.param_names_long = char(M_.param_names_long, 'theta');
M_.param_names = char(M_.param_names, 'theta_eta');
M_.param_names_tex = char(M_.param_names_tex, 'theta\_eta');
M_.param_names_long = char(M_.param_names_long, 'theta_eta');
M_.param_names = char(M_.param_names, 'theta_y');
M_.param_names_tex = char(M_.param_names_tex, 'theta\_y');
M_.param_names_long = char(M_.param_names_long, 'theta_y');
M_.param_names = char(M_.param_names, 'sigma_pc');
M_.param_names_tex = char(M_.param_names_tex, 'sigma\_pc');
M_.param_names_long = char(M_.param_names_long, 'sigma_pc');
M_.param_names = char(M_.param_names, 'sigma_mp');
M_.param_names_tex = char(M_.param_names_tex, 'sigma\_mp');
M_.param_names_long = char(M_.param_names_long, 'sigma_mp');
M_.exo_det_nbr = 0;
M_.exo_nbr = 3;
M_.endo_nbr = 3;
M_.param_nbr = 19;
M_.orig_endo_nbr = 3;
M_.aux_vars = [];
M_.Sigma_e = zeros(3, 3);
M_.Correlation_matrix = eye(3, 3);
M_.H = 0;
M_.Correlation_matrix_ME = 1;
M_.sigma_e_is_diagonal = 1;
options_.linear = 1;
options_.block=0;
options_.bytecode=0;
options_.use_dll=0;
erase_compiled_function('NKV_static');
erase_compiled_function('NKV_dynamic');
M_.lead_lag_incidence = [
 1 0;
 2 4;
 3 5;]';
M_.nstatic = 1;
M_.nfwrd   = 2;
M_.npred   = 0;
M_.nboth   = 0;
M_.nsfwrd   = 2;
M_.nspred   = 0;
M_.ndynamic   = 2;
M_.equations_tags = {
};
M_.static_and_dynamic_models_differ = 0;
M_.exo_names_orig_ord = [1:3];
M_.maximum_lag = 0;
M_.maximum_lead = 1;
M_.maximum_endo_lag = 0;
M_.maximum_endo_lead = 1;
oo_.steady_state = zeros(3, 1);
M_.maximum_exo_lag = 0;
M_.maximum_exo_lead = 0;
oo_.exo_steady_state = zeros(3, 1);
M_.params = NaN(19, 1);
M_.NNZDerivatives = zeros(3, 1);
M_.NNZDerivatives(1) = 13;
M_.NNZDerivatives(2) = -1;
M_.NNZDerivatives(3) = -1;
M_.params( 1 ) = 0.3333333333333333;
alpha = M_.params( 1 );
M_.params( 2 ) = 0.99;
beta = M_.params( 2 );
M_.params( 3 ) = 6;
epsilon = M_.params( 3 );
M_.params( 10 ) = 1;
phi = M_.params( 10 );
M_.params( 11 ) = 1.5;
phi_pi = M_.params( 11 );
M_.params( 12 ) = 0.125;
phi_y = M_.params( 12 );
M_.params( 13 ) = 1;
sigma = M_.params( 13 );
M_.params( 15 ) = 0.6666666666666666;
theta = M_.params( 15 );
M_.params( 18 ) = 1;
sigma_pc = M_.params( 18 );
M_.params( 19 ) = 1;
sigma_mp = M_.params( 19 );
M_.params( 14 ) = 1;
sigma_y = M_.params( 14 );
M_.params( 9 ) = (1-M_.params(1))/(1-M_.params(1)+M_.params(1)*M_.params(3));
omega = M_.params( 9 );
M_.params( 6 ) = (1-M_.params(15))*(1-M_.params(15)*M_.params(2))/M_.params(15)*M_.params(9);
lambda = M_.params( 6 );
M_.params( 5 ) = M_.params(6)*(M_.params(13)+(M_.params(1)+M_.params(10))/(1-M_.params(1)));
kappa = M_.params( 5 );
%
% SHOCKS instructions
%
make_ex_;
M_.exo_det_length = 0;
M_.Sigma_e(1, 1) = 1;
M_.Sigma_e(2, 2) = 1;
M_.Sigma_e(3, 3) = 1;
options_.irf = 12;
options_.order = 1;
var_list_=[];
info = stoch_simul(var_list_);
save('NKV_results.mat', 'oo_', 'M_', 'options_');
if exist('estim_params_', 'var') == 1
  save('NKV_results.mat', 'estim_params_', '-append');
end
if exist('bayestopt_', 'var') == 1
  save('NKV_results.mat', 'bayestopt_', '-append');
end
if exist('dataset_', 'var') == 1
  save('NKV_results.mat', 'dataset_', '-append');
end
if exist('estimation_info', 'var') == 1
  save('NKV_results.mat', 'estimation_info', '-append');
end


disp(['Total computing time : ' dynsec2hms(toc) ]);
if ~isempty(lastwarn)
  disp('Note: warning(s) encountered in MATLAB/Octave code')
end
diary off
