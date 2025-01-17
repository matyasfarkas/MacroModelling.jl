%
% Status : main Dynare file
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

if isoctave || matlab_ver_less_than('8.6')
    clear all
else
    clearvars -global
    clear_persistent_variables(fileparts(which('dynare')), false)
end
tic0 = tic;
% Define global variables.
global M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info estimation_info ys0_ ex0_
options_ = [];
M_.fname = 'RBC_kz';
M_.dynare_version = '5.1';
oo_.dynare_version = '5.1';
options_.dynare_version = '5.1';
%
% Some global variables initialization
%
global_initialization;
M_.exo_names = cell(1,1);
M_.exo_names_tex = cell(1,1);
M_.exo_names_long = cell(1,1);
M_.exo_names(1) = {'EPSz'};
M_.exo_names_tex(1) = {'EPSz'};
M_.exo_names_long(1) = {'EPSz'};
M_.endo_names = cell(3,1);
M_.endo_names_tex = cell(3,1);
M_.endo_names_long = cell(3,1);
M_.endo_names(1) = {'k'};
M_.endo_names_tex(1) = {'k'};
M_.endo_names_long(1) = {'k'};
M_.endo_names(2) = {'z'};
M_.endo_names_tex(2) = {'z'};
M_.endo_names_long(2) = {'z'};
M_.endo_names(3) = {'k_obs'};
M_.endo_names_tex(3) = {'k\_obs'};
M_.endo_names_long(3) = {'k_obs'};
M_.endo_partitions = struct();
M_.param_names = cell(5,1);
M_.param_names_tex = cell(5,1);
M_.param_names_long = cell(5,1);
M_.param_names(1) = {'alpha'};
M_.param_names_tex(1) = {'alpha'};
M_.param_names_long(1) = {'alpha'};
M_.param_names(2) = {'beta'};
M_.param_names_tex(2) = {'beta'};
M_.param_names_long(2) = {'beta'};
M_.param_names(3) = {'sigma'};
M_.param_names_tex(3) = {'sigma'};
M_.param_names_long(3) = {'sigma'};
M_.param_names(4) = {'rho'};
M_.param_names_tex(4) = {'rho'};
M_.param_names_long(4) = {'rho'};
M_.param_names(5) = {'delta'};
M_.param_names_tex(5) = {'delta'};
M_.param_names_long(5) = {'delta'};
M_.param_partitions = struct();
M_.exo_det_nbr = 0;
M_.exo_nbr = 1;
M_.endo_nbr = 3;
M_.param_nbr = 5;
M_.orig_endo_nbr = 3;
M_.aux_vars = [];
options_.varobs = cell(1, 1);
options_.varobs(1)  = {'k_obs'};
options_.varobs_id = [ 3  ];
M_ = setup_solvers(M_);
M_.Sigma_e = zeros(1, 1);
M_.Correlation_matrix = eye(1, 1);
M_.H = 0;
M_.Correlation_matrix_ME = 1;
M_.sigma_e_is_diagonal = true;
M_.det_shocks = [];
M_.surprise_shocks = [];
M_.heteroskedastic_shocks.Qvalue_orig = [];
M_.heteroskedastic_shocks.Qscale_orig = [];
options_.linear = false;
options_.block = false;
options_.bytecode = false;
options_.use_dll = false;
M_.nonzero_hessian_eqs = 1;
M_.hessian_eq_zero = isempty(M_.nonzero_hessian_eqs);
M_.orig_eq_nbr = 3;
M_.eq_nbr = 3;
M_.ramsey_eq_nbr = 0;
M_.set_auxiliary_variables = exist(['./+' M_.fname '/set_auxiliary_variables.m'], 'file') == 2;
M_.epilogue_names = {};
M_.epilogue_var_list_ = {};
M_.orig_maximum_endo_lag = 1;
M_.orig_maximum_endo_lead = 1;
M_.orig_maximum_exo_lag = 0;
M_.orig_maximum_exo_lead = 0;
M_.orig_maximum_exo_det_lag = 0;
M_.orig_maximum_exo_det_lead = 0;
M_.orig_maximum_lag = 1;
M_.orig_maximum_lead = 1;
M_.orig_maximum_lag_with_diffs_expanded = 1;
M_.lead_lag_incidence = [
 1 3 6;
 2 4 0;
 0 5 0;]';
M_.nstatic = 1;
M_.nfwrd   = 0;
M_.npred   = 1;
M_.nboth   = 1;
M_.nsfwrd   = 1;
M_.nspred   = 2;
M_.ndynamic   = 2;
M_.dynamic_tmp_nbr = [5; 4; 0; 0; ];
M_.model_local_variables_dynamic_tt_idxs = {
};
M_.equations_tags = {
  1 , 'name' , '1' ;
  2 , 'name' , 'z' ;
  3 , 'name' , 'k_obs' ;
};
M_.mapping.k.eqidx = [1 3 ];
M_.mapping.z.eqidx = [1 2 ];
M_.mapping.k_obs.eqidx = [3 ];
M_.mapping.EPSz.eqidx = [2 ];
M_.static_and_dynamic_models_differ = false;
M_.has_external_function = false;
M_.state_var = [1 2 ];
M_.exo_names_orig_ord = [1:1];
M_.maximum_lag = 1;
M_.maximum_lead = 1;
M_.maximum_endo_lag = 1;
M_.maximum_endo_lead = 1;
oo_.steady_state = zeros(3, 1);
M_.maximum_exo_lag = 0;
M_.maximum_exo_lead = 0;
oo_.exo_steady_state = zeros(1, 1);
M_.params = NaN(5, 1);
M_.endo_trends = struct('deflator', cell(3, 1), 'log_deflator', cell(3, 1), 'growth_factor', cell(3, 1), 'log_growth_factor', cell(3, 1));
M_.NNZDerivatives = [10; 12; -1; ];
M_.static_tmp_nbr = [4; 1; 0; 0; ];
M_.model_local_variables_static_tt_idxs = {
};
M_.params(1) = 0.5;
alpha = M_.params(1);
M_.params(2) = 0.95;
beta = M_.params(2);
M_.params(3) = 0.01;
sigma = M_.params(3);
M_.params(4) = 0.2;
rho = M_.params(4);
M_.params(5) = 0.02;
delta = M_.params(5);
%
% SHOCKS instructions
%
M_.exo_det_length = 0;
M_.Sigma_e(1, 1) = 1;
estim_params_.var_exo = zeros(0, 10);
estim_params_.var_endo = zeros(0, 10);
estim_params_.corrx = zeros(0, 11);
estim_params_.corrn = zeros(0, 11);
estim_params_.param_vals = zeros(0, 10);
estim_params_.param_vals = [estim_params_.param_vals; 1, 0.25, NaN, NaN, 5, 0.15, 0.45, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 2, 0.95, NaN, NaN, 5, 0.92, 0.9999, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 3, 0.01, NaN, NaN, 5, 0.0, 0.1, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 4, 0.2, NaN, NaN, 5, 0.0, 1.0, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 5, 0.02, NaN, NaN, 5, 0.0001, 0.1, NaN, NaN, NaN ];
options_.TeX=1;  
options_.mh_drop = 0.1;
options_.mh_jscale = 0.3;
options_.mh_nblck = 1;
options_.mh_replic = 100000;
options_.mode_compute = 0;
options_.smoother = true;
options_.datafile = 'rbc_obs';
options_.mode_file = 'RBC_kz_mode';
options_.first_obs = 1;
options_.nobs = 20;
options_.order = 1;
var_list_ = {};
oo_recursive_=dynare_estimation(var_list_);
figure;
disp(' ');
disp('NOW I DO STABILITY MAPPING and prepare sample for Reduced form Mapping');
disp(' ');
disp('Press ENTER to continue'); pause(5);
options_gsa = struct();
options_gsa.Nsam = 512;
options_gsa.nodisplay = true;
options_gsa.redform = 1;
options_.nodisplay = true;
dynare_sensitivity(options_gsa);
disp(' ');
disp('ANALYSIS OF REDUCED FORM COEFFICIENTS');
disp(' ');
disp('Press ENTER to continue'); pause(5);
options_gsa = struct();
options_gsa.Nsam = 512;
options_gsa.load_stab = 1;
options_gsa.nodisplay = true;
options_gsa.redform = 1;
options_gsa.stab = 0;
options_gsa.threshold_redform = [-1 0];
options_gsa.namendo = {':'};
options_gsa.namexo = {':'};
options_gsa.namlagendo = {':'};
options_.nodisplay = true;
dynare_sensitivity(options_gsa);


oo_.time = toc(tic0);
disp(['Total computing time : ' dynsec2hms(oo_.time) ]);
if ~exist([M_.dname filesep 'Output'],'dir')
    mkdir(M_.dname,'Output');
end
save([M_.dname filesep 'Output' filesep 'RBC_kz_results.mat'], 'oo_', 'M_', 'options_');
if exist('estim_params_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'RBC_kz_results.mat'], 'estim_params_', '-append');
end
if exist('bayestopt_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'RBC_kz_results.mat'], 'bayestopt_', '-append');
end
if exist('dataset_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'RBC_kz_results.mat'], 'dataset_', '-append');
end
if exist('estimation_info', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'RBC_kz_results.mat'], 'estimation_info', '-append');
end
if exist('dataset_info', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'RBC_kz_results.mat'], 'dataset_info', '-append');
end
if exist('oo_recursive_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'RBC_kz_results.mat'], 'oo_recursive_', '-append');
end
if ~isempty(lastwarn)
  disp('Note: warning(s) encountered in MATLAB/Octave code')
end
