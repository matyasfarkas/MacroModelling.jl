%
% Status : main Dynare file
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

clearvars -global
clear_persistent_variables(fileparts(which('dynare')), false)
tic0 = tic;
% Define global variables.
global M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info estimation_info
options_ = [];
M_.fname = 'rbc';
M_.dynare_version = '7-unstable-2025-12-17-2025-5cbb8bae';
oo_.dynare_version = '7-unstable-2025-12-17-2025-5cbb8bae';
options_.dynare_version = '7-unstable-2025-12-17-2025-5cbb8bae';
%
% Some global variables initialization
%
global_initialization;
M_.exo_names = cell(1,1);
M_.exo_names_tex = cell(1,1);
M_.exo_names_long = cell(1,1);
M_.exo_names(1) = {'epsilon'};
M_.exo_names_tex(1) = {'epsilon'};
M_.exo_names_long(1) = {'epsilon'};
M_.endo_names = cell(7,1);
M_.endo_names_tex = cell(7,1);
M_.endo_names_long = cell(7,1);
M_.endo_names(1) = {'Capital'};
M_.endo_names_tex(1) = {'Capital'};
M_.endo_names_long(1) = {'Capital'};
M_.endo_names(2) = {'Output'};
M_.endo_names_tex(2) = {'Output'};
M_.endo_names_long(2) = {'Output'};
M_.endo_names(3) = {'Labour'};
M_.endo_names_tex(3) = {'Labour'};
M_.endo_names_long(3) = {'Labour'};
M_.endo_names(4) = {'Consumption'};
M_.endo_names_tex(4) = {'Consumption'};
M_.endo_names_long(4) = {'Consumption'};
M_.endo_names(5) = {'Efficiency'};
M_.endo_names_tex(5) = {'Efficiency'};
M_.endo_names_long(5) = {'Efficiency'};
M_.endo_names(6) = {'efficiency'};
M_.endo_names_tex(6) = {'efficiency'};
M_.endo_names_long(6) = {'efficiency'};
M_.endo_names(7) = {'Investment'};
M_.endo_names_tex(7) = {'Investment'};
M_.endo_names_long(7) = {'Investment'};
M_.endo_partitions = struct();
M_.param_names = cell(9,1);
M_.param_names_tex = cell(9,1);
M_.param_names_long = cell(9,1);
M_.param_names(1) = {'beta'};
M_.param_names_tex(1) = {'beta'};
M_.param_names_long(1) = {'beta'};
M_.param_names(2) = {'theta'};
M_.param_names_tex(2) = {'theta'};
M_.param_names_long(2) = {'theta'};
M_.param_names(3) = {'tau'};
M_.param_names_tex(3) = {'tau'};
M_.param_names_long(3) = {'tau'};
M_.param_names(4) = {'alpha'};
M_.param_names_tex(4) = {'alpha'};
M_.param_names_long(4) = {'alpha'};
M_.param_names(5) = {'psi'};
M_.param_names_tex(5) = {'psi'};
M_.param_names_long(5) = {'psi'};
M_.param_names(6) = {'delta'};
M_.param_names_tex(6) = {'delta'};
M_.param_names_long(6) = {'delta'};
M_.param_names(7) = {'Effstar'};
M_.param_names_tex(7) = {'Effstar'};
M_.param_names_long(7) = {'Effstar'};
M_.param_names(8) = {'rho'};
M_.param_names_tex(8) = {'rho'};
M_.param_names_long(8) = {'rho'};
M_.param_names(9) = {'sigma'};
M_.param_names_tex(9) = {'sigma'};
M_.param_names_long(9) = {'sigma'};
M_.param_partitions = struct();
M_.exo_det_nbr = 0;
M_.exo_nbr = 1;
M_.endo_nbr = 7;
M_.param_nbr = 9;
M_.orig_endo_nbr = 7;
M_.aux_vars = [];
M_.heterogeneity_aggregates = {
};
M_.Sigma_e = zeros(1, 1);
M_.Correlation_matrix = eye(1, 1);
M_.Skew_e = zeros(1, 1, 1);
M_.H = 0;
M_.Correlation_matrix_ME = 1;
M_.sigma_e_is_diagonal = true;
M_.det_shocks = [];
M_.surprise_shocks = [];
M_.learnt_shocks = [];
M_.learnt_endval = [];
M_.heteroskedastic_shocks.Qvalue_orig = [];
M_.heteroskedastic_shocks.Qscale_orig = [];
M_.matched_irfs = {};
M_.matched_irfs_weights = {};
M_.perfect_foresight_controlled_paths = [];
options_.linear = false;
options_.block = false;
options_.bytecode = false;
options_.use_dll = true;
options_.ramsey_policy = false;
options_.discretionary_policy = false;
M_.nonzero_hessian_eqs = [2 3 5 6];
M_.hessian_eq_zero = isempty(M_.nonzero_hessian_eqs);
M_.eq_nbr = 7;
M_.ramsey_orig_eq_nbr = 0;
M_.ramsey_orig_endo_nbr = 0;
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
 1 3 0;
 0 4 10;
 0 5 11;
 0 6 12;
 0 7 0;
 2 8 0;
 0 9 0;]';
M_.nstatic = 2;
M_.nfwrd   = 3;
M_.npred   = 2;
M_.nboth   = 0;
M_.nsfwrd   = 3;
M_.nspred   = 2;
M_.ndynamic   = 5;
M_.dynamic_tmp_nbr = [13; 15; 5; 0; ];
M_.equations_tags = {
  1 , 'name' , 'efficiency' ;
  2 , 'name' , 'Efficiency' ;
  3 , 'name' , 'Output' ;
  4 , 'name' , 'Capital' ;
  5 , 'name' , '5' ;
  6 , 'name' , '6' ;
  7 , 'name' , 'Investment' ;
};
M_.mapping.Capital.eqidx = [3 4 6 ];
M_.mapping.Output.eqidx = [3 4 5 6 7 ];
M_.mapping.Labour.eqidx = [3 5 6 ];
M_.mapping.Consumption.eqidx = [4 5 6 7 ];
M_.mapping.Efficiency.eqidx = [2 3 ];
M_.mapping.efficiency.eqidx = [1 2 ];
M_.mapping.Investment.eqidx = [7 ];
M_.mapping.epsilon.eqidx = [1 ];
M_.static_and_dynamic_models_differ = false;
M_.has_external_function = false;
M_.block_structure.time_recursive = false;
M_.block_structure.block(1).Simulation_Type = 1;
M_.block_structure.block(1).endo_nbr = 2;
M_.block_structure.block(1).mfs = 2;
M_.block_structure.block(1).equation = [ 1 2];
M_.block_structure.block(1).variable = [ 6 5];
M_.block_structure.block(1).is_linear = true;
M_.block_structure.block(1).bytecode_jacob_cols_to_sparse = [1 3 4 ];
M_.block_structure.block(2).Simulation_Type = 8;
M_.block_structure.block(2).endo_nbr = 4;
M_.block_structure.block(2).mfs = 4;
M_.block_structure.block(2).equation = [ 6 3 4 5];
M_.block_structure.block(2).variable = [ 1 3 2 4];
M_.block_structure.block(2).is_linear = false;
M_.block_structure.block(2).bytecode_jacob_cols_to_sparse = [1 5 6 7 8 10 11 12 ];
M_.block_structure.block(3).Simulation_Type = 1;
M_.block_structure.block(3).endo_nbr = 1;
M_.block_structure.block(3).mfs = 1;
M_.block_structure.block(3).equation = [ 7];
M_.block_structure.block(3).variable = [ 7];
M_.block_structure.block(3).is_linear = true;
M_.block_structure.block(3).bytecode_jacob_cols_to_sparse = [2 ];
M_.block_structure.block(1).g1_sparse_rowval = int32([]);
M_.block_structure.block(1).g1_sparse_colval = int32([]);
M_.block_structure.block(1).g1_sparse_colptr = int32([]);
M_.block_structure.block(2).g1_sparse_rowval = int32([2 3 1 3 1 2 4 2 3 4 1 3 4 1 1 1 ]);
M_.block_structure.block(2).g1_sparse_colval = int32([1 1 5 5 6 6 6 7 7 7 8 8 8 10 11 12 ]);
M_.block_structure.block(2).g1_sparse_colptr = int32([1 3 3 3 3 5 8 11 14 14 15 16 17 ]);
M_.block_structure.block(3).g1_sparse_rowval = int32([]);
M_.block_structure.block(3).g1_sparse_colval = int32([]);
M_.block_structure.block(3).g1_sparse_colptr = int32([]);
M_.block_structure.variable_reordered = [ 6 5 1 3 2 4 7];
M_.block_structure.equation_reordered = [ 1 2 6 3 4 5 7];
M_.block_structure.incidence(1).lead_lag = -1;
M_.block_structure.incidence(1).sparse_IM = [
 1 6;
 3 1;
 4 1;
];
M_.block_structure.incidence(2).lead_lag = 0;
M_.block_structure.incidence(2).sparse_IM = [
 1 6;
 2 5;
 2 6;
 3 2;
 3 3;
 3 5;
 4 1;
 4 2;
 4 4;
 5 2;
 5 3;
 5 4;
 6 1;
 6 3;
 6 4;
 7 2;
 7 4;
 7 7;
];
M_.block_structure.incidence(3).lead_lag = 1;
M_.block_structure.incidence(3).sparse_IM = [
 6 2;
 6 3;
 6 4;
];
M_.block_structure.dyn_tmp_nbr = 17;
M_.state_var = [6 1 ];
M_.maximum_lag = 1;
M_.maximum_lead = 1;
M_.maximum_endo_lag = 1;
M_.maximum_endo_lead = 1;
oo_.steady_state = zeros(7, 1);
M_.maximum_exo_lag = 0;
M_.maximum_exo_lead = 0;
oo_.exo_steady_state = zeros(1, 1);
M_.params = NaN(9, 1);
M_.endo_trends = struct('deflator', cell(7, 1), 'log_deflator', cell(7, 1), 'growth_factor', cell(7, 1), 'log_growth_factor', cell(7, 1));
M_.dynamic_g1_sparse_rowval = int32([3 4 1 4 6 3 4 5 7 3 5 6 4 5 6 7 2 3 1 2 7 6 6 6 1 ]);
M_.dynamic_g1_sparse_colval = int32([1 1 6 8 8 9 9 9 9 10 10 10 11 11 11 11 12 12 13 13 14 16 17 18 22 ]);
M_.dynamic_g1_sparse_colptr = int32([1 3 3 3 3 3 4 4 6 10 13 17 19 21 22 22 23 24 25 25 25 25 26 ]);
M_.dynamic_g2_sparse_indices = int32([2 13 13 ;
3 1 1 ;
3 1 10 ;
3 1 12 ;
3 10 10 ;
3 10 12 ;
5 9 9 ;
5 9 10 ;
5 10 10 ;
5 10 11 ;
6 8 8 ;
6 8 16 ;
6 8 17 ;
6 8 18 ;
6 16 16 ;
6 16 17 ;
6 16 18 ;
6 10 10 ;
6 10 11 ;
6 17 17 ;
6 17 18 ;
6 11 11 ;
6 18 18 ;
]);
M_.lhs = {
'efficiency'; 
'Efficiency'; 
'Output'; 
'Capital'; 
'Consumption*(1-theta)/theta/(1-Labour)-(1-alpha)*(Output/Labour)^(1-psi)'; 
'(Consumption^theta*(1-Labour)^(1-theta))^(1-tau)/Consumption'; 
'Investment'; 
};
M_.dynamic_mcp_equations_reordering = [1; 2; 3; 4; 5; 6; 7; ];
M_.static_tmp_nbr = [8; 6; 0; 0; ];
M_.block_structure_stat.block(1).Simulation_Type = 3;
M_.block_structure_stat.block(1).endo_nbr = 1;
M_.block_structure_stat.block(1).mfs = 1;
M_.block_structure_stat.block(1).equation = [ 1];
M_.block_structure_stat.block(1).variable = [ 6];
M_.block_structure_stat.block(2).Simulation_Type = 1;
M_.block_structure_stat.block(2).endo_nbr = 1;
M_.block_structure_stat.block(2).mfs = 1;
M_.block_structure_stat.block(2).equation = [ 2];
M_.block_structure_stat.block(2).variable = [ 5];
M_.block_structure_stat.block(3).Simulation_Type = 6;
M_.block_structure_stat.block(3).endo_nbr = 4;
M_.block_structure_stat.block(3).mfs = 4;
M_.block_structure_stat.block(3).equation = [ 3 4 5 6];
M_.block_structure_stat.block(3).variable = [ 1 4 3 2];
M_.block_structure_stat.block(4).Simulation_Type = 1;
M_.block_structure_stat.block(4).endo_nbr = 1;
M_.block_structure_stat.block(4).mfs = 1;
M_.block_structure_stat.block(4).equation = [ 7];
M_.block_structure_stat.block(4).variable = [ 7];
M_.block_structure_stat.variable_reordered = [ 6 5 1 4 3 2 7];
M_.block_structure_stat.equation_reordered = [ 1 2 3 4 5 6 7];
M_.block_structure_stat.incidence.sparse_IM = [
 1 6;
 2 5;
 2 6;
 3 1;
 3 2;
 3 3;
 3 5;
 4 1;
 4 2;
 4 4;
 5 2;
 5 3;
 5 4;
 6 1;
 6 2;
 6 3;
 6 4;
 7 2;
 7 4;
 7 7;
];
M_.block_structure_stat.tmp_nbr = 12;
M_.block_structure_stat.block(1).g1_sparse_rowval = int32([1 ]);
M_.block_structure_stat.block(1).g1_sparse_colval = int32([1 ]);
M_.block_structure_stat.block(1).g1_sparse_colptr = int32([1 2 ]);
M_.block_structure_stat.block(2).g1_sparse_rowval = int32([]);
M_.block_structure_stat.block(2).g1_sparse_colval = int32([]);
M_.block_structure_stat.block(2).g1_sparse_colptr = int32([]);
M_.block_structure_stat.block(3).g1_sparse_rowval = int32([1 2 4 2 3 4 1 3 4 1 2 3 4 ]);
M_.block_structure_stat.block(3).g1_sparse_colval = int32([1 1 1 2 2 2 3 3 3 4 4 4 4 ]);
M_.block_structure_stat.block(3).g1_sparse_colptr = int32([1 4 7 10 14 ]);
M_.block_structure_stat.block(4).g1_sparse_rowval = int32([]);
M_.block_structure_stat.block(4).g1_sparse_colval = int32([]);
M_.block_structure_stat.block(4).g1_sparse_colptr = int32([]);
M_.static_g1_sparse_rowval = int32([3 4 6 3 4 5 6 7 3 5 6 4 5 6 7 2 3 1 2 7 ]);
M_.static_g1_sparse_colval = int32([1 1 1 2 2 2 2 2 3 3 3 4 4 4 4 5 5 6 6 7 ]);
M_.static_g1_sparse_colptr = int32([1 4 9 12 16 18 20 21 ]);
M_.static_mcp_equations_reordering = [1; 2; 3; 4; 5; 6; 7; ];
M_.params(7) = 1.000;
Effstar = M_.params(7);
M_.params(1) = 0.990;
beta = M_.params(1);
M_.params(2) = 0.357;
theta = M_.params(2);
M_.params(3) = 2.000;
tau = M_.params(3);
M_.params(4) = 0.450;
alpha = M_.params(4);
M_.params(5) = (-0.200);
psi = M_.params(5);
M_.params(6) = 0.010;
delta = M_.params(6);
M_.params(8) = 0.800;
rho = M_.params(8);
effstar =  1.000 ;
M_.params(9) = 0.100;
sigma = M_.params(9);
steady;
%
% SHOCKS instructions
%
M_.Sigma_e(1, 1) = 1;
options_.ep.stochastic.IntegrationAlgorithm = 'Tensor-Gaussian-Quadrature';
options_.ep.stochastic.quadrature.nodes = 3;
options_.ep.stack_solve_algo = 7;
options_.ep.solve_algo = 0;
options_.ep.stochastic.algo = 1;
innovations = zeros(80,1);
innovations(1) = 3;
addpath ../../matlab
close all
options_.ep.periods = 400;
maxorder = 10;
tt = extended_path(oo_.steady_state, 80, innovations, options_, M_, oo_);
ds = transpose(oo_.steady_state);
for order=maxorder:-1:0
options_.ep.stochastic.order = order;
switch order
case maxorder
ts = extended_path(transpose(ds(end,:)), 1, innovations(1), options_, M_, oo_);
ds = [ds; ts.data(2,:)];
case 0
ts = extended_path(transpose(ds(end,:)), 80, zeros(80,1), options_, M_, oo_);
ds = [ds; ts.data(2:end,:)];
otherwise
ts = extended_path(transpose(ds(end,:)), 1, 0, options_, M_, oo_);
ds = [ds; ts.data(2,:)];
end
end
ts = dseries(ds, '1Y', M_.endo_names);
spfirf(tt, ts, 1)
fprintf('\nSaving positive shock (+3 sigma) IRF data to RBC_irf_pos3.csv\n');
csv_data_pos = [(ts.data(1:60,:)./ts.data(1,:)-1)*100, (tt.data(1:60,:)./tt.data(1,:)-1)*100];
csvwrite('RBC_irf_pos3.csv', csv_data_pos);
innovations(1) = -3;
tt = extended_path(oo_.steady_state, 80, innovations, options_, M_, oo_);
ds = transpose(oo_.steady_state);
for order=maxorder:-1:0
options_.ep.stochastic.order = order;
switch order
case maxorder
ts = extended_path(transpose(ds(end,:)), 1, innovations(1), options_, M_, oo_);
ds = [ds; ts.data(2,:)];
case 0
ts = extended_path(transpose(ds(end,:)), 80, zeros(80,1), options_, M_, oo_);
ds = [ds; ts.data(2:end,:)];
otherwise
ts = extended_path(transpose(ds(end,:)), 1, 0, options_, M_, oo_);
ds = [ds; ts.data(2,:)];
end
end
ts = dseries(ds, '1Y', M_.endo_names);
spfirf(tt, ts, 2)
fprintf('\nSaving negative shock (-3 sigma) IRF data to RBC_irf_neg3.csv\n');
csv_data_neg = [(ts.data(1:60,:)./ts.data(1,:)-1)*100, (tt.data(1:60,:)./tt.data(1,:)-1)*100];
csvwrite('RBC_irf_neg3.csv', csv_data_neg);
fprintf('\n========================================\n');
fprintf('IRF data saved to:\n');
fprintf('  - RBC_irf_pos3.csv (positive +3 sigma shock)\n');
fprintf('  - RBC_irf_neg3.csv (negative -3 sigma shock)\n');
fprintf('========================================\n\n');
periods = 10;
order = 1;
options_.ep.stochastic.order = order;
options_.ep.periods = periods;
options_.ep.replic_nb = 1;
if isfield(options_.ep.stochastic, 'algo')
options_.ep.stochastic.algo = 1;  
fprintf('Using fishbone sparse tree algorithm\n');
else
fprintf('Sparse tree algo not available, using default\n');
end
y0 = oo_.steady_state;
oo_ = extended_path([], periods, [], options_, M_, oo_);
endo_simul_data = oo_.data;
yss_dynare = endo_simul_data(:,1);
fprintf('\n');
fprintf('Dynare SEP Stochastic Steady State:\n');
fprintf('====================================\n');
var_names = {'Capital', 'Consumption', 'Efficiency', 'Investment', 'Labour', 'Output', 'efficiency'};
for i = 1:length(var_names)
idx = strmatch(var_names{i}, M_.endo_names, 'exact');
if ~isempty(idx)
fprintf('  %-15s  %.8f\n', var_names{i}, yss_dynare(idx));
end
end
fprintf('\n');
fprintf('Expected MacroModelling.jl values:\n');
fprintf('==================================\n');
fprintf('  Capital          17.77020407\n');
fprintf('  Consumption       1.15488900\n');
fprintf('  Efficiency        1.00000000\n');
fprintf('  Investment        0.17770204\n');
fprintf('  Labour            0.31922644\n');
fprintf('  Output            1.33259104\n');
fprintf('  efficiency       -0.00000000\n');
save('rbc_dynare_sep_validation.mat', 'yss_dynare', 'oo_', 'M_', 'options_');
fprintf('\nResults saved to rbc_dynare_sep_validation.mat\n');


oo_.time = toc(tic0);
disp(['Total computing time : ' dynsec2hms(oo_.time) ]);
if ~exist([M_.dname filesep 'Output'],'dir')
    mkdir(M_.dname,'Output');
end
save([M_.dname filesep 'Output' filesep 'rbc_results.mat'], 'oo_', 'M_', 'options_');
if exist('estim_params_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'rbc_results.mat'], 'estim_params_', '-append');
end
if exist('bayestopt_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'rbc_results.mat'], 'bayestopt_', '-append');
end
if exist('dataset_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'rbc_results.mat'], 'dataset_', '-append');
end
if exist('estimation_info', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'rbc_results.mat'], 'estimation_info', '-append');
end
if exist('dataset_info', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'rbc_results.mat'], 'dataset_info', '-append');
end
if exist('oo_recursive_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'rbc_results.mat'], 'oo_recursive_', '-append');
end
if exist('options_mom_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'rbc_results.mat'], 'options_mom_', '-append');
end
if ~isempty(lastwarn)
  disp('Note: warning(s) encountered in MATLAB/Octave code')
end
