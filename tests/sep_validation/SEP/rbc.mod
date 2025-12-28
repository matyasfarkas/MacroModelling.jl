var Capital, Output, Labour, Consumption, Efficiency, efficiency, Investment;

varexo epsilon;

parameters beta, theta, tau, alpha, psi, delta, Effstar, rho, sigma;

Effstar =  1.000 ;
beta    =  0.990 ;
theta   =  0.357 ;
tau     =  2.000 ;
alpha   =  0.450 ;
psi     = -0.200 ;
delta   =  0.010 ;
rho     =  0.800 ;
effstar =  1.000 ;
sigma   =  0.100 ;

model(use_dll);

  // Logged TFP
  efficiency = rho*efficiency(-1) + sigma*epsilon;

  // TFP
  Efficiency = Effstar*exp(efficiency);

  // Production
  Output = Efficiency*(alpha*Capital(-1)^psi+(1-alpha)*Labour^psi)^(1/psi);

  // Capital law of motion
  Capital = Output-Consumption + (1-delta)*Capital(-1);

  // Consumption/Leisure arbitrage
  (1-theta)/theta*Consumption/(1-Labour) - (1-alpha)*(Output/Labour)^(1-psi);

  // Euler equation
  (Consumption^theta*(1-Labour)^(1-theta))^(1-tau)/Consumption = beta*(Consumption(1)^theta*(1-Labour(1))^(1-theta))^(1-tau)/Consumption(1)*(alpha*(Output(1)/Capital)^(1-psi)+1-delta);

  // Investment
  Investment = Output - Consumption;

end;


steady_state_model;

  efficiency = 0;
  Efficiency = Effstar;

  Output_per_unit_of_Capital = ((1/beta-1+delta)/alpha)^(1/(1-psi));
  Consumption_per_unit_of_Capital = Output_per_unit_of_Capital-delta;
  Labour_per_unit_of_Capital = (((Output_per_unit_of_Capital/Efficiency)^psi-alpha)/(1-alpha))^(1/psi);
  Output_per_unit_of_Labour = Output_per_unit_of_Capital/Labour_per_unit_of_Capital;
  Consumption_per_unit_of_Labour = Consumption_per_unit_of_Capital/Labour_per_unit_of_Capital;

  Labour = 1/(1+Consumption_per_unit_of_Labour/((1-alpha)*theta/(1-theta)*Output_per_unit_of_Labour^(1-psi)));

  Consumption = Consumption_per_unit_of_Labour*Labour;

  Capital = Labour/Labour_per_unit_of_Capital;

  Output = Output_per_unit_of_Capital*Capital;

  Investment = Output - Consumption;

end;

steady;


shocks;
  var epsilon = 1;
end;


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

% Save positive shock (+3) IRF data to CSV
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

% Save negative shock (-3) IRF data to CSV
fprintf('\nSaving negative shock (-3 sigma) IRF data to RBC_irf_neg3.csv\n');
csv_data_neg = [(ts.data(1:60,:)./ts.data(1,:)-1)*100, (tt.data(1:60,:)./tt.data(1,:)-1)*100];
csvwrite('RBC_irf_neg3.csv', csv_data_neg);

fprintf('\n========================================\n');
fprintf('IRF data saved to:\n');
fprintf('  - RBC_irf_pos3.csv (positive +3 sigma shock)\n');
fprintf('  - RBC_irf_neg3.csv (negative -3 sigma shock)\n');
fprintf('========================================\n\n');

  % RBC Sparse Tree SEP Validation for Dynare
  % Run this after: dynare rbc.mod

  % Configuration
  periods = 10;
  order = 1;

  % Set SEP options
  options_.ep.stochastic.order = order;
  options_.ep.periods = periods;
  options_.ep.replic_nb = 1;

  % Try to enable sparse tree if available
  % Note: Sparse tree may not be available in all Dynare versions
  % If options_.ep.stochastic.algo exists, set it to 1 for fishbone
  if isfield(options_.ep.stochastic, 'algo')
      options_.ep.stochastic.algo = 1;  % Fishbone sparse tree (Adjemian-Juillard)
      fprintf('Using fishbone sparse tree algorithm\n');
  else
      fprintf('Sparse tree algo not available, using default\n');
  end

  % Run SEP from deterministic steady state
  y0 = oo_.steady_state;

  % Call extended_path with correct signature
  % In newer Dynare: extended_path(initial_conditions, sample_size, options, M, oo)
  oo_ = extended_path([], periods, [], options_, M_, oo_);

  % Extract stochastic steady state (first period of simulation)
  % Convert dseries to matrix if needed
//  if isa(oo_.endo_simul, 'dseries')
      endo_simul_data = oo_.data;
      yss_dynare = endo_simul_data(:,1);
// else
//   yss_dynare = oo_.endo_simul(:,1);
// end

  % Display results
  fprintf('\n');
  fprintf('Dynare SEP Stochastic Steady State:\n');
  fprintf('====================================\n');

  % Find variable indices
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

  % Save for later analysis
  save('rbc_dynare_sep_validation.mat', 'yss_dynare', 'oo_', 'M_', 'options_');
  fprintf('\nResults saved to rbc_dynare_sep_validation.mat\n');