function ds = dynamic_set_auxiliary_series(ds, params)
%
% Computes auxiliary variables of the dynamic model
%
ds.AUX_ENDO_LAG_98_1=ds.Q(-1);
ds.AUX_ENDO_LAG_98_2=ds.AUX_ENDO_LAG_98_1(-1);
ds.AUX_ENDO_LAG_98_3=ds.AUX_ENDO_LAG_98_2(-1);
ds.AUX_ENDO_LAG_125_1=ds.W_C(-1);
ds.AUX_ENDO_LAG_125_2=ds.AUX_ENDO_LAG_125_1(-1);
ds.AUX_ENDO_LAG_125_3=ds.AUX_ENDO_LAG_125_2(-1);
ds.AUX_ENDO_LAG_127_1=ds.W_C_ST(-1);
ds.AUX_ENDO_LAG_127_2=ds.AUX_ENDO_LAG_127_1(-1);
ds.AUX_ENDO_LAG_127_3=ds.AUX_ENDO_LAG_127_2(-1);
ds.AUX_ENDO_LAG_133_1=ds.Y(-1);
ds.AUX_ENDO_LAG_133_2=ds.AUX_ENDO_LAG_133_1(-1);
ds.AUX_ENDO_LAG_133_3=ds.AUX_ENDO_LAG_133_2(-1);
ds.AUX_ENDO_LAG_145_1=ds.Y_ST(-1);
ds.AUX_ENDO_LAG_145_2=ds.AUX_ENDO_LAG_145_1(-1);
ds.AUX_ENDO_LAG_145_3=ds.AUX_ENDO_LAG_145_2(-1);
end
