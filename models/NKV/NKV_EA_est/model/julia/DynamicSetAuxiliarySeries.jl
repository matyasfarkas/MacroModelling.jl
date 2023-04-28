function dynamic_set_auxiliary_series!(ds, params)
#
# Computes auxiliary variables of the dynamic model
#
@inbounds begin
ds.AUX_ENDO_LAG_2_1 .=lag(ds.ETA);
ds.AUX_ENDO_LAG_2_2 .=lag(ds.AUX_ENDO_LAG_2_1);
end
end
