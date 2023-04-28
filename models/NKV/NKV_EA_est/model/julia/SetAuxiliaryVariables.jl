function set_auxiliary_variables!(y, x, params)
#
# Computes auxiliary variables of the static model
#
@inbounds begin
y[11]=y[3];
y[12]=y[11];
end
end
