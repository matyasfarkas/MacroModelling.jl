function SparseDynamicG1!(T::Vector{<: Real}, g1_v::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    @assert length(T) >= 0
    @assert length(g1_v) == 42
    @assert length(y) == 36
    @assert length(x) == 4
    @assert length(params) == 26
@inbounds begin
g1_v[1]=(-params[7]);
g1_v[2]=params[19];
g1_v[3]=(-1);
g1_v[4]=1;
g1_v[5]=params[22];
g1_v[6]=1;
g1_v[7]=(-params[8]);
g1_v[8]=params[20];
g1_v[9]=(-1);
g1_v[10]=params[21];
g1_v[11]=1;
g1_v[12]=1;
g1_v[13]=params[4];
g1_v[14]=1;
g1_v[15]=1/params[13];
g1_v[16]=1;
g1_v[17]=(-4);
g1_v[18]=1;
g1_v[19]=(-params[11]);
g1_v[20]=(-1);
g1_v[21]=1;
g1_v[22]=(-1);
g1_v[23]=1;
g1_v[24]=params[17];
g1_v[25]=(-params[5]);
g1_v[26]=(-params[12]);
g1_v[27]=(-100);
g1_v[28]=(-1);
g1_v[29]=1;
g1_v[30]=x[1];
g1_v[31]=1;
g1_v[32]=1;
g1_v[33]=1;
g1_v[34]=1;
g1_v[35]=(-(1/params[13]));
g1_v[36]=(-params[2]);
g1_v[37]=(-1);
g1_v[38]=params[16];
g1_v[39]=y[21];
g1_v[40]=(-params[24]);
g1_v[41]=(-params[18]);
g1_v[42]=(-params[23]);
end
    return nothing
end

