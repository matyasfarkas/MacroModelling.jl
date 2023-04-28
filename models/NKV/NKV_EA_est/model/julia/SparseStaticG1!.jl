function SparseStaticG1!(T::Vector{<: Real}, g1_v::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real})
    @assert length(T) >= 0
    @assert length(g1_v) == 29
    @assert length(y) == 12
    @assert length(x) == 4
    @assert length(params) == 26
@inbounds begin
g1_v[1]=1;
g1_v[2]=1;
g1_v[3]=params[4];
g1_v[4]=1-params[7];
g1_v[5]=params[19];
g1_v[6]=(-1);
g1_v[7]=1/params[13];
g1_v[8]=1;
g1_v[9]=(-4);
g1_v[10]=(-(1/params[13]));
g1_v[11]=1-params[2];
g1_v[12]=(-params[11]);
g1_v[13]=(-1);
g1_v[14]=1;
g1_v[15]=(-((-params[17])-params[16]));
g1_v[16]=params[22];
g1_v[17]=(-params[5]);
g1_v[18]=(-params[12]);
g1_v[19]=(-100);
g1_v[20]=1;
g1_v[21]=x[1];
g1_v[22]=1;
g1_v[23]=1;
g1_v[24]=(-params[8]);
g1_v[25]=params[20];
g1_v[26]=1;
g1_v[27]=(-1);
g1_v[28]=params[21];
g1_v[29]=1;
end
    if ~isreal(g1_v)
        g1_v = real(g1_v)+2*imag(g1_v);
    end
    return nothing
end

