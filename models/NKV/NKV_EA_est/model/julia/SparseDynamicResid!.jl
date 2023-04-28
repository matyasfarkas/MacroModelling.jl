function SparseDynamicResid!(T::Vector{<: Real}, residual::AbstractVector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    @assert length(T) >= 0
    @assert length(residual) == 12
    @assert length(y) == 36
    @assert length(x) == 4
    @assert length(params) == 26
@inbounds begin
    residual[1] = (y[19]) - (y[31]-1/params[13]*(y[16]-y[29])-params[4]*y[15]-y[21]*x[1]);
    residual[2] = (y[15]) - (params[18]*x[3]+params[7]*y[3]+params[8]*y[11]-y[19]*params[17]-y[31]*params[16]);
    residual[3] = (y[21]) - (params[14]-y[3]*params[19]-params[20]*y[11]-params[21]*y[12]-params[22]*y[7]);
    residual[4] = (y[17]) - (y[29]*params[2]+y[19]*params[5]+params[24]*x[2]);
    residual[5] = (y[16]) - (y[17]*params[11]+y[19]*params[12]+params[23]*x[4]);
    residual[6] = (y[14]) - (y[20]-y[8]);
    residual[7] = (y[18]) - (y[17]+(params[26]-1)*100);
    residual[8] = (y[20]) - (y[19]*100);
    residual[9] = (y[13]) - (y[18]-y[6]);
    residual[10] = (y[22]) - (y[16]*4+(params[25]-1)*400);
    residual[11] = (y[23]) - (y[3]);
    residual[12] = (y[24]) - (y[11]);
end
    return nothing
end

