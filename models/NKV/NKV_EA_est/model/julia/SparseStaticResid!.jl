function SparseStaticResid!(T::Vector{<: Real}, residual::AbstractVector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real})
    @assert length(T) >= 0
    @assert length(residual) == 12
    @assert length(y) == 12
    @assert length(x) == 4
    @assert length(params) == 26
@inbounds begin
    residual[1] = (y[7]) - (y[7]-1/params[13]*(y[4]-y[5])-params[4]*y[3]-y[9]*x[1]);
    residual[2] = (y[3]) - (params[18]*x[3]+y[3]*params[7]+params[8]*y[11]-y[7]*params[17]-y[7]*params[16]);
    residual[3] = (y[9]) - (params[14]-y[3]*params[19]-y[11]*params[20]-params[21]*y[12]-y[7]*params[22]);
    residual[4] = (y[5]) - (y[5]*params[2]+y[7]*params[5]+params[24]*x[2]);
    residual[5] = (y[4]) - (y[5]*params[11]+y[7]*params[12]+params[23]*x[4]);
residual[6] = y[2];
    residual[7] = (y[6]) - (y[5]+(params[26]-1)*100);
    residual[8] = (y[8]) - (y[7]*100);
residual[9] = y[1];
    residual[10] = (y[10]) - (y[4]*4+(params[25]-1)*400);
    residual[11] = (y[11]) - (y[3]);
    residual[12] = (y[12]) - (y[11]);
end
    if ~isreal(residual)
        residual = real(residual)+imag(residual).^2;
    end
    return nothing
end

