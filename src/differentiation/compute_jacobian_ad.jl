struct ForwardColorJacCache{T,T2,T3,T4,T5,T6}
    t::T
    fx::T2
    dx::T3
    p::T4
    colorvec::T5
    sparsity::T6
    chunksize::Int
end

function default_chunk_size(maxcolor)
    if maxcolor < DEFAULT_CHUNK_THRESHOLD
        Val(maxcolor)
    else
        Val(DEFAULT_CHUNK_THRESHOLD)
    end
end

getsize(::Val{N}) where N = N
getsize(N::Integer) = N

function ForwardColorJacCache(f,x,_chunksize = nothing;
                              dx = nothing,
                              colorvec=1:length(x),
                              sparsity::Union{AbstractArray,Nothing}=nothing)

    if _chunksize isa Nothing
        chunksize = default_chunk_size(maximum(colorvec))
    else
        chunksize = _chunksize
    end

    p = adapt.(typeof(x),generate_chunked_partials(x,colorvec,chunksize))
    _t = Dual{ForwardDiff.Tag(f,eltype(vec(x)))}.(vec(x),first(p))
    t = ArrayInterface.restructure(x,_t)
    if dx isa Nothing
        fx = similar(t)
        _dx = similar(x)
    else
        tup = first(first(p)) .* false
        _pi = adapt.(typeof(dx),[tup for i in 1:length(dx)])
        fx = reshape(Dual{ForwardDiff.Tag(f,eltype(vec(x)))}.(vec(dx),_pi),size(dx)...)
        _dx = dx
    end

    ForwardColorJacCache(t,fx,_dx,p,colorvec,sparsity,getsize(chunksize))
end

generate_chunked_partials(x,colorvec,N::Integer) = generate_chunked_partials(x,colorvec,Val(N))
function generate_chunked_partials(x,colorvec,::Val{chunksize}) where chunksize
    maxcolor = maximum(colorvec)
    num_of_chunks = Int(ceil(maxcolor / chunksize))
    padding_size = (chunksize - (maxcolor % chunksize)) % chunksize
    partials = colorvec .== (1:maxcolor)'
    padding_matrix = BitMatrix(undef, length(x), padding_size)
    partials = hcat(partials, padding_matrix)

    chunked_partials = map(i -> Tuple.(eachrow(partials[:,(i-1)*chunksize+1:i*chunksize])),1:num_of_chunks)
    chunked_partials

end

function forwarddiff_color_jacobian(f,
                x::AbstractArray{<:Number};
                dx = copy(x), #if dx is nothing, we will estimate dx at the cost of a function call
                colorvec = 1:length(x),
                sparsity = nothing,
                jac_prototype = nothing,
                chunksize = nothing)
    if dx isa Nothing
        dx = f(x)
    end
    forwarddiff_color_jacobian(f,x,ForwardColorJacCache(f,x,chunksize,dx=dx,colorvec=colorvec,sparsity=sparsity),jac_prototype)
end

void_setindex!(args...) = (setindex!(args...); return)

function forwarddiff_color_jacobian(f,x::AbstractArray{<:Number},jac_cache::ForwardColorJacCache,jac_prototype=nothing)
    t = jac_cache.t
    dx = jac_cache.dx
    p = jac_cache.p
    colorvec = jac_cache.colorvec
    sparsity = jac_cache.sparsity
    chunksize = jac_cache.chunksize
    color_i = 1
    maxcolor = maximum(colorvec)

    vecx = vec(x)

    J = jac_prototype isa Nothing ? (sparsity isa Nothing ? false .* vec(dx) .* vecx' : zeros(eltype(x),size(sparsity))) : zero(jac_prototype)
    nrows,ncols = size(J)

    if !(sparsity isa Nothing)
        rows_index, cols_index = ArrayInterface.findstructralnz(sparsity)
        rows_index = [rows_index[i] for i in 1:length(rows_index)]
        cols_index = [cols_index[i] for i in 1:length(cols_index)]
    end

    for i in eachindex(p)
        partial_i = p[i]
        t = reshape(Dual{ForwardDiff.Tag(f,eltype(vecx))}.(vecx, partial_i),size(t))
        fx = f(t)
        if !(sparsity isa Nothing)
            for j in 1:chunksize
                dx = vec(partials.(fx, j))
                pick_inds = [i for i in 1:length(rows_index) if colorvec[cols_index[i]] == color_i]
                rows_index_c = rows_index[pick_inds]
                cols_index_c = cols_index[pick_inds]
                if J isa SparseMatrixCSC
                    Ji = sparse(rows_index_c, cols_index_c, dx[rows_index_c],nrows,ncols)
                else
                    len_rows = length(pick_inds)
                    unused_rows = setdiff(1:nrows,rows_index_c)
                    perm_rows = sortperm(vcat(rows_index_c,unused_rows))
                    cols_index_c = vcat(cols_index_c,zeros(Int,nrows-len_rows))[perm_rows]
                    Ji = [j==cols_index_c[i] ? dx[i] : false for i in 1:nrows, j in 1:ncols]
                end
                J = J + Ji
                color_i += 1
                (color_i > maxcolor) && return J
            end
        else
            for j in 1:chunksize
                col_index = (i-1)*chunksize + j
                (col_index > ncols) && return J
                Ji = mapreduce(i -> i==col_index ? partials.(vec(fx), j) : zeros(nrows), hcat, 1:ncols)
                J = J + (size(Ji)!=size(J) ? reshape(Ji,size(J)) : Ji) #branch when size(dx) == (1,) => size(Ji) == (1,) while size(J) == (1,1)
            end
        end
    end
    J
end

function forwarddiff_color_jacobian!(J::AbstractMatrix{<:Number},
                f,
                x::AbstractArray{<:Number};
                dx = similar(x,size(J,1)),
                colorvec = 1:length(x),
                sparsity = ArrayInterface.has_sparsestruct(J) ? J : nothing)
    forwarddiff_color_jacobian!(J,f,x,ForwardColorJacCache(f,x,dx=dx,colorvec=colorvec,sparsity=sparsity))
end

function forwarddiff_color_jacobian!(J::AbstractMatrix{<:Number},
                f,
                x::AbstractArray{<:Number},
                jac_cache::ForwardColorJacCache)

    t = jac_cache.t
    fx = jac_cache.fx
    dx = jac_cache.dx
    p = jac_cache.p
    colorvec = jac_cache.colorvec
    sparsity = jac_cache.sparsity
    chunksize = jac_cache.chunksize
    color_i = 1
    maxcolor = maximum(colorvec)
    fill!(J, zero(eltype(J)))

    if FiniteDiff._use_findstructralnz(sparsity)
        rows_index, cols_index = ArrayInterface.findstructralnz(sparsity)
    else
        rows_index = 1:size(J,1)
        cols_index = 1:size(J,2)
    end

    vecx = vec(x)
    vect = vec(t)
    vecfx= vec(fx)
    vecdx= vec(dx)

    ncols=size(J,2)

    for i in eachindex(p)
        partial_i = p[i]
        vect .= Dual{ForwardDiff.Tag(f,eltype(vecx))}.(vecx, partial_i)
        f(fx,t)
        if !(sparsity isa Nothing)
            for j in 1:chunksize
                dx .= partials.(fx, j)
                if ArrayInterface.fast_scalar_indexing(dx)
                    #dx is implicitly used in vecdx
                    FiniteDiff._colorediteration!(J,sparsity,rows_index,cols_index,vecdx,colorvec,color_i,ncols)
                else
                    #=
                    J.nzval[rows_index] .+= (colorvec[cols_index] .== color_i) .* dx[rows_index]
                    or
                    J[rows_index, cols_index] .+= (colorvec[cols_index] .== color_i) .* dx[rows_index]
                    += means requires a zero'd out start
                    =#
                    if J isa SparseMatrixCSC
                        @. void_setindex!((J.nzval,),getindex((J.nzval,),rows_index) + (getindex((colorvec,),cols_index) == color_i) * getindex((vecdx,),rows_index),rows_index)
                    else
                        @. void_setindex!((J,),getindex((J,),rows_index, cols_index) + (getindex((colorvec,),cols_index) == color_i) * getindex((vecdx,),rows_index),rows_index, cols_index)
                    end
                end
                color_i += 1
                (color_i > maxcolor) && return
            end
        else
            for j in 1:chunksize
                col_index = (i-1)*chunksize + j
                (col_index > ncols) && return
                J[:, col_index] .= partials.(vecfx, j)
            end
        end
    end
    nothing
end
