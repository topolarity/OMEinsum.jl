## non-inplace einsum
@doc raw"
    einsum(code::EinCode, xs, size_dict)

Return the tensor that results from contracting the tensors `xs` according to the contraction code `code`.

### Arguments
- `code`: The einsum notation, which can be an instance of [`EinCode`](@ref), [`NestedEinsum`](@ref), or [`SlicedEinsum`](@ref).
- `xs` - the input tensors
- `size_dict` - a dictionary that maps index-labels to their sizes

### Examples

```jldoctest; setup = :(using OMEinsum)
julia> a, b = rand(2,2), rand(2,2);

julia> einsum(EinCode((('i','j'),('j','k')),('i','k')), (a, b)) ≈ a * b
true

julia> einsum(EinCode((('i','j'),('j','k')),('k','i')), (a, b)) ≈ permutedims(a * b, (2,1))
true
```
"
function einsum(code::AbstractEinsum, @nospecialize(xs::Tuple), size_dict::Dict=get_size_dict!(getixs(code), xs, Dict{labeltype(code),Int}()))
    iy = getiy(code)
    dims = ntuple(Val(length(iy))) do i
        @noinline size_dict[iy[i]]
    end
    y = get_output_array(xs, dims; fillzero=true)
    einsum!(code, xs, y, true, false, size_dict)
end
# identical to above, but more aggressively specialized
function einsum(code::StaticEinCode, @nospecialize(xs::Tuple), size_dict::Dict=get_size_dict!(getixs(code), xs, Dict{labeltype(code),Int}()))
    iy = getiy(code)
    dims = ntuple(Val(length(iy))) do i
        @noinline size_dict[iy[i]]
    end
    y = get_output_array(xs, dims; fillzero=true)
    einsum!(code, xs, y, true, false, size_dict)
end

# inplace einsum, EinCode as the input
"""
    einsum!(code::EinCode, xs, y, sx, sy, size_dict)

Inplace version of `einsum`. The result is stored in `y`.

### Arguments
- `code`: The einsum notation, which can be an instance of [`EinCode`](@ref), [`NestedEinsum`](@ref), or [`SlicedEinsum`](@ref).
- `xs`: The input tensors.
- `y`: The output tensor.
- `sx`: Scale `x` by `sx`.
- `sy`: Scale `y` by `sy`.
- `size_dict`: A dictionary that maps index-labels to their sizes.
"""
function einsum!(code::EinCode, @nospecialize(xs::Tuple), @nospecialize(y), sx, sy, size_dict::Dict=get_size_dict(getixs(code), xs))
    einsum!(Val(getixs(code)), Val(getiy(code)), xs, y, sx, sy, size_dict)
end
# identical to above, but more aggressively specialized
function einsum!(code::StaticEinCode, xs::Tuple, y, sx, sy, size_dict::Dict=get_size_dict(getixs(code), xs))
    @inline einsum!(Val(getixs(code)), Val(getiy(code)), xs, y, sx, sy, size_dict)
end
# inplace einsum, the fallback
function einsum!(::Val{ixs}, ::Val{iy}, @nospecialize(xs::Tuple), @nospecialize(y), sx, sy, size_dict::Dict) where {ixs, iy}
    # @debug "fallback to loop_einsum" ixs => iy size.(xs)
    @inline loop_einsum!(Val(ixs), Val(iy), (xs...,), y, sx, sy, size_dict)
end

struct UnaryOperation{LT}
    type
    ix::Tuple{Vararg{LT}}
    iy::Tuple{Vararg{LT}}
end
# for unary operations
# overhead ~ 2.3us
# @benchmark OMEinsum.einsum(DefaultRule(), $((('a', 'a', 'b'),)), $(('c', 'b','a')), (x,), $(Dict('a'=>1, 'b'=>1, 'c'=>1))) setup=(x=randn(1,1,1))
function unary_pipeline(ix::Tuple{Vararg{LT}}, iy::Tuple{Vararg{LT}}) where {LT}
    ix_unique = _unique(ix)
    iy_unique = _unique(iy)
    iy_a = filter(i -> i ∈ ix, iy_unique)

    operations = ntuple(Val(4)) do i
        @inline
        if i == 1 && length(ix_unique) != length(ix)  # diag
            UnaryOperation(Diag(), ix, ix_unique)
        elseif i == 2
            if length(ix_unique) != length(iy_a)  # sum
                UnaryOperation(Sum(), ix_unique, iy_a)
            elseif ix_unique != iy_a   # permute, high freq
                UnaryOperation(Permutedims(), ix_unique, iy_a)
            end
        elseif i == 3 && length(iy_a) != length(iy_unique)  # repeat
            UnaryOperation(Repeat(), iy_a, iy_unique)
        elseif i == 4 && length(iy_unique) != length(iy)  # duplicate
            UnaryOperation(Duplicate(), iy_unique, iy)
        end
        nothing
    end
    return filter(!isnothing, operations)
end

function einsum!(::Val{ixs}, ::Val{iy}, xs::NTuple{1,Any}, y, sx, sy, size_dict::Dict{LT}) where {ixs, iy, LT}
    # @debug "compiling unary" ixs[1] => iy size(xs[1])
    pipeline = unary_pipeline(ixs[1], iy)

    if length(pipeline) == 0
        @flatten_addmul! sy * y + sx * xs[1]
        return y
    end

    lasttensor = Ref(xs[1])
    ntuple(Val(length(pipeline))) do k
        @inline
        op = pipeline[k]
        if k == length(pipeline)  # last operation
            unary_einsum!(op.type, op.ix, op.iy, lasttensor[], y, sx, sy)
        else
            dims = ntuple(Val(length(op.iy))) do l
                @noinline size_dict[l]
            end
            cache = similar(y, dims)
            unary_einsum!(op.type, op.ix, op.iy, lasttensor[], cache, true, false)
            lasttensor[] = cache
        end
    end

    return y
end

# there are too many combination in the binary case, so nospecialize
function einsum!(::Val{ixs}, ::Val{iy}, xs::NTuple{2,Any}, y, sx, sy, size_dict::Dict{LT}) where {ixs, iy, LT}
    # ix1v, ix2v = _collect.(Ref(LT), ixs)
    # @debug "compiling binary" ixs => iyv size.(xs)
    x1, x2 = xs
    c1, c2, cy, s1, s2, s3, i1, i2, iyb = @noinline analyze_binary(Val(ixs[1]), Val(ixs[2]), Val(iy), size_dict)
    rule = SimpleBinaryRule{i1,i2,iyb}() # this is now compile-time-known!
    xs1 = simplifyto(Val(ixs[1]), Val(c1), x1, size_dict)
    xs2 = simplifyto(Val(ixs[2]), Val(c2), x2, size_dict)
    x1_ = _reshape(xs1, s1)
    x2_ = _reshape(xs2, s2)
    if cy != iy
        y_ = _similar(y, s3)
        dims = ntuple(Val(length(cy))) do i
            @noinline size_dict[cy[i]]
        end
        y_  = binary_einsum!(rule, x1_, x2_, y_, true, false)
        y_ = _reshape(y_, dims)
        return einsum!(Val((cy,)), Val(iy), (y_,), y, sx, sy, size_dict)
    else
        binary_einsum!(rule, x1_, x2_, _reshape(y, s3), sx, sy)
        return y
    end
end
_reshape(a, dims) = @noinline reshape(a, dims)
_similar(a, dims) = @noinline similar(a, dims)

function simplifyto(::Val{ix1}, ::Val{c1}, x1, size_dict::Dict{LT}) where {ix1, c1, LT}
    if c1 != ix1
        dims = ntuple(Val(length(c1))) do i
            @noinline size_dict[c1[i]]
        end
        xs1 = _similar(x1, dims)
        # TODO: specialization?
        return einsum!(Val((ix1,)), Val(c1), (x1,), xs1, true, false, size_dict)
    else
        return x1
    end
end

"""
Get the expected labels.
"""
function analyze_binary(::Val{ix1}, ::Val{ix2}, ::Val{iy}, size_dict::Dict{T,Int}) where {T, ix1, ix2, iy}
    ix_inner, ix1_outer, ix2_outer, batch = _analyze_binary_input(Val(ix1), Val(ix2), Val(iy))

    indices(::Val{label}) where label = begin
        if label === 'i'
            ix1_outer
        elseif label === 'j'
            ix_inner
        elseif label === 'k'
            ix2_outer
        elseif label === 'l'
            batch
        end
    end

    labels = filter((l)->!isempty(indices(Val(l))),
                    ('i','j','k','l'))
    sizes = NamedTuple{Symbol.(labels)}(
        ntuple(Val(length(labels))) do i
            dims = indices(Val(labels[i]))
            prod(map(x -> (@noinline size_dict[x]), dims))
        end
    )

    # TODO: has_idxs === iy (?)
    c1 = (ix1_outer..., ix_inner...,  batch...)
    c2 = (ix_inner...,  ix2_outer..., batch...)
    cy = (ix1_outer..., ix2_outer..., batch...)

    i1    = filter(in(('i','j','l')), labels)
    i2    = filter(in(('j','k','l')), labels)
    iyb   = filter(in(('i','k','l')), labels)

    s1    = values(Base.structdiff(sizes, NamedTuple{(:k,)})) # i j l
    s2    = values(Base.structdiff(sizes, NamedTuple{(:i,)})) # j k l
    s3    = values(Base.structdiff(sizes, NamedTuple{(:j,)})) # i k l

    return c1, c2, cy, s1, s2, s3, i1, i2, iyb
end

# TODO: name clash
Base.@assume_effects :foldable function _unique(items::Tuple{Vararg{T}}) where T
    length(items) == 0 && return ()
    uniqued = T[]
    for item in items
        if item ∉ uniqued
            push!(uniqued, item)
        end
    end
    (uniqued...,)
end

function _analyze_binary_input(::Val{ix1}, ::Val{ix2}, ::Val{iy}) where {ix1, ix2, iy}
    # XXX: functions are carefully chosen to be eligible for compile-time execution
    ix1_batch = _unique(filter((l1) -> l1 ∈ ix2 && l1 ∈ iy, ix1))
    ix1_inner = _unique(filter((l1) -> l1 ∈ ix2 && l1 ∉ iy, ix1))
    ix1_outer = _unique(filter((l1) -> l1 ∉ ix2 && l1 ∈ iy, ix1))
    ix2_outer = _unique(filter((l2) -> l2 ∉ ix1 && l2 ∈ iy, ix2))

    ix1_inner, ix1_outer, ix2_outer, ix1_batch
end
