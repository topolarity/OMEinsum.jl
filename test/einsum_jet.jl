using Test
using OMEinsum
using OMEinsum: get_size_dict
using SymEngine
using LinearAlgebra: I, tr
using JET

# SymEngine.free_symbols(syms::Union{Real, Complex}) = Basic[]
# SymEngine.free_symbols(syms::AbstractArray{T}) where {T<:Union{Real, Complex}} = Basic[]
# function rand_assign(syms...)
    # fs = union(free_symbols.(syms)...)
    # Dict(zip(fs, randn(length(fs))))
# end

# function _basic_approx(x, y; atol=1e-8)
    # diff = x-y
    # assign = rand_assign(x, y)
    # length(assign) > 0 && (diff = subs.(diff, Ref.((assign...,))...))
    # nres = ComplexF64.(diff)
    # all(isapprox.(nres, 0; atol=atol))
# end

# Base.:≈(x::AbstractArray{<:Basic}, y::AbstractArray; atol=1e-8) = _basic_approx(x, y, atol=atol)
# Base.:≈(x::AbstractArray, y::AbstractArray{<:Basic}; atol=1e-8) = _basic_approx(x, y, atol=atol)
# Base.:≈(x::AbstractArray{<:Basic}, y::AbstractArray{<:Basic}; atol=1e-8) = _basic_approx(x, y, atol=atol)
# Base.Complex{T}(a::Basic) where T = T(real(a)) + im*T(imag(a))

@testset "unary einsum" begin
    size_dict = Dict(1=>3,2=>3,3=>3,4=>4,5=>5)
    ix = (1,2,3,3,4)
    x = randn(3,3,3,3,4)
    iy = (3,5,1,1,2,5)
    y = randn(3,5,3,3,3,5)
    # Diag, Sum, Repeat, Duplicate
    @test_opt einsum!(Val((ix,)), Val(iy), (x,), y, true, false, size_dict)
    @test_opt loop_einsum(StaticEinCode((ix,), iy), (x,), size_dict)
    ix = (1,2,3,4)
    x = randn(3,3,3,4)
    iy = (4,3,1,2)
    y = randn(4,3,3,3)
    # Permutedims
    @test_opt einsum!(Val((ix,)), Val(iy), (x,), y, true, false, size_dict)
    @test_opt loop_einsum(StaticEinCode((ix,), iy), (x,), size_dict)
    # None
    ix = (1,2,3,4)
    x = randn(3,3,3,4)
    iy = (1,2,3,4)
    y = randn(3,3,3,4)
    @test_opt einsum!(Val((ix,)), Val(iy), (x,), y, true, false, size_dict)
end

@testset "nary, einsum" begin
     size_dict = Dict(1=>3,2=>3,3=>3,4=>4,5=>5)
    ix = (1,2,3,3,4)
    x = randn(3,3,3,3,4)
    iy = (3,5,1)
    y = randn(3,5,3)
    iz = (1,2,3,4,5,5)
    z = randn(3,3,3,4,5,5)
    @test_opt einsum!(Val((ix, iy, iz)), Val(()), (x, y, z), fill(1.0), true, false, size_dict)
end


@testset "einsum" begin
    # matrix and vector multiplication
    a,b,c = randn(2,2), rand(2,2), rand(2,2)
    v = rand(2)
    t = randn(2,2,2,2)
    @test_opt einsum(ein"ijkl -> ijkl", (t,))
    @test_opt einsum(ein"αβγδ -> αβγδ", (t,))
    @test_opt einsum(StaticEinCode(((1,2),(2,3),(3,4)),(1,4)), (a,b,c))
    @test_opt einsum(StaticEinCode(((1,20),(20,3),(3,4)), (1,4)), (a,b,c))
    @test_opt einsum(StaticEinCode(((1,2),(2,3),(3,4)),(4,1)), (a,b,c))
    @test_opt einsum(StaticEinCode(((1,2),(2,)), (1,)), (a,v))

    # more matmul
    @test_opt ein"ij,jk -> ik"(a,a)
    @test_opt ein"ij,jk -> ki"(a,a)
    @test_opt ein"ij,kj -> ik"(a,a)
    @test_opt ein"ij,kj -> ki"(a,a)
    @test_opt ein"ji,jk -> ik"(a,a)
    @test_opt ein"ji,jk -> ki"(a,a)
    @test_opt ein"ji,kj -> ik"(a,a)
    @test_opt ein"ji,kj -> ki"(a,a)

    # contract to 0-dim array
    @test_opt einsum(StaticEinCode(((1,2),(1,2)), ()), (a,a))

    # trace
    @test_opt einsum(StaticEinCode(((1,1),),()), (a,))[]
    aa = rand(2,4,4,2)
    @test_opt einsum(StaticEinCode(((1,2,2,1),), ()), (aa,))

    # partial trace
    @test_opt einsum(StaticEinCode(((1,2,2,3),), (1,3)), (aa,))
    # with permutation
    @test_opt einsum(StaticEinCode(((1,2,2,3),), (3,1)), (aa,))

    # diag
    @test_opt einsum(StaticEinCode(((1,2,2,3),), (1,2,3)), (aa,))

    # permutation
    @test_opt einsum(StaticEinCode(((1,2),), (2,1)), (a,))
    @test_opt einsum(StaticEinCode(((1,2,3,4),),(2,3,1,4)), (t,))

    # tensor contraction
    ta = zeros(size(t)[[1,2]]...)
    for (i,j,k,l) in Iterators.product(1:2,1:2,1:2,1:2)
        ta[i,l] += t[i,j,k,l] * a[j,k]
    end
    @test_opt einsum(StaticEinCode(((1,2,3,4), (2,3)), (1,4)), (t,a))

    ta = zeros(size(t)[[1,2]]...)
    for (i,j,k,l) in Iterators.product(1:2,1:2,1:2,1:2)
        ta[i,l] += t[l,k,j,i] * a[j,k]
    end
    @test_opt einsum(StaticEinCode(((4,3,2,1), (2,3)),(1,4)), (t,a))


    @test_opt einsum(StaticEinCode(((1,2),(1,2),(1,3)), (3,)), (a,a,a))

    # index-sum
    a = rand(2,2,5)
    @test_opt einsum(StaticEinCode(((1,2,3),),(1,2)),(a,))
    # with permutation
    @test_opt ein"ijk -> ki"(a)
    @test_opt ein"ijk -> "(a)

    # Hadamard product
    a = rand(2,3)
    b = rand(2,3)
    @test_opt einsum(StaticEinCode(((1,2),(1,2)), (1,2)), (a,b))

    # Outer
    a = rand(2,3)
    b = rand(2,3)
    @test_opt einsum(StaticEinCode(((1,2),(3,4)),(1,2,3,4)),(a,b))

    # Broadcasting
    @test_opt ein"->ii"(OMEinsum.asarray(1); size_info=Dict('i'=>5))

    # trivil
    @test_opt ein"->ii"(asarray(1); size_info=Dict('i'=>5))

    # Projecting to diag
    a = rand(2,2)
    a2 = [a[1] 0; 0 a[4]]
    @test_opt einsum(StaticEinCode(((1,1),), (1,1)), (a,))

    ## operations that can be combined
    a = rand(2,2,2,2)
    @test_opt einsum(StaticEinCode(((1,1,2,2),), ()), (a,))[]

    @test_opt einsum(StaticEinCode(((1,2,3,4), (3,4,5,6)), (1,2,5,6)), (a,a))
end


@testset "string input" begin
    a = randn(3,3)
    @test_opt einsum(ein"ij,jk -> ik", (a,a))
    @test_opt ein"ij,jk -> ik"(a,a)
    @test_opt ein"αβ,βγ -> αγ"(a,a)
    # Note: the following statement is nolonger testable, since will cause load error now!
    #@test_throws ArgumentError einsum(ein"ij,123 -> k", (a,a))
end


# @testset "inplace macro input" begin
    # a = randn(2,2)
    # b = randn(2,2)
    # c = randn(2,2)
    # t = randn(2,2)
    # cc = copy(c)
    # @test_opt(@ein! t[i,k] := a[i,j] * b[j,k])
    # @test_opt(@ein! c[i,k] += a[i,j] * b[j,k])
# end


@testset "isbatchmul" begin
    for (ixs, iy) in [(((1,2), (2,3)), (1,3)), (((1,2,3), (2,3)), (1,3)),
                        (((7,1,2,3), (2,4,3,7)), (1,4,3)),
                        (((3,), (3,)), (3,)), (((3,1), (3,)), (3,1))
                        ]
        xs = ([randn(ComplexF64, fill(4,length(ix))...) for ix in ixs]...,)
        @test_opt StaticEinCode(ixs, iy)(xs...)
    end
end

@testset "issue 136" begin
    @test_opt StaticEinCode(((1,2,3),(2,)),(1,3))(ones(2,2,1), ones(2))
    @test_opt StaticEinCode(((1,2,3),(2,)),(1,3))(ones(2,2,0), ones(2))
end

@testset "fix rule cc,cb->bc" begin
    size_dict = Dict('a'=>2,'b'=>2,'c'=>2)
    for code in [ein"c,c->cc", ein"c,cc->c", ein"cc,c->cc", ein"cc,cc->cc", ein"cc,cb->bc", ein"cb,bc->cc", ein"ac,cc->ac"]
        @info code
        a = randn(fill(2, length(getixsv(code)[1]))...)
        b = randn(fill(2, length(getixsv(code)[2]))...)
        @test_opt code(a, b)
    end
end
