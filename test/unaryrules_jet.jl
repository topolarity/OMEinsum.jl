using OMEinsum, Test
using OMEinsum: unary_einsum!, Duplicate, Sum, Tr, Permutedims, Repeat, Diag, Identity
using SymEngine: Basic

using JET

@testset "Duplicate + JET" begin
    ix = (1,2,3)
    iy = (3,2,1,1,2)
    size_dict = Dict(1=>3,2=>3,3=>3)
    x = randn(3,3,3)
    y = randn(3,3,3,3,3)
    @test_opt OMEinsum.duplicate!(y, x, ix, iy, true, false)
    @test_opt OMEinsum.loop_einsum(StaticEinCode((ix,),iy), (x,), size_dict)
    @test_opt unary_einsum!(Duplicate(), Val(ix), Val(iy), x, y, true, false)
    @test_opt OMEinsum.loop_einsum(StaticEinCode((ix,),iy), (x,), size_dict)
end

@testset "Diag + JET" begin
    ix = (3,2,1,1,2)
    iy = (1,2,3)
    size_dict = Dict(1=>3,2=>3,3=>3)
    x = randn(3,3,3,3,3)
    y = randn(3,3,3)
    @test_opt unary_einsum!(Diag(), Val(ix), Val(iy), x, copy(y), true, false)
    @test_opt unary_einsum!(Diag(), Val(ix), Val(iy), x, copy(y), 1.0, 2.0)
end

@testset "Repeat + JET" begin
    ix = (1,2,3)
    iy = (3,4,2,1)
    size_dict = Dict(1=>3,2=>3,3=>3,4=>5)
    x = randn(3,3,3)
    y = randn(3,5,3,3)
    @test_opt unary_einsum!(Repeat(), Val(ix), Val(iy), x, y, true, false)
end

@testset "Tr + JET" begin
    a = rand(5,5)
    @test_opt unary_einsum!(Tr(), Val((1,1)), Val(()), a, fill(1.0), true, false)
    @test_opt unary_einsum!(Tr(), Val((1,1)), Val(()), a, fill(Basic(0)), 1, 0)

end

@testset "Permutedims + JET" begin
    a = rand(5,5,3)
    @test_opt unary_einsum!(Permutedims(), Val((1,2,3)), Val((2,3,1)), a, zeros(5, 3, 5), true, false)
end

@testset "Identity + JET" begin
    a = rand(5,5,3)
    @test_opt unary_einsum!(Identity(), Val((1,2,3)), Val((1,2,3)), a, ones(5, 5, 3), 2.0, 3.0)
end

@testset "Sum + JET" begin
    # index-sum
    a = rand(2,2,5)
    @test_opt unary_einsum!(Sum(), Val((1, 2, 3)), Val((1,2)), a, zeros(2, 2), true, false)
    a = Basic.(rand(1:100, 2,2,5))
    @test_opt unary_einsum!(Sum(), Val((1, 2, 3)), Val((1,2)), a, zeros(Basic, 2, 2), 1, 0)
end
