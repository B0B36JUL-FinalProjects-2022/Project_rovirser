using SeaTurtleID
using Test

@testset "SeaTurtleID.jl" begin
    # Write your tests here.

    x="My name is Sergi"
    @test greet2() == x

end
