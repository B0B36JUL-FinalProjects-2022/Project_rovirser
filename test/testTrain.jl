import SeaTurtleID as ST
using Test

@testset "train.jl" begin

    X_train, y_train, X_test, y_test = loadSets()
    @test typeof(X_train) == Array{Float32, 4}
    @test size(y_train) ==  (400, 6066)
    @test typeof(X_test) == Array{Float32, 4}
    @test size(y_test) ==  (400, 1516)

end
