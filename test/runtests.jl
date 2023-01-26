using SeaTurtleID
using Test, DataFrames

@testset "SeaTurtleID.jl" begin
    # Write your tests here.

    df = DataFrame()
    @test typeof(loadDataset()) == typeof(df)

end
