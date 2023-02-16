import SeaTurtleID as ST
using Test, DataFrames, JSON

@testset "loadDataset.jl" begin

    # Write your tests here.
    df = DataFrame()
    data = ST.loadDataset()
    @test typeof(data) == typeof(df) 

end
