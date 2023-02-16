import SeaTurtleID as ST
using Test, DataFrames, JSON

@testset "SeaTurtleID.jl" begin

    include("testDataset.jl")
    include("testImages.jl")
    include("testTrain.jl")
   
end
