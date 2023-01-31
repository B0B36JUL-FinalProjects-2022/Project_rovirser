import SeaTurtleID: sea
using Test, DataFrames

@testset "SeaTurtleID.jl" begin
    # Write your tests here.

    df = DataFrame()
    @test typeof(loadDataset()) == typeof(df)

    images = ["./data/images/t001/anuJvqUqBB.JPG"]
    @test typeof(loadImages()) == typeof(images)

    #add test preprocess images, wait if I have to gray them

end
