import SeaTurtleID as ST
using Test, DataFrames, JSON

@testset "images.jl" begin

    images = loadImages()
    @test typeof(images) == Vector{Matrix{RGB{N0f8}}}

    imagesProc = Vector{Matrix{Float32}}(undef, 7582)
    totalImages = processImages(images)
    @test typeof(totalImages) == typeof(imagesProc)  
    @test size(imagesProc) == size(totalImages)

    turtlesID = Vector{String}(undef, 7582)
    labels = loadLabels()
    @test typeof(loadLabels()) == typeof(turtlesID)
    @test length(turtlesID) == length(labels)


end
