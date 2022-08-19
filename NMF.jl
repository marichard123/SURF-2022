using NMF
using CSV
using DataFrames
using DSP
using Statistics
using BasicBSpline
using Images
using LinearAlgebra
using Plots

println("This is the NMF program")
data_directory = raw"C:\Users\Richard\Documents\NIST_Summer_2022\XRD\Raw_Data_Folder_CoCrAl" #input directory where raw data is stored
#data_file = raw"C:\Users\Richard\Documents\NIST_Summer_2022\XRD\input_matrix_no_background.csv"
data_file = raw"C:\Users\Richard\Documents\NIST_Summer_2022\XRD\input_matrix_sqrt_with_background.csv"

use_raw_data = true #if true, add raw data to matrix without removing backgroud. If false, remove background before adding data to matrix
read_from_directory = false #if true, reads files in from a directory.if false, reads data in from a single file formed already formed into a matrix.

function ReadCSV(Filename)
    DataTable = CSV.read(Filename, DataFrame, header = false)
    DataTable = Matrix(DataTable)
    TwoTheta = zeros(Float64, length(DataTable)-1)
    Intensity = zeros(Float64, length(DataTable)-1)
    for i = 2:length(DataTable)
        TwoThetaValue = split(DataTable[i], " ")[1]
        IntensityValue = split(DataTable[i], " ")[2]
        TwoTheta[i-1] = parse(Float64,TwoThetaValue)
        Intensity[i-1] = parse(Float64,IntensityValue)
    end
    return (TwoTheta, Intensity)
end
"""
    rangewarp(x)
shift and scale `x` to the range (-1, 1)
"""
rangewarp(x) = 1 * (x .- minimum(x)) ./ (maximum(x) - minimum(x)) .- 0
"""
    sonneveld_visser(x, y)
Empirical background estimation method from 10.1107/S0021889875009417
"""
function sonneveld_visser(
    x,
    y;
    tol=1e-9,
    err_thresh=1e-12,
    max_iters=1e5,
    skip_ends=false,
    fix_bounds=(true, true),
    )
    p = deepcopy(y)
    prev = deepcopy(p)

    fix_left, fix_right = fix_bounds
    end_vals = p[[1, end]]

    for idx in 1:max_iters

        # line (2): calculate p_{i+1} + p_{i-1} / 2

        m = conv(p, [0.5, 0.0, 0.5]) #Applies convolution to m using the given filter
        m = m[2:(end - 1)] #discards first and last value of m

        # line (3): replace p with m if p > m
        p[p .> m .+ tol] = m[p .> m .+ tol] #I believe this step refers to,
        # in the paper, setting a point equal to its neighnors if it is significantly greater than its neighbors

        # p[[0, -1]] = end_vals
        if fix_left
            p[1] = end_vals[1] #literally fixes the ends; stops end points from being affected by the iterations by repeatedly setting them to their original values
        end

        if fix_right
            p[end] = end_vals[1]#literally fixes the ends; stops end points from being affected by the iterations by repeatedly setting them to their original values
        end

        delta = mean(prev .- p) #this code measures the change between iterations; if the change is small, then iteration stops on its own
        prev = deepcopy(p)

        if delta < err_thresh
            break
        end
    end

    if skip_ends
        # skip the endpoints since they are pinned in place
        x = x[2:end]
        p = p[2:end]
    else
        # clip the endpoints to match their neighbors instead...
        p[1] = p[2]
        p[end] = p[end - 1]
    end

    return x, p
end
"""
    sonneveld_visser_noise_level(y)
estimate the noise floor from the background-corrected signal
"""
function sonneveld_visser_noise_level(y; noise_thresh=1.96, max_iters=10)
    mask = y .== y

    for idx in 1:max_iters
        mu, sigma = mean(y[mask]), std(y[mask])
        mask[y .> mu .+ noise_thresh .* (3*sigma)] .= 0
    end

    return mean(y[mask]), std(y[mask]) #approximating mean and std of noise as described in the paper
end
"""
    backgroundpoints(x, y)
Estimate empirical background points using Sonneveld and Visser method
"""
function backgroundpoints(x, y; max_iters=1e5)
    _, sv = sonneveld_visser(x, y; max_iters=max_iters) #this code simply does the iterative background estimation (clipping off points that are significantly greater than their neigbors)
    #sv returned by the sonneveld_visser function is simply the background estimation by the SV fuction, _, is the unchanged? 2theta values
    mu, sig = sonneveld_visser_noise_level(y - sv) #input to the noise level function is the original y minus background estimate,
    #output is the estimated mean and standard deviation of the noise
    sel = (y - sv) .< (mu + sig)

    return (x[sel], y[sel], mu, sig, sv) #returns the points that are less than the sum of the mean and standard deviation
end

function cardinal_basis(x; n_knots=7)
    vmin, vmax = minimum(x), maximum(x)
    # dv = (vmax - vmin) / (n_knots - 1)

    return LinRange(vmin, vmax, n_knots)
    # return vmin:dv:vmax
end


function quasi_peak_indentification(x,y)
    #identifies all groups of potential peaks after values not meeting the minimum were removed
    #println("You have reached the quasi_peak_indentification function")
    StartIndicesArray = zeros(Int64, 0)
    StopIndicesArray = zeros(Int64, 0)
    for i = 1:length(y)
        if y[i] != 0
            if i==1
                append!(StartIndicesArray, i)
                #println(x[i])
            end
            if i >1
                if y[i-1] ==0
                    append!(StartIndicesArray, i-1)
                    #println(x[i])
                end
            end
            if i ==length(y)
                append!(StopIndicesArray, i)
                #println(x[i])
            end
            if i < length(y)
                if y[i+1] ==0
                    append!(StopIndicesArray, i+1)
                    #println(x[i])
                end
            end
        end

    end
    return StartIndicesArray, StopIndicesArray
end

function area_under_curve(x,y, StartIndex, StopIndex)
    #println("You have reached the area under the curve function")
    total_area = 0
    for i = StartIndex:(StopIndex-1)
        rectangle_area = 0
        triangle_area = 0
        if y[i] < y[i+1]
            lesser_value = y[i]
            greater_value = y[i+1]
        else
            lesser_value = y[i+1]
            greater_value = y[i]
        end
        rectangle_area = lesser_value* (x[i+1]-x[i])
        triangle_area = (greater_value-lesser_value) * (x[i+1]-x[i]) * .5
        #println(rectangle_area)
        #println(triangle_area)
        total_area = total_area + rectangle_area + triangle_area
    end
    #println(total_area)
    return total_area
end

function read_CSV_remove_background(filename)
    (two_theta, intensity) = ReadCSV(filename)
    #intensity = sqrt.(intensity) #remove this line of code at the end
    (background_x, background_y, mu, sig, sv) = backgroundpoints(two_theta,intensity)

    for i = 1:length(intensity)
        if intensity[i] - (sv[i]) >=0
            intensity[i] = intensity[i] - (sv[i])     #restore these three lines of code
        end
        #intensity[i] = (sv[i])

        #if two_theta[i] in background_x
    #        intensity[i] = 0 #Setting the values that do not meet a previously calculated mininum threshold to zero
        #end
        #if intensity[i] > (sv[i] + mu)
        #    intensity[i] = intensity[i] - (sv[i] + mu)
        #else
        #    intensity[i] = 0
        #end

    end
    #StartIndicesArray,StopIndicesArray = quasi_peak_indentification(two_theta,intensity)

    #for i = 1:length(StartIndicesArray)


    #    if area_under_curve(two_theta, intensity, StartIndicesArray[i],StopIndicesArray[i] ) < (sig*10)

            #intensity[StartIndicesArray[i]:StopIndicesArray[i]] = zeros(Int64,(StopIndicesArray[i]-StartIndicesArray[i])+1 )
    #    end
    #end

    return two_theta, intensity
end

function write_to_CSV(input_matrix, filename)
    #nmf_output.H = AbstractVecOrMat(nmf_output.H)
    #df = DataFrame(A = Float64[], B = Float64[])
    output_rank, output_length = size(input_matrix)
    columnnames = String[]
    for i = 1:output_length
        push!(columnnames, string(i))
    end
    columns = [Symbol(col) => Float64[] for col in columnnames]
    df = DataFrame(columns)

    for i = 1:output_rank
        push!(df,input_matrix[i, 1:end] )
    end
    #println("output rank:")
    #println(output_rank)
    CSV.write(filename, df, header = false)
    println("File successfully written to current directory")
end
"""
    Lines above this are function definitions
        Lines below this is the actual code
"""

if read_from_directory
    filenames = (readdir(data_directory, join = true))


    deleteat!(filenames, findall(x->x[end-2:end]!=".xy",filenames)) #deletes all filenames from the list that do not have an .xy ending

    sample_two_theta, sample_intensity = ReadCSV(filenames[1])
    file_length = length(sample_intensity)

    nmf_input_matrix = zeros(Float64,length(filenames), file_length)




    for i = 1:length(filenames)
        if use_raw_data
            two_theta, intensity = ReadCSV(filenames[i])
        else
            two_theta, intensity = read_CSV_remove_background(filenames[i])
            #intensity = sqrt.(intensity)
        end
        #intensity = sqrt.(intensity) # remove this line of code after running once
        nmf_input_matrix[i, 1:file_length] = intensity
        println(string(i)*" out of "*string(length(filenames))*" loaded into the input matrix")
    end
else
    #sample_two_theta, sample_intensity = ReadCSV(filenames[1])
    println("Directly reading in input matrix file")
    nmf_input_matrix = CSV.read(data_file, DataFrame, header = false)
    nmf_input_matrix = Matrix(nmf_input_matrix)
    #println(typeof(nmf_input_matrix))
    #println(size(nmf_input_matrix))
    #println(" ")
    #println(nmf_input_matrix)
    filenames = (readdir(data_directory, join = true))


    deleteat!(filenames, findall(x->x[end-2:end]!=".xy",filenames)) #deletes all filenames from the list that do not have an .xy ending

    sample_two_theta, sample_intensity = ReadCSV(filenames[1])
end


println("Matrix undergoing nmf...")
nmf_output =nnmf(nmf_input_matrix, 7; alg=:multmse, maxiter=5000, tol=1.0e-4)
println("Converged: " *string(nmf_output.converged))
println("Size of first product matrix: "*string(size(nmf_output.W)))
println("Size of second product matrix: "*string(size(nmf_output.H)))

#write_to_CSV(nmf_output.W,"sqrt_rank4_weights.csv")
write_to_CSV(nmf_output.H,"sqrt_with_background_rank7_hidden_units.csv")
#write_to_CSV(nmf_input_matrix,"input_matrix_sqrt_no_background.csv")
#write_to_CSV(nmf_input_matrix,"input_matrix_raw_data.csv")
#write_to_CSV(nmf_input_matrix,"background_of_sqrt_data.csv")
#write_to_CSV(nmf_input_matrix,"input_matrix_sqrt_with_background.csv")
#two_theta_matrix = zeros(Float64,1, length(sample_two_theta))
#two_theta_matrix[1,1:length(sample_two_theta)] = sample_two_theta
#write_to_CSV(two_theta_matrix,"two_theta_values.csv")
output_rank, output_length = size(nmf_output.H)
for i = 1:output_rank
    display(plot!(sample_two_theta, nmf_output.H[i,1:output_length],ylims = (0,1)))
end
