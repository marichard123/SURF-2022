clear all; close all; clc;
plot_components = true;
log_scale = false;
plot_vertical_lines = true;

two_theta = readtable('two_theta_values_matlab.csv');
two_theta = table2array(two_theta);
two_theta = two_theta(1,:);
ResidualPlot = readtable('fitted_peaks_matrix31.csv');
IndividualPeaks = readtable('composition_fitted_peaks_matrix31.csv');
peak_parameters = readtable('fitted_peak_parameters31.csv');
peak_parameters = table2cell(peak_parameters);
IndividualPeaks = table2array(IndividualPeaks);
ResidualPlot = table2array(ResidualPlot);
OriginalPlot = readtable('input_matrix_raw_data.csv');
OriginalPlot = table2array(OriginalPlot);


IndividualPeaksIndex = 1;



figure;
pause(2)
for i = 1:177
    hold on;
    plot(two_theta,OriginalPlot(i,1:end));
    hold on;
    plot(two_theta,ResidualPlot(i,1:end));
    ylim([0 10])
    legend('Original XRD Plot', 'Fitted Plot')
    ylabel("Intensity (log scale)")
    xlabel("Two theta (degrees)")
    title(strcat('Sample #'), num2str(i));
    if log_scale
        set(gca, 'YScale', 'log')
    end
    
    if plot_components
        while IndividualPeaks(IndividualPeaksIndex,size(IndividualPeaks,2)) ~= zeros(1,size(IndividualPeaks,2))
            hold on;
            plot(two_theta, IndividualPeaks(IndividualPeaksIndex,:));
            IndividualPeaksIndex = IndividualPeaksIndex+1;
        end
        IndividualPeaksIndex = IndividualPeaksIndex+1;
    end
    
    if plot_vertical_lines
      index = find([peak_parameters{:,1}] == i-1);
      peak_centers = peak_parameters(:,4);
      peak_centers = peak_centers(index);
      for ii = 1:length(peak_centers)
          xline(peak_centers{ii},'r')
      end
    end
    pause(1);
    clf;
end