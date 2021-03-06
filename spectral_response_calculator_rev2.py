#import glob
#import os
#import numbers
#import re
import numpy as np
import pandas as pd
#from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
#import json

from sklearn.metrics import mean_squared_error

from scipy import signal, misc



'''
User Input
'''

constant_for_population_calculation = 6.04E21 #constant


'''
Prepare data from the databases
'''

# Reads the spectral data files (and converts to numeric and coerces uncovertable values to NaN)

energy_nist_database = pd.read_csv("nist_energy_levels.csv", header=0, names = ['element_name',
                                                                'species',
                                                                'degeneracy',
                                                                'energy_ev'])

nist_database = pd.read_csv("nist_spectral_data.csv", header=0, names = ['element_name',
                                                'species',
                                                'wavelength_nm',
                                                'transition_probability',
                                                'lower_energy',
                                                'upper_energy',
                                                'lower_degeneracy',
                                                'upper_degeneracy',
                                                'ionization_energy'])
#nist_database = pd.to_numeric(test_nist_database, errors = 'ignore')
nist_database['species'] = pd.to_numeric(nist_database['species'],errors = 'coerce')
nist_database['wavelength_nm'] = pd.to_numeric(nist_database['wavelength_nm'],errors = 'coerce')
nist_database['transition_probability'] = pd.to_numeric(nist_database['transition_probability'],errors = 'coerce')
nist_database['lower_energy'] = pd.to_numeric(nist_database['lower_energy'],errors = 'coerce')
nist_database['upper_energy'] = pd.to_numeric(nist_database['upper_energy'],errors = 'coerce')
nist_database['lower_degeneracy'] = pd.to_numeric(nist_database['lower_degeneracy'],errors = 'coerce')
nist_database['upper_degeneracy'] = pd.to_numeric(nist_database['upper_degeneracy'],errors = 'coerce')
nist_database['ionization_energy'] = pd.to_numeric(nist_database['ionization_energy'],errors = 'coerce')


'''
Define functions
'''


class single_element_plasma(object):
    
    def __init__(self, element, plasma_temperature = 1, electron_density = 1E17, resolution = 0, wavelength_min = 200, wavelength_max = 1000):
        
        self.element = element
        self.plasma_temperature = plasma_temperature
        self.electron_density = electron_density
        self.species = 1
        self.resolution = resolution
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        
        
    def partition_function(self, curr_species):       
        
        if curr_species < 4:
            select_energy_level_data = energy_nist_database[(energy_nist_database['element_name'] == self.element) & (energy_nist_database['species'] == curr_species)] #Extract data that meets criteria
            level_degeneracy = select_energy_level_data['degeneracy']
            level_energy = pd.to_numeric(select_energy_level_data['energy_ev'])
            partition_function = level_degeneracy * np.exp(- level_energy / self.plasma_temperature)    
        
        return partition_function


    def species_population_ratio(self, species_lower, species_higher):

        select_energy_level_data = nist_database[(nist_database['element_name'] == self.element) & (nist_database['species'] == species_lower)].max()
        level_ionization_energy = select_energy_level_data['ionization_energy']
        temporary = (1 / self.electron_density) * constant_for_population_calculation * (self.plasma_temperature ** (3/2))
        temporary = temporary * (self.partition_function(species_higher).sum() / self.partition_function(species_lower).sum())
        temporary = temporary * np.exp(- level_ionization_energy / self.plasma_temperature)
       
        return temporary


    def species_population_percentages(self):
        
        s2_to_s1 = self.species_population_ratio(1, 2)
        s3_to_s2 = self.species_population_ratio(2, 3)
        s1_to_s1 = 1
        s3_to_s1 = s2_to_s1 * s3_to_s2
        sum_ratios = s3_to_s1 + s2_to_s1 + s1_to_s1
        first = s1_to_s1 / sum_ratios
        second = s2_to_s1 / sum_ratios
        third = s3_to_s1 / sum_ratios
        population_percentages = [first, second, third]
        
        return population_percentages
    
    
    def line_intensities(self, curr_species):
        
        select_energy_level_data = nist_database[(nist_database['element_name'] == self.element) & (nist_database['species'] == curr_species)].dropna()
        radiative_lifetime = select_energy_level_data['transition_probability']
        upper_degeneracy = select_energy_level_data['upper_degeneracy']
        upper_energy = select_energy_level_data['upper_energy']
        species_concentration = self.species_population_percentages()[curr_species-1]
        part_fun = self.partition_function(curr_species).sum()   
        line_intensity = species_concentration * radiative_lifetime * upper_degeneracy
        line_intensity = line_intensity  * np.exp(- (upper_energy / self.plasma_temperature))
        line_intensity = line_intensity * (1 / part_fun)
        emission_wavelength = select_energy_level_data['wavelength_nm'] #uses database wavelength
        line_intensities = pd.concat([emission_wavelength, line_intensity], axis=1)
        line_intensities.columns = ['wavelength_nm', 'intensity']
        
        return line_intensities
    
        
    def calculate_emission_spectrum_single_element(self):
       
        component_one = self.line_intensities(1)
        component_two = self.line_intensities(2)
        component_three = self.line_intensities(3)
        spectrum_single_element = component_one
        spectrum_single_element = spectrum_single_element.append(component_two, ignore_index = True)
        spectrum_single_element = spectrum_single_element.append(component_three, ignore_index = True)
        
        return spectrum_single_element


class multi_element_plasma(object):

    def __init__(self, element_list, concentration_list, plasma_temperature = 1, electron_density = 1E17, resolution = 0, wavelength_min = 200, wavelength_max = 1000):
        
        self.element_list = element_list
        self.concentration_list = concentration_list
        self.plasma_temperature = plasma_temperature
        self.electron_density = electron_density
        self.species = 1
        self.resolution = resolution
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max


    def species_population_percentages(self):
        
        population_percentages = pd.DataFrame()        
        for element in self.element_list:
            temp1 = single_element_plasma(element, plasma_temperature = self.plasma_temperature, electron_density = self.electron_density, resolution = self.resolution, wavelength_min = self.wavelength_min, wavelength_max = self.wavelength_max)
            temp2 = pd.DataFrame(temp1.species_population_percentages(), index = ['Neutral', 'Singly', 'Doubly'], columns = [element]).T
            population_percentages = population_percentages.append(temp2)
        
        return population_percentages


    def emission_spectrum_multi_element(self):
        
        multi_emission_spectrum = pd.DataFrame()
        for element, concentration in zip(self.element_list, self.concentration_list):
            temp1 = single_element_plasma(element, plasma_temperature = self.plasma_temperature, electron_density = self.electron_density, resolution = self.resolution, wavelength_min = self.wavelength_min, wavelength_max = self.wavelength_max)
            temp2 = pd.DataFrame(temp1.calculate_emission_spectrum_single_element())
            temp2 = temp2 * concentration
            multi_emission_spectrum = multi_emission_spectrum.append(temp2)
            
        return multi_emission_spectrum


    def binned_spectrum(self):
        
        limit_intensity = 0
        temp1 = self.emission_spectrum_multi_element()
        temp1 = temp1.round({'wavelength_nm': self.resolution})
        temp1 = temp1.groupby('wavelength_nm')['intensity'].sum()
        temp1 = temp1[(temp1.index >= self.wavelength_min) & (temp1.index <= self.wavelength_max)]
        temp1 = temp1.where(temp1 >= limit_intensity)
        
        return temp1
    

class adjusted_spectrum(object):
    
    def __init__ (self, spectrum_to_adjust, resolution = 0):
        
        self.spectrum_to_adjust = spectrum_to_adjust
        self.resolution = resolution
        self.wavelength_min = 200.
        self.wavelength_max = 1000.
        self.wavelength_partition = 1 + ((self.wavelength_max - self.wavelength_min) * (10 ** resolution))
#        self.wavelength_partition = 80001

    def gaussian(self, current_wavelength):
        ''' Define a gaussian efficiency function '''
        x = current_wavelength
        x0 = 600
        sigma = 250
        gaussian_value = np.exp(-np.power((x - x0)/sigma, 2.)/2.)
        
        return gaussian_value

    def spectrum_adjust(self):
        ''' Adjust the meausred intensities by a detection efficiency function '''
        adjusted_spectrum = pd.DataFrame()
        for current_index, intensity in enumerate(self.spectrum_to_adjust):
            self.current_wavelength = self.spectrum_to_adjust.index[current_index]
            temp_val1 = intensity * self.gaussian(self.current_wavelength)
            temp2 = pd.DataFrame(temp_val1, columns = ['adjusted_intensity'], index = [self.current_wavelength])
            adjusted_spectrum = adjusted_spectrum.append(temp2)        
        return adjusted_spectrum

    def full_scale_spectra(self):
        theoretical_lines_binned = pd.DataFrame(self.spectrum_to_adjust)
        
        adjusted_lines_binned = self.spectrum_adjust()
        
        combined_spectra = pd.concat([theoretical_lines_binned, adjusted_lines_binned], axis =1)
        
        index_for_full_scale_spectra = pd.DataFrame(np.linspace(self.wavelength_min, self.wavelength_max, self.wavelength_partition, endpoint = True)).round(self.resolution)
        
        full_scale_spectra = pd.DataFrame(index = index_for_full_scale_spectra[0])
        
        full_scale_spectra = pd.concat([full_scale_spectra, combined_spectra], axis = 1)
        
        full_scale_spectra = full_scale_spectra.fillna(0)
        
        return full_scale_spectra

    def ratio_spectra(self):
        temp1 = self.full_scale_spectra()
        
        temp1 = temp1['adjusted_intensity'] / temp1['intensity']
        index_for_full_scale_spectra = pd.DataFrame(np.linspace(self.wavelength_min, self.wavelength_max, self.wavelength_partition, endpoint = True)).round(self.resolution)
        full_scale_spectra_ratio = pd.DataFrame(index = index_for_full_scale_spectra[0])
        
        full_scale_spectra_ratio = pd.concat([full_scale_spectra_ratio, temp1], axis = 1)
        
        full_scale_spectra_ratio = full_scale_spectra_ratio.interpolate(method = 'linear', axis = 0, limit_direction = 'both')
        return full_scale_spectra_ratio

    def ground_truth(self):
        '''Currently doesn't work properly'''
        index_for_full_scale_spectra = pd.DataFrame(np.linspace(self.wavelength_min, self.wavelength_max, self.wavelength_partition, endpoint = True)).round(self.resolution)
        adjusted_values = index_for_full_scale_spectra.apply(self.gaussian)
#        adjusted_values = adjusted_values.reindex(index_for_full_scale_spectra[0])
#        ground_truth_function = pd.DataFrame([adjusted_values[0]], index = index_for_full_scale_spectra[0], columns = ['ground_truth_intensity'])

        return adjusted_values





''' Create plasma spectra '''

plasma1 = multi_element_plasma(['Cr', 'Ni'], [1, 3]) # make plasma object

adjusted_plasma1 = adjusted_spectrum(plasma1.binned_spectrum()) # adjust the produced spectra

test_ratio_spectra = adjusted_plasma1.ratio_spectra()

test_full_spectra = adjusted_plasma1.full_scale_spectra()

#test_ground_truth = adjusted_plasma1.ground_truth()

#determined_rmse = np.sqrt(mean_squared_error(test_ground_truth, test_ratio_spectra))

#element_list = ['Ba', 'Ca', 'Mg', 'Sr']
##element_list = ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
##                'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Br', 'Kr', 'Rb', 'Sr', 'Y',
##                'Zr', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
##                'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
##                'Au', 'Hg', 'Tl', 'Pb', 'Bi'] # LIST does not include H, He, Tc, Pm, Se, Nb, Ho or heavier than Bi
#
#results_rmse = pd.DataFrame()
#
#for element in element_list:
#    try:
#        plasma1 = multi_element_plasma([element]) # make plasma object
#        
#        adjusted_plasma1 = adjusted_spectrum(plasma1.binned_spectrum()) # adjust the produced spectra
#        
#        test_ratio_spectra = adjusted_plasma1.ratio_spectra()
#        
#        test_full_spectra = adjusted_plasma1.full_scale_spectra()
#        
#        test_ground_truth = adjusted_plasma1.ground_truth()
#        
#        determined_mse = mean_squared_error(test_ground_truth, test_ratio_spectra)
#        
#        determined_rmse = np.sqrt(determined_mse)
#        
#        temp_rmse = pd.DataFrame([determined_rmse], index = [element])
#        
#        results_rmse = pd.concat([results_rmse, temp_rmse], axis = 0)
#    except:
#        pass


test_experimental_spectrum = pd.read_csv("IARM Ni C276-18_J200_1 L min Ar_20 percent energy_1 us delay_1 accumulation.txt", header=0, names = ['intensity'], delimiter = '\t')


index_peaks, _ = signal.find_peaks(test_experimental_spectrum['intensity'], height = 800, distance = 3, threshold = 5, prominence = 1000)

#plt.plot(test_experimental_spectrum.iloc[index_peaks].index, test_experimental_spectrum.iloc[index_peaks], "x")
#plt.plot(test_experimental_spectrum)

test_index = pd.DataFrame(test_experimental_spectrum.iloc[index_peaks].index)


experimental_peaks = pd.DataFrame(test_experimental_spectrum.iloc[index_peaks], index = test_index[0])
experimental_peaks.index.name = 'wavelength_nm'
experimental_peaks.index = experimental_peaks.index.to_series().apply(lambda x: np.round(x, 0))

#peaks, _ = signal.find_peaks(test_full_spectra['intensity'], height = 1000)
#
#plt.plot(test_full_spectra.iloc[peaks].index, test_full_spectra.iloc[peaks], "x")
#plt.plot(test_full_spectra)



class adjusted_experimental_spectrum(object):
    
    def __init__ (self, spectrum_to_adjust, resolution = 0):
        
        self.spectrum_to_adjust = spectrum_to_adjust
        self.resolution = resolution
        self.wavelength_min = 200.
        self.wavelength_max = 1000.
        self.wavelength_partition = 1 + ((self.wavelength_max - self.wavelength_min) * (10 ** resolution))
#        self.wavelength_partition = 80001

    def gaussian(self, current_wavelength):
        ''' Define a gaussian efficiency function '''
        x = current_wavelength
        x0 = 600
        sigma = 250
        gaussian_value = np.exp(-np.power((x - x0)/sigma, 2.)/2.)
        
        return gaussian_value

    def spectrum_adjust(self):
        ''' Adjust the meausred intensities by a detection efficiency function '''
        adjusted_spectrum = pd.DataFrame()
        for current_index, intensity in enumerate(self.spectrum_to_adjust):
            self.current_wavelength = self.spectrum_to_adjust.index[current_index]
            temp_val1 = intensity * self.gaussian(self.current_wavelength)
            temp2 = pd.DataFrame(temp_val1, columns = ['adjusted_intensity'], index = [self.current_wavelength])
            adjusted_spectrum = adjusted_spectrum.append(temp2)        
        return adjusted_spectrum

    def full_scale_spectra(self):
        theoretical_lines_binned = pd.DataFrame(self.spectrum_to_adjust)
        
#        adjusted_lines_binned = self.spectrum_adjust()
        
        combined_spectra = pd.concat([theoretical_lines_binned], axis =1)
        
        index_for_full_scale_spectra = pd.DataFrame(np.linspace(self.wavelength_min, self.wavelength_max, self.wavelength_partition, endpoint = True)).round(self.resolution)
        
        full_scale_spectra = pd.DataFrame(index = index_for_full_scale_spectra[0])
        
        full_scale_spectra = pd.concat([full_scale_spectra, combined_spectra], axis = 1)
        
        full_scale_spectra = full_scale_spectra.fillna(0)
        
        return full_scale_spectra


def bin_experimental_spectrum(spectrum_to_bin):
    
    limit_intensity = 1000
    temp1 = spectrum_to_bin
    temp1 = temp1.round({'wavelength_nm': 0})
    temp1 = temp1.groupby('wavelength_nm')['intensity'].sum()
    temp1 = temp1[(temp1.index >= 200) & (temp1.index <= 1000)]
    temp1 = temp1.where(temp1 >= limit_intensity)
    
    return temp1


binned_experimental_spectrum = bin_experimental_spectrum(experimental_peaks)

experimental_spectrum_1 = adjusted_experimental_spectrum(binned_experimental_spectrum)
experimental_spectrum_1_binned = experimental_spectrum_1.full_scale_spectra()

def get_ratio_measured_to_theoretical(spectrum_measured, spectrum_theoretical):
    
    
    temp1 = spectrum_measured['intensity'] / spectrum_theoretical['intensity']
    index_for_full_scale_spectra = pd.DataFrame(np.linspace(200, 1000, 801, endpoint = True)).round(0)
    ratio_measured_to_theoretical = pd.DataFrame(index = index_for_full_scale_spectra[0])
    
    ratio_measured_to_theoretical = pd.concat([ratio_measured_to_theoretical, temp1], axis = 1)
    ratio_measured_to_theoretical = ratio_measured_to_theoretical.replace([np.inf, -np.inf, 0], np.nan)
#    ratio_measured_to_theoretical = ratio_measured_to_theoretical.fillna(0)
    ratio_measured_to_theoretical = ratio_measured_to_theoretical.interpolate(method = 'linear', axis = 0, limit_direction = 'both')
    
    return ratio_measured_to_theoretical



calculated_efficiency_curve = get_ratio_measured_to_theoretical(experimental_spectrum_1_binned, test_full_spectra)
calculated_efficiency_curve = calculated_efficiency_curve / calculated_efficiency_curve.max()


plt.figure()
plt.plot(calculated_efficiency_curve)

plt.figure()
plt.plot(experimental_spectrum_1_binned)
plt.plot(test_full_spectra['intensity'])

