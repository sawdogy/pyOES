#Need to add modules to import

import periodictable as pt #may not need this if everything if own database is given


'''
Functions
'''



def get_database_data(file_name=''):
  
  if not os.path.exists(file_name):
     raise IOError("File {} does not exist!".format(file_name))
  df = pd.read_csv(file_name, headers=1)
  return df

def get_lowerdegen(database_file_name=''):
  """retrieve the degeneracy value for the lower energy level of a transition for specific species
  """
  
  if file_extension != '.csv':
    raise IOError("File type must be '.csv'")
  else:
    _df = get_database_data(file_name=database_file_name)
    



def get_upperdegen

def get_lowerenergy

def get_upperenergy

def get_radiativelifetime

def get_ionizationenergy

def calculate_partitionfunc(species, temperature, elecdens,):
  """ Calculates the partition function of element
  
  """
  
  partitionfunc = upperdegen * npexp(- upperenergy / temperature) #will need to make this a summming function
  
  return partitionfunc

def calculate_lineintensity (species, temperature, elecdens,):
  """ Calculates the intensity of an emission line
  
  """
  
  lineintensity = speciesconc * radiativelifetime * (upperdegen / partitionfunc) * np.exp(- upperenergy / temperature)
  
  return np.array(lineintensity)
