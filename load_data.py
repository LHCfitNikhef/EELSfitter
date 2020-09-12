###############################################
import numpy as np
import random
from numpy import loadtxt
import math
import sklearn
from scipy import integrate
import pandas as pd


###### Load spectra ###########################

fname = 'data/Specimen_4/'
    
ZLP_y14 = np.loadtxt(fname + "(14)_m4d054eV_45d471eV.txt")
ZLP_y15 = np.loadtxt(fname + "(15)_m4d054eV_45d471eV.txt")
ZLP_y16 = np.loadtxt(fname + "(16)_m4d054eV_45d471eV.txt")
ZLP_y17 = np.loadtxt(fname + "(17)_Vacuum_m4d054eV_45d471eV.txt")
ZLP_y19 = np.loadtxt(fname + "(19)_m4d054eV_45d471eV.txt")
ZLP_y20 = np.loadtxt(fname + "(20)_m4d054eV_45d471eV.txt")
ZLP_y21 = np.loadtxt(fname + "(21)_m4d054eV_45d471eV.txt")
ZLP_y22 = np.loadtxt(fname + "(22)_Vacuum_m4d054eV_45d471eV.txt")
ZLP_y23 = np.loadtxt(fname + "(23)_Vacuum_m4d054eV_45d471eV.txt")

###############################################

ndat=int(len(ZLP_y14))

# Energy loss values
ZLP_x14 = np.zeros(ndat)
Eloss_min = -4.054 # eV
Eloss_max = +45.471 # eV
i=0
while(i<ndat):
    ZLP_x14[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat) 
    i = i + 1

ZLP_x15, ZLP_x16, ZLP_x17, ZLP_x19, ZLP_x20, ZLP_x21, ZLP_x22, ZLP_x23 = \
    ZLP_x14,  ZLP_x14,  ZLP_x14,  ZLP_x14,  ZLP_x14, ZLP_x14,  ZLP_x14, ZLP_x14

file14 = pd.DataFrame({"x":ZLP_x14, "y":ZLP_y14})
file15 = pd.DataFrame({"x":ZLP_x15, "y":ZLP_y15})
file16 = pd.DataFrame({"x":ZLP_x16, "y":ZLP_y16})
file17 = pd.DataFrame({"x":ZLP_x17, "y":ZLP_y17})
file19 = pd.DataFrame({"x":ZLP_x19, "y":ZLP_y19})
file20 = pd.DataFrame({"x":ZLP_x20, "y":ZLP_y20})
file21 = pd.DataFrame({"x":ZLP_x21, "y":ZLP_y21})
file22 = pd.DataFrame({"x":ZLP_x22, "y":ZLP_y22})
file23 = pd.DataFrame({"x":ZLP_x23, "y":ZLP_y23})


################## Shift spectra to have peak position at dE = 0  ##################

for i, file in enumerate([file14, file15, file16, file17, file19, file20, file21, file22, file23]):
    zeropoint = file[file['y'] == file['y'].max()]['x']
    file['x_shifted'] = file['x'] - float(zeropoint)
    x = file['x_shifted']
    y = file['y']
    y_int = integrate.cumtrapz(y, x, initial=0)
    normalization = y_int[-1]
    file['y_norm'] = file['y'] / float(normalization)
    
    
    
##############  Put all datafiles into one DataFrame  ##############################

df = pd.concat((file14, file15, file16, file19, file20, file21))
df = df.sort_values('x').reset_index().drop('index', axis=1)

df_vacuum = pd.concat((file17, file22, file23))
df_vacuum = df_vacuum.sort_values('x').reset_index().drop('index', axis=1).dropna()

df['log_y'] = np.log(df['y'])
#df_vacuum['log_y'] = np.log(df_vacuum['y'])



################ Use [x_shifted, y_norm] values as training inputs #################


x14, y14 = file14['x_shifted'], file14['y_norm']
x15, y15 = file15['x_shifted'], file15['y_norm']
x16, y16 = file16['x_shifted'], file16['y_norm']
x17, y17 = file17['x_shifted'], file17['y_norm']
x19, y19 = file19['x_shifted'], file19['y_norm']
x20, y20 = file20['x_shifted'], file20['y_norm']
x21, y21 = file21['x_shifted'], file21['y_norm']
x22, y22 = file22['x_shifted'], file22['y_norm']
x23, y23 = file23['x_shifted'], file23['y_norm']



print('Files have been created \n')

print('\n Sample files:')
for i in ([14,15,16,19,20,21]):
    print('file' + str(i))
print('\n Vacuum files:')
for i in ([17,22,23]):
    print('file' + str(i))
    
    
print('\n Total samples file: "df" \n', df.describe())
print('\n Total vacuum file: "df_vacuum" \n', df_vacuum.describe())



