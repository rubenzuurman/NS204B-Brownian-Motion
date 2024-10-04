# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:09:51 2024

@author: Quint
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:41:45 2024

@author: Quint
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr
import scipy
import os

file = os.path.dirname(__file__)
os.chdir(file)

Theo = 0.2204 #theoritsche waarde voor rc
Theosig= 0.00843 #afwijking van theo door particle size deviatie
import pickle

def Diffusionspitter(time, particle_distancement,naam): #functie die een diffusie coeffecient geeft voor set aan particle positions
    time = time[0:100]
    particle_positions = np.array(particle_distancement).T[0:100]

    R_2 = []
    Rs_2 = []
    for momentje in particle_positions:
        R_2.append(np.nanmean(momentje))
        Rs_2.append(np.nanstd(momentje))

    x= time
    y= R_2
    y_err = Rs_2

    def f(B, x): # ODR fit
        return B[0]*x
    A_start=4

    odr_model = odr.Model(f) 
    odr_data  = odr.RealData(x, y, sy=y_err) 
    odr_obj   = odr.ODR(odr_data,odr_model,beta0=[A_start]) 
    odr_res   = odr_obj.run() 
    # odr_res.pprint()
    par_best = odr_res.beta 
    
    # odr_obj.set_job(fit_type=0) # type fit algoritme

    xplot = np.linspace(0, 10, num=20)
    plt.errorbar(x, y, yerr=Rs_2, fmt= 'k.', elinewidth= 0)
    # plt.errorbar(x, y, yerr=Rs_2, fmt= 'none', ecolor= 'blue', alpha = 0.4)
    plt.fill_between(x, np.array(y) - np.array(Rs_2), np.array(y) + np.array(Rs_2), color='blue', alpha=0.5, label='±1 std. dev.')
    plt.plot(xplot,f(par_best, xplot), color= 'red')
    plt.ylabel('Mean squared displacemnt \n '  r'$\langle R^2 \rangle$ $ ( \mu m^2 )$' ) 
    plt.xlabel(r'Time since particle detection $t$ (s)') 
    particlelen.append(len(particle_distances))
    plt.xlim(0,10)
    plt.ylim(0,30)
    #OPTIE beschrijving figuren
    # plt.figtext(0.12, -0.05, 'Figure 1: Shows the average distance of particles to the startingpoint \n and the traces of individual particles of sample: ' + naam[17:23] + naam[-22:-21] + naam[-20:-19] )
    # plt.figtext(0.12, -0.05, 'Sample: ' + naam[17:23] + naam[-22:-21] + naam[-20:-19] )
    #OPTIE laten zien van individuelen traces
    # for particle in particle_distances: #individuelen traces
    #     plt.plot(time[0:100], particle[0:100])
    plt.tight_layout()
    plt.savefig( naam[:-6] + 'losse.png') #opslaan figuren
    plt.show()

    return(odr_res.beta/(4), odr_res.res_var/(4))



particlelen = [] # hoe veel particles er per sample zijn
Namen_D = [] # de namen die ingelezen woorden voor de verschillende recordings
Lijst_waarden_D = [] #Lijst met de 45 waarden voor de diffusie coefficient
for naam in ["005_mass_percent_500nm_40x_A_1_output_data.pickle", "005_mass_percent_500nm_40x_A_2_output_data.pickle", "005_mass_percent_500nm_40x_A_3_output_data.pickle", "005_mass_percent_500nm_40x_B_1_output_data.pickle", "005_mass_percent_500nm_40x_B_2_output_data.pickle", "005_mass_percent_500nm_40x_B_3_output_data.pickle", "005_mass_percent_500nm_40x_C_1_output_data.pickle","005_mass_percent_500nm_40x_C_2_output_data.pickle","005_mass_percent_500nm_40x_C_3_output_data.pickle"    ,   "005_mass_percent_750nm_40x_A_1_output_data.pickle", "005_mass_percent_750nm_40x_A_2_output_data.pickle", "005_mass_percent_750nm_40x_A_3_output_data.pickle", "005_mass_percent_750nm_40x_B_1_output_data.pickle", "005_mass_percent_750nm_40x_B_2_output_data.pickle", "005_mass_percent_750nm_40x_B_3_output_data.pickle", "005_mass_percent_750nm_40x_C_1_output_data.pickle","005_mass_percent_750nm_40x_C_2_output_data.pickle","005_mass_percent_750nm_40x_C_3_output_data.pickle"     ,     "005_mass_percent_1000nm_40x_A_1_output_data.pickle", "005_mass_percent_1000nm_40x_A_2_output_data.pickle", "005_mass_percent_1000nm_40x_A_3_output_data.pickle", "005_mass_percent_1000nm_40x_B_1_output_data.pickle", "005_mass_percent_1000nm_40x_B_2_output_data.pickle", "005_mass_percent_1000nm_40x_B_3_output_data.pickle", "005_mass_percent_1000nm_40x_C_1_output_data.pickle","005_mass_percent_1000nm_40x_C_2_output_data.pickle","005_mass_percent_1000nm_40x_C_3_output_data.pickle"    ,     "005_mass_percent_1500nm_40x_A_1_output_data.pickle", "005_mass_percent_1500nm_40x_A_2_output_data.pickle", "005_mass_percent_1500nm_40x_A_3_output_data.pickle", "005_mass_percent_1500nm_40x_B_1_output_data.pickle", "005_mass_percent_1500nm_40x_B_2_output_data.pickle", "005_mass_percent_1500nm_40x_B_3_output_data.pickle", "005_mass_percent_1500nm_40x_C_1_output_data.pickle","005_mass_percent_1500nm_40x_C_2_output_data.pickle","005_mass_percent_1500nm_40x_C_3_output_data.pickle"     ,     "005_mass_percent_2000nm_40x_A_1_output_data.pickle", "005_mass_percent_2000nm_40x_A_2_output_data.pickle", "005_mass_percent_2000nm_40x_A_3_output_data.pickle", "005_mass_percent_2000nm_40x_B_1_output_data.pickle", "005_mass_percent_2000nm_40x_B_2_output_data.pickle", "005_mass_percent_2000nm_40x_B_3_output_data.pickle", "005_mass_percent_2000nm_40x_C_1_output_data.pickle","005_mass_percent_2000nm_40x_C_2_output_data.pickle","005_mass_percent_2000nm_40x_C_3_output_data.pickle"     ]:
    with open( naam , "rb") as file:
        data = pickle.load(file)
    # print("Available dict keys from pickle:", data.keys())
    time = data["time"]
    particle_distances = data["particle_distances"]
    Lijst_waarden_D.append(Diffusionspitter(time, particle_distances,naam))
    Namen_D.append(naam[17:23] + naam[-22:-21] + naam[-20:-19])

#%% Voor plot met 5 gemmidelde lijntjes    
R_2_L= []
for naam in ["005_mass_percent_500nm_40x_A_1_output_data.pickle", "005_mass_percent_500nm_40x_A_2_output_data.pickle", "005_mass_percent_500nm_40x_A_3_output_data.pickle", "005_mass_percent_500nm_40x_B_1_output_data.pickle", "005_mass_percent_500nm_40x_B_2_output_data.pickle", "005_mass_percent_500nm_40x_B_3_output_data.pickle", "005_mass_percent_500nm_40x_C_1_output_data.pickle","005_mass_percent_500nm_40x_C_2_output_data.pickle","005_mass_percent_500nm_40x_C_3_output_data.pickle"    ,   "005_mass_percent_750nm_40x_A_1_output_data.pickle", "005_mass_percent_750nm_40x_A_2_output_data.pickle", "005_mass_percent_750nm_40x_A_3_output_data.pickle", "005_mass_percent_750nm_40x_B_1_output_data.pickle", "005_mass_percent_750nm_40x_B_2_output_data.pickle", "005_mass_percent_750nm_40x_B_3_output_data.pickle", "005_mass_percent_750nm_40x_C_1_output_data.pickle","005_mass_percent_750nm_40x_C_2_output_data.pickle","005_mass_percent_750nm_40x_C_3_output_data.pickle"     ,     "005_mass_percent_1000nm_40x_A_1_output_data.pickle", "005_mass_percent_1000nm_40x_A_2_output_data.pickle", "005_mass_percent_1000nm_40x_A_3_output_data.pickle", "005_mass_percent_1000nm_40x_B_1_output_data.pickle", "005_mass_percent_1000nm_40x_B_2_output_data.pickle", "005_mass_percent_1000nm_40x_B_3_output_data.pickle", "005_mass_percent_1000nm_40x_C_1_output_data.pickle","005_mass_percent_1000nm_40x_C_2_output_data.pickle","005_mass_percent_1000nm_40x_C_3_output_data.pickle"    ,     "005_mass_percent_1500nm_40x_A_1_output_data.pickle", "005_mass_percent_1500nm_40x_A_2_output_data.pickle", "005_mass_percent_1500nm_40x_A_3_output_data.pickle", "005_mass_percent_1500nm_40x_B_1_output_data.pickle", "005_mass_percent_1500nm_40x_B_2_output_data.pickle", "005_mass_percent_1500nm_40x_B_3_output_data.pickle", "005_mass_percent_1500nm_40x_C_1_output_data.pickle","005_mass_percent_1500nm_40x_C_2_output_data.pickle","005_mass_percent_1500nm_40x_C_3_output_data.pickle"     ,     "005_mass_percent_2000nm_40x_A_1_output_data.pickle", "005_mass_percent_2000nm_40x_A_2_output_data.pickle", "005_mass_percent_2000nm_40x_A_3_output_data.pickle", "005_mass_percent_2000nm_40x_B_1_output_data.pickle", "005_mass_percent_2000nm_40x_B_2_output_data.pickle", "005_mass_percent_2000nm_40x_B_3_output_data.pickle", "005_mass_percent_2000nm_40x_C_1_output_data.pickle","005_mass_percent_2000nm_40x_C_2_output_data.pickle","005_mass_percent_2000nm_40x_C_3_output_data.pickle"     ]:
    with open( naam , "rb") as file:
        data = pickle.load(file)
    particle_distances = np.array(data["particle_distances"]).T
    time = data["time"]
    R_2 = []
    for momentje in particle_distances[0:100]:
        R_2.append(np.nanmean(momentje))
    R_2_L.append(R_2)
zesdelijst = []
for L in [0,8,17,26,35]:
    averaged_list = [sum(values) / len(values) for values in zip(*R_2_L[L:L+9])]
    zesdelijst.append(averaged_list)
groottes = ['500nm', '750nm', '1000nm', '1500nm', '2000nm']    
for i in range(len(zesdelijst)):
    plt.errorbar(time[0:len(zesdelijst[i])],zesdelijst[i], fmt='.', label = groottes[i])      
plt.ylim(0,10) 
plt.xlim(0) 

plt.ylabel('Mean squared displacemnt \n '  r'$\langle R^2 \rangle$ $ ( \mu m^2 )$', size= 20) 
plt.xlabel(r'Time since particle detection $t$ (s)', size= 20)
#OPTIE bijschrift
# plt.figtext(0.12, -0.05,  'Average squared distance of all the samples per size', size = 20) 
plt.legend(fontsize = 16)  
plt.tight_layout()
plt.show()  
#%% Finale plotje van Diffusie vs Size
xas= []
for grootte in np.array([0.51,0.746,0.99,1.5,1.93])/2:
    for i in range(9):
        xas.append(grootte)
error= [np.array(beta[1]) for beta in Lijst_waarden_D]
Diffusie = np.array([beta[0] for beta in Lijst_waarden_D]).flatten()
plt.errorbar(xas , Diffusie ,yerr=error,xerr=0, fmt='g.', ecolor='blue', alpha= 0.4)
plt.xlim(0)
plt.ylim(0)
plt.ylabel('Diffusion coefficient \n $D $ $(\mu m^2 s^{-1} $)')
plt.xlabel('Particle radius \n $a $ $( \mu m )$')

def f(B, x): #ODR fit
    print(B)
    return B[0]/x 
A_start=4

odr_model = odr.Model(f) 
odr_data  = odr.RealData(xas[1:], Diffusie[1:], sy=error[1:]) 
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[A_start]) 
odr_res   = odr_obj.run() 
odr_res.pprint()
par_best = odr_res.beta 
# for i, label in enumerate(Namen_D): # naamgeven van alle punten 
    # plt.annotate(label, (xas[i], Diffusie[i]), textcoords="offset points", xytext=(0,10), ha='center')
   
plt.plot(np.linspace(0, 1,100),f(par_best, np.linspace(0, 1,100)), color= 'navy')
plt.plot(np.linspace(0, 1,100),f([Theo],(np.linspace(0, 1,100))), color = 'red')
plt.plot(0.255,0.027, marker='x', color = 'red') 
plt.annotate('500nm A_1', (xas[0],Diffusie[0]), textcoords="offset points", xytext=(0,10), ha='center') #annotatie 500nm A 1
plt.figtext(0,0, '')

# berekening onzerheid 
V_500_grootte = 0.01**2/np.average(particlelen[0:8]) #deviation given by manufacturer/average amount of data points
V_750_grootte = 0.022**2/np.average(particlelen[9:17])
V_1000_grootte = 0.03**2/np.average(particlelen[18:26])
V_1500_grootte = 0.04**2/np.average(particlelen[27:35]) #schatting onbekend
V_2000_grootte = 0.054**2/np.average(particlelen[36:44])

V_beweging = ((50/(268+5) - 50/(268-5))/2)**2 + (3*50/(268))**2 # callibratie varaiantie + variantie trackpy in geven middelpunt particle

V_500 =  (V_beweging  + V_500_grootte)/100 #devided by amount of frames
V_750 =  (V_beweging  + V_750_grootte)/100
V_1000 = (V_beweging  + V_1000_grootte)/100
V_1500 = (V_beweging  + V_1500_grootte)/100
V_2000 = (V_beweging  + V_2000_grootte)/100


S_totaal = np.sqrt( V_500/8 + V_750/9 + V_1000/9 + V_1500/9 + V_2000/9) #devision by amount of recordings


plt.fill_between(np.linspace(0, 1,100),f(par_best - 2*S_totaal, np.linspace(0, 1,100)), f((par_best + 2*S_totaal), np.linspace(0, 1,100)), color ='cyan', alpha = 0.5)
plt.fill_between(np.linspace(0, 1,100),(Theo - 2*Theosig)/np.linspace(0, 1,100) , (Theo + 2*Theosig)/np.linspace(0, 1,100), color ='orange', alpha = 0.3)

plt.show()
#%% 
