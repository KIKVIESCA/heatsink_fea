"""
0.1.0 First working prototype.
0.1.1 Thermal scale. Power img. Vertical.
0.1.2 Temperature integral.
0.1.3 Forced convection.
0.1.4 Reading of palettes.
0.2.0 Materials.
0.3.0 Emissivity and package.
"""

import json
import tkinter as tk
import tkinter.filedialog as tk_filedialog
import tkinter.ttk as ttk
from datetime import datetime
from os import startfile
from pathlib import Path
from tkinter.messagebox import askyesno, showerror, showinfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk


__author__ = "Kostadin Kostadinov"
__credits__ = ["Kostadin Kostadinov"]
__license__ = "TBD"
__version__ = "0.3.0"
__maintainer__ = "Kostadin Kostadinov"
__status__ = "Alpha"

WORSPACE_DIR = Path('./')/'simulation'#Path.home() / "Desktop"
WORSPACE_DIR.mkdir(exist_ok=True)
LAST_DESIGN = WORSPACE_DIR / "test.json"
COLOR_PATH = Path('./color_palette.json')

simulation_data = {}
MAXIMUM_ITERATIONS = 99
SIMULATION_SECONDS = 1
print_help = False


# mm, K/W
DATABASE_PATH = Path('./database.json')

# user configuration
USER_PATH = Path('./.config.json')
if USER_PATH.is_file():
    with open(USER_PATH, 'r') as reader:
        user_config = json.load(reader)
else:
    user_config = {'recent':{}}

def fn_wrp(func, *args, **kwargs):
    """ Function wrapper with arguments
    Return:
        reference to func(*args, **kwargs)
    """
    def func0():
        return func(*args, **kwargs)
    return func0

def reset_main():
    global main_frame
    main_frame.pack_forget()
    main_frame = ttk.Frame(tk_mgr)
    main_frame.pack(fill=tk.BOTH, expand=1)

class Database:
    def __init__(self, db_path):
        with open(db_path, 'r') as reader:
            d = json.load(reader)
            self.heatsinks = d["heatsink"]
            self.materials =  d["material"]
            self.finish = d["finish"]
            self.packages = d["package"]
        self.e_list = []

    def list_heatsink(self):
        return [f'{k}:{self.heatsinks[k]["description"]}' for k in sorted(self.heatsinks)]

    def list_material(self):
        return [f'{k}:{self.materials[k]["description"]}' for k in sorted(self.materials)]
    
    def list_surface_treatment(self):
        return list(self.finish)
    
    def list_package(self):
        return list(self.packages)

    def get_package(self, k):
        return self.packages[k]
    
    def heatsink_info(self, hs_key):
        if hs_key in self.heatsinks:
            return self.heatsinks[hs_key]
        return {}
    
    def material_info(self, mat_key):
        if mat_key in self.materials:
            return self.materials[mat_key]
        return {}
    
    def scaled_conductivity(self, mat_key, telem_array=25):
        """ Thermal conductivity, in W/m.K"""
        kth_array = np.array(self.materials[mat_key]['thermal_conductivity'])
        return np.interp(telem_array, kth_array[:,0], kth_array[:,1])
    
    def select_heatsink(self, hs_key, baseplate_width, flow_conditions, air_velocity=None):
        self.hs_key = hs_key
        # check input
        if hs_key not in self.heatsinks:
            self.e_list.append(f'Heastink {hs_key} not found in database.')
            return False
        if flow_conditions not in self.heatsinks[hs_key]:
            self.e_list.append(f'No {flow_conditions} data available for heatsink {hs_key}.')
            return False
        if flow_conditions == 'forced_convection' and air_velocity not in self.heatsinks[hs_key][flow_conditions]['thermal_resistance']:
            self.e_list.append(f'Heatsink {hs_key}: nNo data for {air_velocity} m/s in database.')
            return False
        # datasheet temperature difference
        self.exp_dt = self.heatsinks[self.hs_key][flow_conditions]['steady_temperature']-self.heatsinks[hs_key][flow_conditions]['ambient_temperature']
        # lookup convection
        """ Read convection data from database.
        Returns list of z coordinates (in mm) and average admittance (in W/K) to that z"""

        if flow_conditions == 'forced_convection':
            convection_array = np.array(self.heatsinks[hs_key][flow_conditions]['thermal_resistance'][air_velocity])
        elif flow_conditions == 'natural_convection':
            # pull convection data from database. admittance on 0 is 0
            convection_array = np.array(self.heatsinks[hs_key][flow_conditions]['thermal_resistance'])
        else:
            assert True
        self.convection_data = convection_array[convection_array[:, 0].argsort()]
        self.convection_data[:,1] *= self.heatsinks[hs_key]['baseplate_width']/baseplate_width*self.convection_data[:,0]**.5 # Rth*z**.5 
        return True
    
    def get_baseplate_thickness(self):
        """ Get baseplate thickness, in SI"""
        return self.heatsinks[self.hs_key]['baseplate_thickness']*1E-3
    
    def get_z_factor(self, dz):
        """ Integration pending
        dz
        theta_list = np.linspace(0,np.atan(dz/(fin_height+baseplate_thickness)),100)
        z_factor = 1+np.sum(np.sin(theta_list)/np.cos(theta_list)**2)*theta_list[1]/100*fin_width/(fin_width+fin_space)
        """
        d = self.heatsinks[self.hs_key]
        x_list = np.linspace(0,d['fin_height'],100)+d['baseplate_thickness']
        dx = x_list[1]-x_list[0]
        fin_factor = dx*np.sum((x_list**2+dz**2)**-.5)
        return (1+fin_factor*d['fin_width']/(d['fin_width']+d['fin_space']))

    def scaled_convection(self, z_list):
        """Estimate vertical convection from 0 to 1st z, 1st to 2nd, etc.
        admittance is in W/K, corresponds to dz*simulation_baseplate_width
        Z ascending list contains vertical coordinates
        """
        # interpolate for z: Rth*z**.5 = constant
        Rthr_list = np.interp(z_list, self.convection_data[:,0], self.convection_data[:,1])
        Yth_list = Rthr_list**-1*z_list**.5
        if type(Yth_list) is not np.ndarray:
            return Yth_list
        # differental
        Yth_list = np.insert(Yth_list, 0, 0)
        return Yth_list[1:]-Yth_list[:-1]
    
def plot_array(telm_array, color_array, file_path="loop.png", down2ambient=False):
    """ Plot thermal distribution
    """
    #array_shape = np.shape(telm_array) # number of rows, number of columns
    # create new image
    data_range = (simulation_data["ambient_temperature"] if down2ambient else np.min(telm_array), np.max(telm_array))
    ra,ga,ba = color_interp(telm_array, color_array, data_range)
    # scale colors to palette color
    thermal_img = Image.fromarray(np.stack((ra,ga,ba), axis=-1), mode='RGB')
    # resize to nearest
    img_size = np.array(
        (simulation_data["baseplate_width"], simulation_data["baseplate_height"])
    )
    thermal_img = thermal_img.resize(
        img_size.tolist(),
        resample=Image.Resampling.NEAREST,
    )
    # draw sources
    draw = ImageDraw.Draw(thermal_img)
    source_dict = simulation_data["source_layout"]
    for k in source_dict:
        heat_dict = source_dict[k]
        rect_data = np.array(heat_dict["rect"])
        yz_bl_h = rect_data[:2]
        yz_tr_h = yz_bl_h + rect_data[2:]
        draw.rectangle(
            yz_bl_h.tolist() + yz_tr_h.tolist(),
            outline=tuple(color_array[0]),
            width=1,
        )
    thermal_img = thermal_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    # FULL IMAGE
    full_size = img_size+ (100,100)
    full_image = Image.new(mode='RGB', size=full_size.tolist(), color="white")
    base_image = Image.new(mode='RGB', size=img_size.tolist(), color='gray')
    t = int(DBH.heatsinks[simulation_data["heatsink"]]["baseplate_thickness"]/2**.5)
    full_image.paste(base_image,  (t,100-t))
    full_image.paste(thermal_img, (0,100))
    draw = ImageDraw.Draw(full_image)
    # fins
    dx = int(DBH.heatsinks[simulation_data["heatsink"]]["fin_height"]/2**.5)
    dy = 10
    for i in range(int(img_size[0]/dy)):
        yz0 = np.array((dy*(i+.5)+t,100-t), dtype=int)
        yz1 = yz0+(dx,-dx)
        draw.line([tuple(yz.tolist()) for yz in (yz0, yz1)], fill="grey", width=3)
    yz0 = (img_size *(1,0)+(t,100-t)).astype(int)
    yz1 = yz0+(dx,-dx)
    yz2 = yz1+img_size*(0,1)
    yz3 = yz2-(dx,-dx)
    draw.line([tuple(yz.tolist()) for yz in (yz0, yz1, yz2, yz3)], fill="grey", width=1)
    # arrow
    yz0 = (img_size *(0.5,0)+(0,20)).astype(int)
    yz1 = yz0+(0,60)
    draw.line([tuple(yz.tolist()) for yz in (yz0, yz1)], fill="black", width=10)
    yz1 = yz0+(20,20)
    yz2 = yz0+(-20,20)
    draw.line([tuple(yz.tolist()) for yz in (yz2, yz0, yz1)], fill="black", width=10, joint='curve')
    # temperature scale
    fnt = ImageFont.load_default()
    if False:
        draw.text(tuple((img_size *(1,0)+(0,50)).astype(int)), file_path.name, font=fnt, fill="black")
    palette_array = np.repeat(np.arange(256),2).reshape(256,2)
    ra,ga,ba = color_interp(palette_array, color_array)
    palette_img = Image.fromarray(np.stack((ra,ga,ba), axis=-1), mode='RGB')
    palette_img = palette_img.resize((20, 256)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    full_image.paste(palette_img, (full_size[0]-30, full_size[1]-266))
    dy_list = ((20,0),(40,0))
    # minimum
    t0 = int(data_range[0])
    yz0 = full_size-(60,10)
    draw.line([tuple((yz0+dy).tolist()) for dy in dy_list], fill="black", width=1)
    draw.text(yz0.tolist(), f'{t0}', font=fnt, fill="black")
    # maximum
    t0 = int(data_range[1])
    yz0 -= (0, 256)
    draw.line([tuple((yz0+dy).tolist()) for dy in dy_list], fill="black", width=1)
    draw.text(tuple(yz0), f'{t0}', font=fnt, fill="black")
    if down2ambient:
        # heatsink
        t0 = int(np.min(telm_array))
        yz0 += (0, int((data_range[1]-t0)/(data_range[1]-data_range[0])*256))
        draw.line([tuple((yz0+dy).tolist()) for dy in dy_list], fill="black", width=1)
        draw.text(tuple(yz0), f'{t0}', font=fnt, fill="black")
    # SAVE
    full_image.save(file_path)
    print(f"See figure {file_path}")
    return True

def run_simulation():
    # cartesian coordinate grid
    """y_list = np.arange(0, simulation_data["baseplate_width"])
    z_list = np.arange(0, simulation_data["baseplate_height"])
    y_val, z_val = np.meshgrid(y_list, z_list)"""
    global simulation_data
    # PULL DATA FROM DATABASE
    if not DBH.select_heatsink(simulation_data["heatsink"], simulation_data["baseplate_width"], simulation_data["flow_conditions"], simulation_data['air_velocity'] if 'air_velocity' in simulation_data else None):
        showerror('HEATSINK ERROR', '\n'.join(DBH.e_list))
        DBH.e_list = []
        return False
    thermal_conductivity = DBH.scaled_conductivity(simulation_data["material"])
    if thermal_conductivity is None:
        showerror('MATERIAL ERROR', '\n'.join(DBH.e_list))
        DBH.e_list = []
        return False
    kth_y = thermal_conductivity*DBH.get_baseplate_thickness()
    kth_x = thermal_conductivity/DBH.get_baseplate_thickness()
    eth = DBH.finish[simulation_data['finish']]

    # check heat sources
    heat_count = 0
    source_dict = simulation_data["source_layout"]
    for k in source_dict:
        heat_dict = source_dict[k]
        rect_data = np.array(heat_dict["rect"])
        yz_bl_h = rect_data[:2]
        yz_tr_h = yz_bl_h + rect_data[2:]
        if np.any(yz_bl_h < (0, 0)) or np.any(yz_tr_h > yz_tr_h):
            showerror("RANGE SOURCE", f"Source {k} out of range.")
            return False
        """
        for k2 in source_dict:
            if k2 != k:
                rect_data2 = np.array(source_dict[k2]["rect"])
                yz_bl_2 = rect_data2[:2]
                yz_tr_2 = yz_bl_2 + rect_data2[2:]"""
        heat_count += heat_dict["power"]
    if heat_count < 1:  # W
        showerror("LOW SOURCE", f"Source power too low")
        return False
    Yavg_convection = DBH.scaled_convection(simulation_data["baseplate_height"])
    print(f"Rth_cv={np.round(Yavg_convection**-1,4)} K/W")
    
    # loop, starting 2x2 matrix
    last_stamp = datetime.now()
    map_partition = 0
    mm_step = np.array(
        (simulation_data["baseplate_width"], simulation_data["baseplate_height"])
    ).astype(float)/2
    ryz_cnt = 2*np.ones(2) # 2 rows vertically, 2 colums horizontally
    tamb_scalar = simulation_data["ambient_temperature"]
    telm_list = tamb_scalar*np.ones((2,2)) + 50 # as stated in datasheet
    while map_partition < 20 and (datetime.now() - last_stamp).total_seconds() < SIMULATION_SECONDS:
        print(f"PARTITION #{map_partition} " + 60 * "#")
        # loop variables update and reshape
        last_stamp = datetime.now()
        #map_partition > 0:
        if mm_step[0] < mm_step[1]:
            # split vertically, upper and lower part
            # TEMPERATURE reshape : shape is 1 x matrix_size
            mm_step /= (1,2) # split cell height
            ryz_cnt *= (2,1) # 2 rows, 1 column
            ryz_cnt = ryz_cnt.astype(int)
            telm_list = np.repeat(telm_list,2, axis=0).reshape(ryz_cnt)
        else:
            # split horizontally, left and right part
            mm_step /= (2,1) # split cell width
            ryz_cnt *= (1,2) # 1 row, 2 columns
            ryz_cnt = ryz_cnt.astype(int)
            telm_list = np.repeat(telm_list,2).reshape(ryz_cnt)
        matrix_size = np.prod(ryz_cnt)
        integral_matrix=np.tril(np.ones((ryz_cnt[0],ryz_cnt[0]))) #np.transpose(np.triu(np.ones((ryz_cnt[0],ryz_cnt[0])))/(1+np.arange(ryz_cnt[0])))
        # horizontal index is column id, vertical identifier is row id
        ry, rz = np.meshgrid(np.arange(ryz_cnt[1]), np.arange(ryz_cnt[0]))
        # heat dissipation matrix
        y_grid = ry * mm_step[0]
        z_grid = rz * mm_step[1]
        #yz_list = np.transpose(np.vstack((y_rav, z_rav)))
        pyz_list = 0#np.zeros(matrix_size)
        for heat_dict in simulation_data["source_layout"].values():
            rect_data = np.array(heat_dict["rect"])
            heat_density = heat_dict["power"] / np.prod(rect_data[2:])  # W/mm2
            dy = np.minimum(y_grid+ mm_step[0], rect_data[0]+rect_data[2]) - np.maximum(y_grid, rect_data[0])
            dz = np.minimum(z_grid+ mm_step[1], rect_data[1]+rect_data[3]) - np.maximum(z_grid, rect_data[1])
            pyz_list += np.maximum(dy, 0)*np.maximum(dz, 0) * heat_density
        #print(f"Pyz={round(np.sum(pyz_list))} W")
        # CONVECTION ADMITTANCE LIST: shape is 1 x matrix_size W/K
        z_list = (np.arange(ryz_cnt[0]) + 1) * mm_step[1]
        conv_adm_column = DBH.scaled_convection(z_list)
        #if simulation_data["flow_conditions"] == 'forced_convection':
        Yth_conv_list = np.repeat(conv_adm_column, ryz_cnt[1])/ryz_cnt[1]
        Yth_conv_list = np.reshape(Yth_conv_list, ryz_cnt)
        # CONDUCTION singular matrix in W/K
        conduction_matrix = 0 #matrix_size x matrix_size
        #if map_partition > 0:
        # conduction with adjacent roll and next roll
        # kk for di in (1, ryz_cnt[0]):  # np.arange(1, matrix_size):
        eye_sum = 0
        kth_config = (
            (kth_y, 1),# horizontal y conduction
            (kth_y*DBH.get_z_factor(mm_step[1]), ryz_cnt[1]), # vertical z conduction
        )
        for kth_yz, di in  kth_config:#for roll_axis in (0, 1): #range(1,matrix_size):
            # roll to the next coordinate
            ry2 = ry-di%ryz_cnt[1] # horizontal conduction, move 1 position
            rz2 = rz-int(di/ryz_cnt[1]) # vertical conduction, move 1 ryz_cnt[1] positions
            boundary_condition = np.logical_and(ry2 >= 0, rz2 >= 0)
            if False:
                print(
                    f"Boundary sum is {np.sum(boundary_condition)}, should be {2**map_partition*(2**map_partition-1)}"
                )
            # print(f'Roll elements = {di}')
            dy = (ry - np.roll(ry, di)) * mm_step[0]
            dz = (rz - np.roll(rz, di)) * mm_step[1]
            # compute
            # dt * contact length (=np.cross product) / distance, values in K
            dyzr = (mm_step[1] * dy + mm_step[0] * dz) / (dy**2 + dz**2)
            dyzr = np.where(boundary_condition, dyzr, 0)
            if print_help:  # boudary condition verification: removes negative
                """dist_list = [np.min(dyzr), np.sum(dyzr), np.max(dyzr)]
                print(f"Relative distance is {dist_list}")
                dyzr = np.where(boundary_condition, dyzr, 0)
                dist_list = [np.min(dyzr), np.sum(dyzr), np.max(dyzr)]
                print(f"Relative distance is {dist_list}")
                """
                eye1 = (
                    np.eye(matrix_size, k=di)
                    * np.where(boundary_condition, 1, 0).ravel()
                )
                eye2 = np.transpose(eye1)
                eye_sum+= np.roll(eye1, -di) - eye1 + np.roll(eye2, di) - eye2
            # last di elements are excluded
            dyzr_eye = np.eye(matrix_size, k=di) * dyzr.ravel()# * dtr.ravel()
            # additional 10% vertical thermal conductance trough fins
            if False:#roll_axis == 1:
                dyzr_eye *= 1.1
            eye2 = np.transpose(dyzr_eye)
            # values in W/K
            conduction_matrix += (
                np.roll(dyzr_eye, -di) - dyzr_eye + np.roll(eye2, di) - eye2
            )*kth_yz
        if False:
            print("CONDUCTION TRANSFER EYE MATRIX")
            print(np.reshape(np.diag(eye_sum), ryz_cnt))
        # loop until error gets below 10C divided by all the elements
        simulation_count = 0
        dt_error = 100  #
        while dt_error > 2 and simulation_count < MAXIMUM_ITERATIONS:
            #print(f"LOOP {simulation_count} t_error={round(dt_error,6)}") # + 60 * "#"
            # CONVECTION TEMPERATURE CORRECTION
            if simulation_data["flow_conditions"] == 'natural_convection':
                dt_loop = (telm_list-tamb_scalar)/DBH.exp_dt
                Yth_conv_list = np.interp(integral_matrix@dt_loop, (1+np.arange(ryz_cnt[0])), conv_adm_column/ryz_cnt[1])
                #Ycv_scale_factor = Yavg_convection/np.sum(Yth_conv_list)
                #Yth_conv_list *= Ycv_scale_factor
                #print(f'Convection scale factor is {np.round(Ycv_scale_factor,3)}')
                #print(Yth_conv_list)
                print(f'Total convection resistance is {np.round(np.sum(Yth_conv_list)**-1,3)} K/W')
                # RADIATION ADMITTANCE LIST: shape is 1 x matrix_size W/K
                Yth_rad_list = (
                    eth
                    * 5.67e-8
                    * np.prod(mm_step)
                    * 1e-6
                    * ((telm_list + 273.15) ** 4 - (273.15 + tamb_scalar) ** 4)
                ) / (telm_list - tamb_scalar)
            else:
                Yth_rad_list = 0
            #print(f'Total radiation resistance is {np.round(np.sum(Yth_rad_list)**-1,3)} K/W')
            #print(Yth_rad_list)
            # CONVECTION + RADIATION admittance in W/K
            Rth0_list = (Yth_rad_list + Yth_conv_list)**-1+(kth_x*np.prod(mm_step)*1E-6)**-1
            admittance_matrix = np.eye(matrix_size) * (Rth0_list**-1).ravel()
                # CONVECTION + RADIATION admittance in W/K
            # RESISTANCE in K/W
            resistance_matrix = np.linalg.inv(admittance_matrix+conduction_matrix)
            tnew_list = tamb_scalar + (resistance_matrix @ (pyz_list.ravel())).reshape(ryz_cnt)
            # limit
            tnew_list = np.maximum(tamb_scalar + 1, tnew_list)
            dt_error = np.max(np.abs(tnew_list - telm_list))
            tnew_list = np.minimum(tamb_scalar + 199, tnew_list)
            # recalculate error
            simulation_count += 1
            #print(telm_list.astype(np.uint8))
            telm_list = np.round(tnew_list,2)
            if np.any(np.isnan(telm_list)):
                print(f"T max is {np.max(telm_list)}ºC")
                return False
        if print_help and (np.min(ryz_cnt)>1.5):
            print('POWER GENERATION MATRIX (W)')
            print(pyz_list.astype(int))
            print('POWER TROUGH CONVECTION (W)')
            print(Yth_conv_list.reshape(ryz_cnt)*(telm_list-tamb_scalar).astype(int))
            print('POWER TROUGH RADIATION (W)')
            print(Yth_rad_list.reshape(ryz_cnt)*(telm_list-tamb_scalar).astype(int))
            print('POWER TROUGH CONDUCTION (W)')
            print(np.diag(conduction_matrix).reshape(ryz_cnt)*(telm_list-tamb_scalar).astype(int))
            print('TEMPERATURE OF ELEMENTS')
            print(telm_list.astype(np.uint8))
        rth_avg = np.average(resistance_matrix)
        print(f"Rth loop is {round(rth_avg, 4)} K/W, Pyz loop = {round(np.sum(pyz_list))} W")
        print(f"T in {np.min(telm_list)}ºC ... {np.max(telm_list)}ºC")
        map_partition += 1
        # plot
        if False:#np.min(ryz_cnt)>=2:
            plot_path = Path('./gif') / f"loop{map_partition}.png"
            plot_array(telm_list,color_array,plot_path)
        if simulation_count + 1 > MAXIMUM_ITERATIONS:
            showerror('CONVERGING ERROR', f'MAXIMUM ITERATIONS REACH WITHOUT CONVERGING: dt_error={round(dt_error)} K, with shape {np.shape(telm_list)}')
            return False
    # power grid
    simulation_data['results'] = {}
    # for back surface: -kth_x*np.prod(mm_step)*1E-6
    simulation_data['results']['heatsink'] = telm_list.tolist()
    simulation_data['results']['source'] = {}
    for dsgn in simulation_data["source_layout"]:
        heat_dict = simulation_data["source_layout"][dsgn]
        dt = heat_dict['power']
        if 'Rthjc' in heat_dict:
            dt *= heat_dict['Rthjc']
        else:
            dt *= heat_dict['rect'][2]*heat_dict['rect'][3]/62/108*0.06
        rect_data = np.array(heat_dict["rect"])
        dy = np.minimum(y_grid+ mm_step[0], rect_data[0]+rect_data[2]) - np.maximum(y_grid, rect_data[0])
        dz = np.minimum(z_grid+ mm_step[1], rect_data[1]+rect_data[3]) - np.maximum(z_grid, rect_data[1])
        t = np.sum(telm_list*np.maximum(dy, 0)*np.maximum(dz, 0))/np.prod(rect_data[2:])
        simulation_data['results']['source'][dsgn] = int(t+dt)
    with open(LAST_DESIGN, "w") as writer:
        json.dump(simulation_data, writer)
    print("END OF SIMULATION" + 60 * "#")
    showinfo('DATA GENERATED', '\n'.join((f'{k}:{v}ºC' for (k,v) in simulation_data['results']['source'].items()))+f'\n\nSimulation results generated in file {LAST_DESIGN.name}')
    return True

def color_interp(data_list, color_array, data_range=None):
    """ Interpolate colours
    """
    if data_range is None:
        data_range = (np.min(data_list), np.max(data_list))
    color_len = len(color_array)
    data_range = np.interp(np.arange(color_len), (0,color_len), data_range) 
    for i in range(3):
        yield np.interp(data_list, data_range, color_array[:,i]).astype(np.uint8) 

def open_simulation(prj_path=None):
    global simulation_data, LAST_DESIGN
    # READ PROJECT
    if prj_path.is_file():
        with prj_path.open("r") as reader:
            simulation_data = json.load(reader)
        # save recent
        user_config['recent'][prj_path.as_posix()] = datetime.now().strftime('%Y-%m-%d')
        with open(USER_PATH, "w") as writer:
            json.dump(user_config, writer)
    LAST_DESIGN = prj_path
    def get_data():
        global simulation_data
        try:
            simulation_data["baseplate_width"] = int(y_combo.get())
            simulation_data["baseplate_height"] = int(z_combo.get())
        except ValueError as e:
            showerror('ERROR DE DATOS', f'Las dimensiones deben ser numeros.')
            return False
        simulation_data["heatsink"] = model_combo.get().split(':',1)[0]
        simulation_data['material'] = mat_combo.get().split(':',1)[0]
        simulation_data["ambient_temperature"] = int(tamb_combo.get())
        flow_conditions = flow_combo.get()
        simulation_data["flow_conditions"] = flow_conditions
        if flow_conditions == 'forced_convection':
            v = speed_combo.get().split('.',1)[0]
            if not v.isdigit():
                showerror('AIRFLOW ERROR', f'Air speed velocity must be a positive integer.')
                return False
            simulation_data['air_velocity'] = v
        elif flow_conditions == 'natural_convection' and 'air_velocity' in simulation_data:
            del simulation_data['air_velocity']
        simulation_data['color_palette'] = color_combo.get()
        simulation_data['description'] = desc_combo.get()
        simulation_data['finish']=sutr_combo.get()
        return True
    
    reset_main()

    desc_combo = ttk.Combobox(main_frame, width=70)
    if 'description' in simulation_data:
        desc_combo.set(simulation_data["description"])
    desc_combo.pack(padx=20, pady=20)

    sim_book = ttk.Notebook(main_frame)
    sim_book.pack(fill=tk.BOTH, padx=20, pady=20)
    #ttk.Label(main_frame, text=f"HEATSINK THERMAL SIMULATION {LAST_DESIGN.name}").pack(padx=20, pady=20)
    page_frame = ttk.Frame(sim_book)
    sim_book.add(page_frame, text=f"Boundary Conditions")
    
    grid_frame = ttk.LabelFrame(page_frame, text=f"Environmental conditions")
    grid_frame.pack(fill=tk.BOTH, padx=20, pady=20)
    row_id = 0
    ttk.Label(grid_frame, text="Ambient temperature:\nDefine the ambient temperature surrounding your cooling project.\nThis temperature is used at the inlet where flows enters the system.\nA higher inlet temperature will result in a higher solution temperature.\nA default temperature value of 40°C is selected if you do not know the actual working ambient temperature of your project.").grid(row=row_id, column=0, padx=10, sticky='w')
    tamb_combo = ttk.Combobox(grid_frame)
    tamb_combo["values"] = list(range(50, 90, 10))
    if 'ambient_temperature' in simulation_data:
        tamb_combo.set(simulation_data["ambient_temperature"])
    else:
        tamb_combo.set(45)
    tamb_combo.grid(row=row_id, column=1, padx=10, pady=10)
    ttk.Label(grid_frame, text="ºC").grid(row=row_id, column=2, padx=10, sticky='w')
    row_id += 1
    ttk.Label(grid_frame, text="Altitude:\nDefine the altitude in which your project will be working in.\nThis value determines the pressure at the inlet where flows enters the system.\nA higher inlet temperature will result in a higher solution temperature").grid(row=row_id, column=0, padx=10, sticky='w')
    
    grid_frame = ttk.LabelFrame(page_frame, text=f"Heatsink Volume")
    grid_frame.pack(fill=tk.BOTH, padx=20, pady=20)
    row_id = 0

    ttk.Label(grid_frame, text="Width Y").grid(row=row_id, column=0, padx=10)
    y_combo = ttk.Combobox(grid_frame)
    y_combo["values"] = list(range(300, 1000, 50))
    if 'baseplate_width' in simulation_data:
        y_combo.set(simulation_data["baseplate_width"])
    else:
        y_combo.set(300)
    y_combo.grid(row=row_id, column=1, padx=10, pady=10)
    ttk.Label(grid_frame, text="mm").grid(row=row_id, column=2, padx=10, sticky='w')

    row_id += 1
    ttk.Label(grid_frame, text="Height Z").grid(row=row_id, column=0, padx=10)
    z_combo = ttk.Combobox(grid_frame)
    z_combo["values"] = list(range(300, 1000, 50))
    if 'baseplate_height' in simulation_data:
        z_combo.set(simulation_data["baseplate_height"])
    else:
        z_combo.set(400)
    z_combo.grid(row=row_id, column=1, padx=10, pady=10)
    ttk.Label(grid_frame, text="mm").grid(row=row_id, column=2, padx=10, sticky='w')
    row_id += 1
    with Image.open(Path('./img/baseplate-dimensions-mbf.png')) as i:
        global dimens_img
        dimens_img = ImageTk.PhotoImage(i.resize((400,200)))
    tk.Label(grid_frame, image=dimens_img).grid(row=0, column=3, padx=10, rowspan=2)
    

    page_frame = ttk.Frame(sim_book)
    sim_book.add(page_frame, text=f"Source layout")

    grid_frame = ttk.LabelFrame(page_frame, text=f"Definition")
    grid_frame.pack(fill=tk.BOTH, padx=20, pady=20)

    row_id = 0
    ttk.Label(grid_frame, text='Designator').grid(row=row_id, column=0, padx=10)
    dsgn_combo = ttk.Combobox(grid_frame)
    if 'source_layout' not in simulation_data:
        simulation_data['source_layout'] = {}
    dsgn_list = sorted(list(simulation_data['source_layout']))
    dsgn_combo['values'] = dsgn_list
    dsgn_combo.grid(row=row_id, column=1, padx=10, pady=10)

    row_id += 1
    ttk.Label(grid_frame, text='Power').grid(row=row_id, column=0, padx=10)
    power_combo = ttk.Combobox(grid_frame)
    power_combo['values'] = [100*i for i in range(1,5)]
    power_combo.grid(row=row_id, column=1, padx=10, pady=10)
    ttk.Label(grid_frame, text='W').grid(row=row_id, column=2, padx=10)
    
    row_id += 1
    ttk.Label(grid_frame, text='Position Y,Z').grid(row=row_id, column=0, padx=10)
    sourceyz_combo = ttk.Combobox(grid_frame)
    sourceyz_combo['values'] = ['100,100','200,200','300,300']
    sourceyz_combo.grid(row=row_id, column=1, padx=10, pady=10)
    ttk.Label(grid_frame, text='mm').grid(row=row_id, column=2, padx=10)

    row_id += 1
    ttk.Label(grid_frame, text='Module').grid(row=row_id, column=0, padx=10)
    package_combo = ttk.Combobox(grid_frame)
    package_combo['values'] = DBH.list_package()
    package_combo['state'] = 'readonly'
    package_combo.grid(row=row_id, column=1, padx=10, pady=10)
    ttk.Label(grid_frame, text='mm').grid(row=row_id, column=2, padx=10)
    def set_package(event=None):
        w,h,r = DBH.get_package(package_combo.get())
        sourcewh_combo.set(f'{w}x{h}')
        thjc_combo.set(r)
    package_combo.bind("<<ComboboxSelected>>", set_package)

    row_id += 1
    ttk.Label(grid_frame, text='Size WxH').grid(row=row_id, column=0, padx=10)
    sourcewh_combo = ttk.Combobox(grid_frame)
    sourcewh_combo['values'] = ['20x92', '25x38', '34x94','42x105', '73x140', '62x108','89x172', '89x250']
    ttk.Label(grid_frame, text='mm').grid(row=row_id, column=2, padx=10, pady=10)
    sourcewh_combo.grid(row=row_id, column=1, padx=10)
    def flip_wh():
        wh = sourcewh_combo.get().replace('x',',')
        if wh.count(',') != 1:
            return False 
        w,h = wh.split(',')
        sourcewh_combo.set(f'{h}x{w}')
    ttk.Button(grid_frame, text='FLIP', command=flip_wh).grid(row=row_id, column=3, padx=10)

    row_id += 1
    ttk.Label(grid_frame, text='Rth(j,c)').grid(row=row_id, column=0, padx=10)
    thjc_combo = ttk.Combobox(grid_frame)
    thjc_combo['values'] = [0.01*i for i in range(5,20)]
    thjc_combo.grid(row=row_id, column=1, padx=10, pady=10)
    ttk.Label(grid_frame, text='K/W').grid(row=row_id, column=2, padx=10)

    with Image.open(Path('./img/heat_source.png')) as i:
        global xywh_img
        xywh_img = ImageTk.PhotoImage(i.resize((300,200)))
    tk.Label(grid_frame, image=xywh_img).grid(row=0, column=3, padx=10, rowspan=5)

    # functionality
    def read_dsgn(event=None):
        dsgn = dsgn_combo.get()
        if dsgn == '':
            return False
        if dsgn in simulation_data['source_layout']:
            d = simulation_data['source_layout'][dsgn]
            sourceyz_combo.set(','.join((str(i) for i in d['rect'][:2])))
            sourcewh_combo.set('x'.join((str(i) for i in d['rect'][2:])))
            power_combo.set(d['power'])
            if 'Rthjc' in d:
                thjc_combo.set(d['Rthjc'])
            else:
                thjc_combo.set(round(d['rect'][2]*d['rect'][3]/62/108*0.06,3))
    dsgn_combo.bind("<<ComboboxSelected>>", read_dsgn)
    dsgn_combo.bind("<Return>", read_dsgn)
    if len(dsgn_list)>0:
        dsgn_combo.set(dsgn_list[0])
        read_dsgn()

    def save_source():
        dsgn = dsgn_combo.get()
        if dsgn == '':
            return False
        try:
            p = int(power_combo.get())
            y,z = (int(s) for s in sourceyz_combo.get().split(',',1))
            w,h = (int(s) for s in sourcewh_combo.get().lower().replace('x',',').split(',',1))
            thjc = float(thjc_combo.get())
        except ValueError as e:
            showerror('DATA INPUT ERROR', e)
            return False
        if p<1:
            if dsgn in simulation_data['source_layout']:
                del simulation_data['source_layout'][dsgn]
            return True
        simulation_data['source_layout'][dsgn] = {
            "rect":[y,z,w,h],
            'power':p,
            'Rthjc':thjc
        }
        dsgn_combo['values'] = sorted(list(simulation_data['source_layout']))
        return True
        
    row_id += 1
    ttk.Button(grid_frame, text='SAVE HEAT SOURCE', command=save_source).grid(row=row_id, column=0, columnspan=2, padx=10, pady=10
    )

    def show_layout():
        """ Generate layout image."""
        if not get_data():
            return False
        out_dir = LAST_DESIGN.parent / LAST_DESIGN.name.split(('.'))[0]
        out_dir.mkdir(exist_ok=True)
        file_path=out_dir / "source_layout.png"
        img_size = (
            simulation_data["baseplate_width"], simulation_data["baseplate_height"]
            )
        # create new image
        power_img = Image.new(mode='L', size=img_size, color='black')
        # draw power elements
        v1 = img_size*np.array((0,1))
        m1 = np.array([[1,0],[0,-1]])
        draw = ImageDraw.Draw(power_img)
        fnt = ImageFont.load_default()
        source_dict = simulation_data["source_layout"]
        for k in source_dict:
            heat_dict = source_dict[k]
            rect_data = np.array(heat_dict["rect"])
            yz_bl_h = rect_data[:2]
            yz_tr_h = yz_bl_h + rect_data[2:]
            yz_bl_h = v1 + m1@yz_bl_h
            yz_tr_h = v1 + m1@yz_tr_h
            draw.rectangle(
                yz_bl_h.tolist() + yz_tr_h.tolist(),
                outline='white',
                fill='white',
                width=1,
            )
            p = heat_dict['power']
            draw.text(yz_bl_h.tolist(), f'{k}:{p}W', font=fnt, fill='white')
        power_img.save(file_path)
        print(f"See source layout {file_path}")
        startfile(file_path)
        return True

    ttk.Button(page_frame, text="SHOW ALL HEAT SOURCES", command=show_layout).pack(padx=10, pady=10
    )
    def rotate_layout():
        for dsgn in simulation_data['source_layout']:
            r = simulation_data['source_layout'][dsgn]['rect']
            yz = np.array((simulation_data["baseplate_width"],simulation_data["baseplate_height"]))-r[:2]-r[2:]
            simulation_data['source_layout'][dsgn]['rect'] = yz.tolist()+r[2:]
        #showinfo('ROTATED SOURCES', f'Heat sources have been rotated.')
        return show_layout()
    ttk.Button(page_frame, text="ROTATE HEATSINK 180 degrees along depth X", command=rotate_layout).pack(padx=10, pady=10
    )


    grid_frame = ttk.Frame(sim_book)
    sim_book.add(grid_frame, text=f"Flow conditions")

    row_id = 0
    ttk.Label(grid_frame, text="Air flow type").grid(row=row_id, column=0, padx=10)
    flow_combo = ttk.Combobox(grid_frame)
    flow_combo["values"] = ['natural_convection','forced_convection']
    if 'flow_conditions' in simulation_data:
        flow_combo.set(simulation_data["flow_conditions"])
    else:
        flow_combo.set('natural_convection')
    flow_combo.grid(row=row_id, column=1, padx=10, pady=10)

    row_id += 1
    ttk.Label(grid_frame, text="Air velocity").grid(row=row_id, column=0, padx=10)
    speed_combo = ttk.Combobox(grid_frame)
    speed_combo["values"] = list(range(1,10))
    speed_combo['state'] = 'readonly'  
    if 'air_velocity' in simulation_data:
        speed_combo.set(simulation_data["air_velocity"])
    elif 'flow_conditions' in simulation_data and simulation_data['flow_conditions']=='forced_convection':
        speed_combo.set(4)
    speed_combo.grid(row=row_id, column=1, padx=10, pady=10)
    ttk.Label(grid_frame, text="m/s (forced convection only)").grid(row=row_id, column=2, padx=10, sticky='w')
    
    grid_frame = ttk.Frame(sim_book)
    sim_book.add(grid_frame, text=f"Heatsink selection")
    row_id = 0
    ttk.Label(grid_frame, text="Model").grid(row=row_id, column=0, padx=10)
    model_combo = ttk.Combobox(grid_frame, width=70)
    model_combo["state"] = "readonly"
    HEATSINK_LIST = DBH.list_heatsink()
    model_combo["values"] = HEATSINK_LIST
    if 'heatsink' in simulation_data:
        k = simulation_data['heatsink']
        for s in HEATSINK_LIST:
            if s.split(':',1)[0] == k:
                model_combo.set(s)
    else:
        model_combo.set(HEATSINK_LIST[0])
    model_combo.grid(row=row_id, column=1, padx=10, pady=10)
    hs_info = tk.StringVar()
    ttk.Label(grid_frame, textvariable=hs_info).grid(row=row_id, column=2, padx=10, pady=10)
    def estimate_increase(event=None):
        """ Final temperature estimation"""
        if get_data():
            if not DBH.select_heatsink(simulation_data["heatsink"], simulation_data["baseplate_width"], simulation_data["flow_conditions"], simulation_data['air_velocity'] if 'air_velocity' in simulation_data else None):
                hs_info.set('. '.join(DBH.e_list))
                DBH.e_list = []
                return False
            heat_sum = sum(d['power'] for d in simulation_data["source_layout"].values())
            Telm = simulation_data["ambient_temperature"] + heat_sum/DBH.scaled_convection(simulation_data['baseplate_height'])
            Telm = Telm.astype(int)
            hs_info.set(f'Heatsink average temperature will be {Telm} C')
    model_combo.bind("<<ComboboxSelected>>", estimate_increase)
    row_id += 1
    ttk.Label(grid_frame, text="Material").grid(row=row_id, column=0, padx=10)
    mat_combo = ttk.Combobox(grid_frame, width=70)
    mat_combo["state"] = "readonly"
    MATERIAL_LIST = DBH.list_material()
    mat_combo["values"] = MATERIAL_LIST
    if 'material' in simulation_data:
        k = simulation_data['material']
        for s in MATERIAL_LIST:
            if s.split(':',1)[0] == k:
                mat_combo.set(s)
    else:
        mat_combo.set(MATERIAL_LIST[0])
    mat_combo.grid(row=row_id, column=1, padx=10, pady=10)
    mat_info = tk.StringVar()
    ttk.Label(grid_frame, textvariable=mat_info).grid(row=row_id, column=2, padx=10, pady=10)
    def show_conductivity(event=None):
        if get_data():
            kth = DBH.scaled_conductivity(simulation_data["material"]).astype(int)
            if kth is None:
                mat_info.set('. '.join(DBH.e_list))
                DBH.e_list = []
                return False
            mat_info.set(f'Thermal conductivity {kth} W/m.K @ 25 C.')
    mat_combo.bind("<<ComboboxSelected>>", show_conductivity)
    row_id += 1
    ttk.Label(grid_frame, text="Finish").grid(row=row_id, column=0, padx=10)
    sutr_combo = ttk.Combobox(grid_frame, width=70)
    sutr_combo["state"] = "readonly"
    SUTR_LIST = DBH.list_surface_treatment()
    sutr_combo["values"] = SUTR_LIST
    if 'finish' in simulation_data:
        k = simulation_data['finish']
        for s in SUTR_LIST:
            if s.split(':',1)[0] == k:
                sutr_combo.set(s)
    else:
        sutr_combo.set(SUTR_LIST[0])
    sutr_combo.grid(row=row_id, column=1, padx=10, pady=10)
    sfc_info = tk.StringVar()
    ttk.Label(grid_frame, textvariable=sfc_info).grid(row=row_id, column=2, padx=10, pady=10)
    def show_emissivity(event=None):
        if get_data():
            k = DBH.finish[simulation_data['finish']]
            sfc_info.set(f'Surface emissivity is {k}')
    sutr_combo.bind("<<ComboboxSelected>>", show_emissivity)


    grid_frame = ttk.Frame(sim_book)
    sim_book.add(grid_frame, text=f"Plot")
    
    row_id = 0
    with open(COLOR_PATH,'r') as reader:
        palette_dict = json.load(reader)
    ttk.Label(grid_frame, text="Color palette").grid(row=row_id, column=0, padx=10)
    color_combo = ttk.Combobox(grid_frame)
    palette_list = list(palette_dict)
    color_combo["values"] = palette_list
    color_combo['state']='readonly'
    color_combo.grid(row=row_id, column=1, padx=10, pady=10)
    color_canvas = tk.Canvas(grid_frame, height=20, width=512)
    color_canvas.grid(row=row_id, column=2, padx=10, pady=10)
    def show_palette(event=None):
        color_name = color_combo.get()
        color_array = np.array(palette_dict[color_name])
        color_x = np.arange(512)
        ra, ga, ba = color_interp(color_x, color_array)
        for i in color_x:
            rgb = "#%02x%02x%02x" % (ra[i],ga[i],ba[i])
            color_canvas.create_line(i, 0, i, 20, fill=rgb)
    color_combo.bind("<<ComboboxSelected>>", show_palette)
    if "color_palette" in simulation_data:
        color_combo.set(simulation_data['color_palette'])
        show_palette()
    elif len(palette_list)>0:
        color_combo.set(palette_list[0])
        show_palette()
    
    row_id += 1
    ttk.Label(grid_frame, text="Minimum temperature").grid(row=row_id, column=0, padx=10)
    min_combo = ttk.Combobox(grid_frame)
    min_combo["values"] = ['ambient_temperature','minimum_temperature']
    min_combo.set('ambient_temperature')
    min_combo['state']='readonly'
    min_combo.grid(row=row_id, column=1, padx=10, pady=10)

    def run_sim():
        if get_data():
            return run_simulation()
    
    grid_frame = ttk.LabelFrame(main_frame, text=f"Simulation")
    grid_frame.pack(fill=tk.BOTH, padx=20, pady=20)
    
    ttk.Button(grid_frame, text="RUN SIMULATION", command=run_sim).grid(row=0, column=0,
        padx=20, pady=20
    )

    def plot_results():
        if 'results' not in simulation_data:
            showerror('RESULTS NOT FOUND', f'Simulation results re missing.')
            return False
        if not get_data():
            return False
        out_dir = LAST_DESIGN.parent / LAST_DESIGN.name.split(('.'))[0] #+'-'+str(int(datetime.now().timestamp())))
        out_dir.mkdir(exist_ok=True)
        # OUTPUT
        # thermal image
        plot_path = out_dir / "thermal_distribution.png"
        telm_list = np.array(simulation_data['results']['heatsink'])
        with open(COLOR_PATH,'r') as reader:
            color_array = np.array(json.load(reader)[simulation_data["color_palette"]]).astype(np.uint8)
        if plot_array(
            telm_list,
            color_array,
            plot_path,
            min_combo.get()[0] == 'a'
        ):
            startfile(plot_path)

    ttk.Button(grid_frame, text="PLOT RESULT", command=plot_results).grid(row=0, column=1,
        padx=20, pady=20
    )


    def save_simulation():
        file_path = tk_filedialog.asksaveasfilename(
            confirmoverwrite=False,
            initialdir=WORSPACE_DIR,
            initialfile=LAST_DESIGN,
            title="Select project 0XXXXX.json file",
            filetypes=(("json files", "*.json"), ("all files", "*.*")),
        )

        if not get_data():
            return False
        with open(file_path, "w") as writer:
            json.dump(simulation_data, writer)
        return True

    ttk.Button(grid_frame, text="SAVE CONFIGURATION", command=save_simulation).grid(row=0, column=2,
        padx=20, pady=20
    )
    ttk.Button(
        grid_frame, text="OPEN A DIFFERENT CONFIGURATION", command=start_page
    ).grid(row=0, column=3,
        padx=20, pady=20
    )

def start_page():
    reset_main()
    grid_frame = ttk.LabelFrame(main_frame, text=f"Recent projects")
    grid_frame.pack(fill=tk.BOTH, padx=20, pady=20)

    row_id = 0
    ttk.Label(grid_frame, text="Path").grid(row=row_id, column=0, padx=10, pady=10)
    path_combo = ttk.Combobox(grid_frame, width=70)
    recent_list = [Path(k) for k in user_config['recent'] if Path(k).is_file()]
    path_combo["values"] = sorted(recent_list)
    if len(recent_list)>0:
       path_combo.set(recent_list[0])
    path_combo.grid(row=row_id, column=1, padx=10, pady=10)
    def open_selected():
        prj_path = Path(path_combo.get())
        if prj_path.is_file():
            return open_simulation(prj_path)
        showerror('BROKEN PATH', f'File {prj_path} not found.')
        return False
    
    ttk.Button(grid_frame, text="OPEN", command=open_selected).grid(row=row_id, column=2, padx=10, pady=10)

    def browse_configuration():
        # open
        prj_path = Path(
            tk_filedialog.askopenfilename(
                initialdir=WORSPACE_DIR,
                title="Select project 0XXXXX.json file",
                filetypes=(("json files", "*.json"), ("all files", "*.*")),
            )
        )
        if prj_path.is_file():
            path_combo.set(prj_path)

    ttk.Button(main_frame, text="BROWSE CONFIGURATION", command=browse_configuration).pack(
        padx=20, pady=20
    )
    def new_simulation():
        global simulation_data
        simulation_data = {}
        return open_simulation(WORSPACE_DIR / "new_project.json")

    ttk.Button(main_frame, text="NEW CONFIGURATION", command=new_simulation).pack(
        padx=20, pady=20
    )


if __name__ == "__main__":
    if not DATABASE_PATH.is_file():
        showerror('DATABASE MISSING', f'File {DATABASE_PATH} is missing.')
        exit(0)
    DBH = Database(DATABASE_PATH)
    
    tk_mgr = tk.Tk()
    tk_mgr.geometry("1280x720")
    tk_mgr.config(bg="#F8F8FF")
    tk_mgr.title(f"HEATSINK SIMULATOR (version {__version__}).")

    main_frame = ttk.Frame(tk_mgr)
    main_frame.pack(fill=tk.BOTH, expand=1)

    
    start_page()

    tk_mgr.mainloop()
