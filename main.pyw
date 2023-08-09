"""
0.1.0 First working prototype.
0.1.1 Thermal scale. Power img. Vertical.
0.1.2 Temperature integral.
0.1.3 Forced convection.
"""

import json
import tkinter as tk
import tkinter.filedialog as tk_filedialog
import tkinter.ttk as ttk
from datetime import datetime
from os import startfile
from pathlib import Path
from tkinter.messagebox import showerror, showinfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__author__ = "Kostadin Kostadinov"
__copyright__ = "INGENIERIA VIESCA S.L."
__credits__ = ["Kostadin Kostadinov"]
__license__ = "TBD"
__version__ = "0.1.3"
__maintainer__ = "Kostadin Kostadinov"
__email__ = "kostadin.ivanov@ingenieriaviesca.com"
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
if not DATABASE_PATH.is_file():
    showerror('DATABASE MISSING', f'File {DATABASE_PATH} is missing.')
    exit(0)
with open(DATABASE_PATH, 'r') as reader:
    HEATSINK_DATA = json.load(reader)
    HEATSINK_LIST = [f'{k}:{HEATSINK_DATA[k]["description"]}' for k in sorted(HEATSINK_DATA)]

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

def scaled_convection(z_list, convection_z, convection_Y):
    """Estimate vertical convection from 0 to 1st z, 1st to 2nd, etc.
    admittance is in W/K, corresponds to dz*simulation_baseplate_width
    Z ascending list contains vertical coordinates
    """
    # interpolate for z
    Y_values = np.interp(np.append(z_list, 0), convection_z, convection_Y)
    return (Y_values-np.roll(Y_values, 1))[:-1]


def edit_layout():
    startfile(LAST_DESIGN)
    #showinfo("SOURCE LAYOUT", simulation_data["source_layout"])


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
    full_image.paste(thermal_img, (0,100))
    draw = ImageDraw.Draw(full_image)
    # arrow
    yz0 = (img_size *(0.5,0)+(0,20)).astype(int)
    yz1 = yz0+(0,60)
    draw.line([tuple(yz.tolist()) for yz in (yz0, yz1)], fill="black", width=3)
    yz1 = yz0+(20,20)
    yz2 = yz0+(-20,20)
    draw.line([tuple(yz.tolist()) for yz in (yz2, yz0, yz1)], fill="black", width=3)
    # temperature scale
    
    fnt = ImageFont.load_default()
    palette_array = np.repeat(np.arange(256),2).reshape(256,2)
    ra,ga,ba = color_interp(palette_array, color_array)
    palette_img = Image.fromarray(np.stack((ra,ga,ba), axis=-1), mode='RGB')
    palette_img = palette_img.resize((20, 256)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    full_image.paste(palette_img, (full_size[0]-30, full_size[1]-266))
    # minimum
    t0 = int(data_range[0])
    yz0 = full_size-(60,10)
    draw.point(tuple(yz0+(29,0)), fill="black")
    draw.text(yz0.tolist(), f'{t0}', font=fnt, fill="black")
    # maximum
    t0 = int(data_range[1])
    yz0 -= (0, 256)
    draw.point(tuple(yz0+(29,0)), fill="black")
    draw.text(tuple(yz0), f'{t0}', font=fnt, fill="black")
    if down2ambient:
        # heatsink
        t0 = int(np.min(telm_array))
        yz0 += (0, int((data_range[1]-t0)/(data_range[1]-data_range[0])*256))
        draw.point(tuple(yz0+(29,0)), fill="black")
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
    conv_type = simulation_data["flow_conditions"]
    hs_key = simulation_data["heatsink"]
    if conv_type not in HEATSINK_DATA[hs_key]:
        showerror('HEATSINK ERROR', f'No {conv_type} data available for heatsink {hs_key}.')
        return False
    # pull convection data from database. admittance on 0 is 0
    convection_array = np.array(HEATSINK_DATA[hs_key][conv_type]['thermal_resistance'])
    convection_z = np.insert(convection_array[:,0], 0, 0)
    convection_Y = np.insert(convection_array[:, 1]**-1*simulation_data["baseplate_width"]/HEATSINK_DATA[hs_key]['baseplate_width'],0,0)
    if conv_type == 'forced_convection':
        gain_array = np.array(HEATSINK_DATA[hs_key][conv_type]['correction_factor'])
        conv_gain = np.interp(simulation_data["air_velocity"], gain_array[:,0],gain_array[:,1]**-1)
        convection_Y *= conv_gain
    convection_data = [convection_z, convection_Y]
    conductivity_yz = HEATSINK_DATA[hs_key]["thermal_conductivity"]*HEATSINK_DATA[hs_key]["baseplate_thickness"]
    dt_experimental = HEATSINK_DATA[hs_key][conv_type]['steady_temperature']-HEATSINK_DATA[hs_key][conv_type]['ambient_temperature']
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
    Yavg_convection = scaled_convection(simulation_data["baseplate_height"], convection_data[0], convection_data[1])
    print(f"Rth_cv={np.round(Yavg_convection**-1,4)} K/W")

    #conductivity_x = HEATSINK_DATA[hs_key]["thermal_conductivity"]/HEATSINK_DATA[hs_key]["baseplate_thickness"]
    
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
        conv_adm_column = scaled_convection(z_list, convection_data[0], convection_data[1])
        amb_heat_k = (1+np.arange(ryz_cnt[0]))*(dt_experimental**0.25)
        if conv_type == 'forced_convection':
            conv_adm_list = np.repeat(conv_adm_column, ryz_cnt[1])/ryz_cnt[1]
            conv_adm_list = np.reshape(conv_adm_list, ryz_cnt)
        # loop until error gets below 10C divided by all the elements
        simulation_count = 0
        dt_error = 100  #
        while dt_error > 3 and simulation_count < MAXIMUM_ITERATIONS:
            #print(f"LOOP {simulation_count} t_error={round(dt_error,6)}") # + 60 * "#"
            # CONVECTION TEMPERATURE CORRECTION
            if conv_type == 'natural_convection':
                dt_loop = telm_list-tamb_scalar
                conv_adm_list = np.interp(integral_matrix@(dt_loop**0.25), amb_heat_k, conv_adm_column/ryz_cnt[1])
            #Ycv_scale_factor = Yavg_convection/np.sum(conv_adm_list)
            #conv_adm_list *= Ycv_scale_factor
            #print(f'Convection scale factor is {np.round(Ycv_scale_factor,3)}')
            #print(conv_adm_list)
            print(f'Total convection resistance is {np.round(np.sum(conv_adm_list)**-1,3)} K/W')
            '''rel_th = np.round((np.min(conv_adm_list), np.max(conv_adm_list)), 2)
            print(
                f"Rth_cv avg is {round(np.sum(conv_adm_list)**-1, 4)} K/W, in range {rel_th}"
            )'''
            # RADIATION ADMITTANCE LIST: shape is 1 x matrix_size W/K
            rad_adm_list = (
                0.9
                * 5.67e-8
                * np.prod(mm_step)
                * 1e-6
                * ((telm_list + 273.15) ** 4 - (273.15 + tamb_scalar) ** 4)
            ) / (telm_list - tamb_scalar)
            #print(f'Total radiation resistance is {np.round(np.sum(rad_adm_list)**-1,3)} K/W')
            #print(rad_adm_list)
            # CONVECTION + RADIATION admittance in W/K
            admittance_matrix = np.eye(matrix_size) * (rad_adm_list + conv_adm_list).ravel()
            # CONDUCTION singular matrix in W/K
            conduction_matrix = 0 #matrix_size x matrix_size
            if True:#map_partition > 0:
                # conduction with adjacent roll and next roll
                # kk for di in (1, ryz_cnt[0]):  # np.arange(1, matrix_size):
                eye_sum = 0
                for di in  (1, ryz_cnt[1]):#for roll_axis in (0, 1): #range(1,matrix_size):
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
                    t_loop = np.roll(telm_list, di)
                    # temperature difference to ambient
                    # np.where(dtr > 1 temperature gradient over difference to ambient, when at least 1 C of difference to ambient is present
                    #gt = np.abs(telm_list - t_loop)
                    #dtr = gt/(telm_list/2 +t_loop/2 - tamb_scalar)
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
                    conduction_matrix += (
                        np.roll(dyzr_eye, -di) - dyzr_eye + np.roll(eye2, di) - eye2
                    )
                if False:
                    print("CONDUCTION TRANSFER EYE MATRIX")
                    print(np.reshape(np.diag(eye_sum), ryz_cnt))
                # values in W/K
                conduction_matrix *= conductivity_yz
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
            print(conv_adm_list.reshape(ryz_cnt)*(telm_list-tamb_scalar).astype(int))
            print('POWER TROUGH RADIATION (W)')
            print(rad_adm_list.reshape(ryz_cnt)*(telm_list-tamb_scalar).astype(int))
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
            plot_path = out_dir / f"loop{map_partition}.png"
            plot_array(
                np.reshape(telm_list, ryz_cnt),
                color_array,
                plot_path
            )
        if simulation_count + 1 > MAXIMUM_ITERATIONS:
            showerror('CONVERGING ERROR', f'MAXIMUM ITERATIONS REACH WITHOUT CONVERGING: dt_error={round(dt_error)} K, with shape {np.shape(telm_list)}')
            return False
    # power grid
    simulation_data['results'] = {}
    simulation_data['results']['temperature'] = telm_list.tolist()
    with open(LAST_DESIGN, "w") as writer:
        json.dump(simulation_data, writer)
    print("END OF SIMULATION" + 60 * "#")
    showinfo('DATA GENERATED', f'Simulation results generated in file {LAST_DESIGN.name}')
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
        simulation_data["ambient_temperature"] = int(tamb_combo.get())
        conv_type = flow_combo.get()
        simulation_data["flow_conditions"] = conv_type
        if conv_type == 'forced_convection':
            simulation_data['air_velocity'] = round(float(speed_combo.get()), 1)
        elif conv_type == 'natural_convection' and 'air_velocity' in simulation_data:
            del simulation_data['air_velocity']
        simulation_data['color_palette'] = color_combo.get()
        return True
    
    reset_main()
    #ttk.Label(main_frame, text=f"HEATSINK THERMAL SIMULATION {LAST_DESIGN.name}").pack(padx=20, pady=20)
    grid_frame = ttk.LabelFrame(main_frame, text=f"Heatsink configuration")
    grid_frame.pack(fill=tk.BOTH, padx=20, pady=20)

    row_id = 0
    ttk.Label(grid_frame, text="Description").grid(row=row_id, column=0, padx=10)
    desc_combo = ttk.Combobox(grid_frame, width=70)
    if 'description' in simulation_data:
        desc_combo.set(simulation_data["description"])
    desc_combo.grid(row=row_id, column=1, padx=10, pady=10, columnspan=2)
    
    row_id = 1
    ttk.Label(grid_frame, text="Model").grid(row=row_id, column=0, padx=10)
    model_combo = ttk.Combobox(grid_frame, width=70)
    model_combo["state"] = "readonly"
    model_combo["values"] = HEATSINK_LIST
    if 'heatsink' in simulation_data:
        k = simulation_data['heatsink']
        model_combo.set(f'{k}:{HEATSINK_DATA[k]["description"]}')
    else:
        model_combo.set(HEATSINK_LIST[0])
    model_combo.grid(row=row_id, column=1, padx=10, columnspan=2)

    row_id += 1
    ttk.Label(grid_frame, text="Width").grid(row=row_id, column=0, padx=10)
    y_combo = ttk.Combobox(grid_frame)
    y_combo["values"] = list(range(300, 1000, 50))
    if 'baseplate_width' in simulation_data:
        y_combo.set(simulation_data["baseplate_width"])
    else:
        y_combo.set(300)
    y_combo.grid(row=row_id, column=1, padx=10, pady=10)
    ttk.Label(grid_frame, text="mm").grid(row=row_id, column=2, padx=10, sticky='w')

    row_id += 1
    ttk.Label(grid_frame, text="Height").grid(row=row_id, column=0, padx=10)
    z_combo = ttk.Combobox(grid_frame)
    z_combo["values"] = list(range(300, 1000, 50))
    if 'baseplate_height' in simulation_data:
        z_combo.set(simulation_data["baseplate_height"])
    else:
        z_combo.set(400)
    z_combo.grid(row=row_id, column=1, padx=10, pady=10)
    ttk.Label(grid_frame, text="mm").grid(row=row_id, column=2, padx=10, sticky='w')

    row_id += 1
    ttk.Label(grid_frame, text="Source layout").grid(row=row_id, column=0, padx=10)
    ttk.Button(grid_frame, text="EDIT", command=edit_layout).grid(
        row=row_id, column=1, padx=10, pady=10
    )
    def show_layout():
        """ Generate layout image."""
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

    ttk.Button(grid_frame, text="VIEW", command=show_layout).grid(
        row=row_id, column=2, padx=10, pady=10
    )


    grid_frame = ttk.LabelFrame(main_frame, text=f"Setup configuration")
    grid_frame.pack(fill=tk.BOTH, padx=20, pady=20)

    row_id = 0
    ttk.Label(grid_frame, text="Ambient temperature").grid(row=row_id, column=0, padx=10)
    tamb_combo = ttk.Combobox(grid_frame)
    tamb_combo["values"] = list(range(50, 90, 10))
    if 'ambient_temperature' in simulation_data:
        tamb_combo.set(simulation_data["ambient_temperature"])
    else:
        tamb_combo.set(45)
    tamb_combo.grid(row=row_id, column=1, padx=10, pady=10)
    ttk.Label(grid_frame, text="ºC").grid(row=row_id, column=2, padx=10, sticky='w')

    row_id += 1
    ttk.Label(grid_frame, text="Flow conditions").grid(row=row_id, column=0, padx=10)
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
    if 'air_velocity' in simulation_data:
        speed_combo.set(simulation_data["air_velocity"])
    elif 'flow_conditions' in simulation_data and simulation_data['flow_conditions']=='forced_convection':
        speed_combo.set(4)
    speed_combo.grid(row=row_id, column=1, padx=10, pady=10)
    ttk.Label(grid_frame, text="m/s (forced convection only)").grid(row=row_id, column=2, padx=10, sticky='w')
    
    row_id += 1
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
        out_dir = LAST_DESIGN.parent / LAST_DESIGN.name.split(('.'))[0] #+'-'+str(int(datetime.now().timestamp())))
        out_dir.mkdir(exist_ok=True)
        # OUTPUT
        # thermal image
        plot_path = out_dir / "thermal_distribution.png"
        telm_list = np.array(simulation_data['results']['temperature'])
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

    ttk.Button(main_frame, text="SAVE CONFIGURATION", command=save_simulation).pack(
        padx=20, pady=20
    )
    ttk.Button(
        main_frame, text="OPEN A DIFFERENT CONFIGURATION", command=start_page
    ).pack(padx=20, pady=20)

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
    tk_mgr = tk.Tk()
    tk_mgr.geometry("900x600")
    tk_mgr.config(bg="#F8F8FF")
    tk_mgr.title(f"HEATSINK SIMULATOR (version {__version__}). {__copyright__}")

    main_frame = ttk.Frame(tk_mgr)
    main_frame.pack(fill=tk.BOTH, expand=1)

    start_page()

    tk_mgr.mainloop()
