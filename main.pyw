"""
0.1.0 First working prototype.
"""

import json
import tkinter as tk
import tkinter.filedialog as tk_filedialog
import tkinter.ttk as ttk
from datetime import datetime
from os import startfile
from pathlib import Path
from tkinter import messagebox
from tkinter.messagebox import showinfo

import numpy as np
from PIL import Image, ImageDraw

__author__ = "Kostadin Kostadinov"
__copyright__ = "INGENIERIA VIESCA S.L."
__credits__ = ["Kostadin Kostadinov"]
__license__ = "TBD"
__version__ = "0.1.0"
__maintainer__ = "Kostadin Kostadinov"
__email__ = "kostadin.ivanov@ingenieriaviesca.com"
__status__ = "Alpha"

WORSPACE_DIR = Path('./')/'simulation'#Path.home() / "Desktop"
WORSPACE_DIR.mkdir(exist_ok=True)
LAST_DESIGN = WORSPACE_DIR / "test.json"
COLOR_DIR = Path('./color_palette')


simulation_data = {}
MAXIMUM_ITERATIONS = 999
SIMULATION_SECONDS = 20
print_help = False


# mm, K/W
DATABASE_PATH = Path('./database.json')
if not DATABASE_PATH.is_file():
    messagebox.showerror('DATABASE MISSING', f'File {DATABASE_PATH} is missing.')
    exit(0)
with open(DATABASE_PATH, 'r') as reader:
    HEATSINK_DATA = json.load(reader)

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


def scaled_convection(z_loop):
    """Estimate vertical convection factors using tata from datasheet"""
    conv_type = simulation_data["flow_conditions"]
    hs_key = simulation_data["heatsink"]
    y_loop = simulation_data["baseplate_width"]
    # find closest x
    closest_x = min(abs(yzR[0] - y_loop) for yzR in HEATSINK_DATA[hs_key][conv_type]['thermal_resistance'])
    # scale for y
    convection_z = [0]
    convection_R = []
    for y, z, R in HEATSINK_DATA[hs_key][conv_type]['thermal_resistance']:
        if abs(y - y_loop) < closest_x + 1:
            convection_z.append(z)
            # same size
            convection_R.append(R * y / y_loop)
    # interpolate for z
    return np.interp(
        z_loop,
        convection_z,
        [convection_R[0]] + convection_R,
    )


def edit_layout():
    showinfo("SOURCE LAYOUT", simulation_data["source_layout"])


def plot_array(array2d, file_path="loop.png"):
    array_shape = np.shape(array2d) # number of rows, number of columns
    # create new image
    thermal_img = Image.new(mode='RGB', size=(array_shape[1], array_shape[0]))
    """ Map values from 0 to 1 from color palette
    relative_temperature is np.array
    """
    # read palette
    if "color_palette" in simulation_data:
        palette_name = simulation_data['color_palette']
    else:
        palette_name = 'e95_rainbow_hc'
    palette_img = Image.open(COLOR_DIR/f'{palette_name}.jpg')
    color_list = [
        palette_img.getpixel((627, k)) for k in range(423, 29, -1)]
    # assign scale
    color_len = len(color_list)
    # apply palette
    range_list = (simulation_data["ambient_temperature"], np.max(array2d)) #np.min(array2d)
    color_count = np.interp(array2d, range_list, (0, color_len-1)).astype(np.uint16)
    for i in range(array_shape[0]):
        for j in range(array_shape[1]):
            color_index = color_count[i,j]
            thermal_img.putpixel((j,i), color_list[color_index])
    # resize to nearest
    thermal_img = thermal_img.resize(
        (simulation_data["baseplate_width"], simulation_data["baseplate_height"]),
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
            outline=color_list[0],
            width=1,
        )
    thermal_img = thermal_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    thermal_img.save(file_path)
    print(f"See figure {file_path}")
    return True

def run_simulation():
    # cartesian coordinate grid
    """y_list = np.arange(0, simulation_data["baseplate_width"])
    z_list = np.arange(0, simulation_data["baseplate_height"])
    y_val, z_val = np.meshgrid(y_list, z_list)"""
    out_dir = LAST_DESIGN.parent / (LAST_DESIGN.name.split(('.'))[0] +'-'+str(int(datetime.now().timestamp())))
    out_dir.mkdir(exist_ok=True)
    with open(out_dir/LAST_DESIGN.name, "w") as writer:
        json.dump(simulation_data, writer)
    
    mm_step = np.array(
        (simulation_data["baseplate_width"], simulation_data["baseplate_height"])
    ).astype(float)
    # check heat sources
    heat_count = 0
    source_dict = simulation_data["source_layout"]
    for k in source_dict:
        heat_dict = source_dict[k]
        rect_data = np.array(heat_dict["rect"])
        yz_bl_h = rect_data[:2]
        yz_tr_h = yz_bl_h + rect_data[2:]
        if np.any(yz_bl_h < (0, 0)) or np.any(yz_tr_h > yz_tr_h):
            messagebox.showerror("RANGE SOURCE", f"Source {k} out of range.")
            return False
        """
        for k2 in source_dict:
            if k2 != k:
                rect_data2 = np.array(source_dict[k2]["rect"])
                yz_bl_2 = rect_data2[:2]
                yz_tr_2 = yz_bl_2 + rect_data2[2:]"""
        heat_count += heat_dict["power"]
    if heat_count < 1:  # W
        messagebox.showerror("LOW SOURCE", f"Source power too low")
        return False
    # heat transfer
    conv_type = simulation_data["flow_conditions"]
    if conv_type == 'natural_convection':
        hs_key = simulation_data["heatsink"]
        dt_experimental = HEATSINK_DATA[hs_key][conv_type]['steady_temperature']-HEATSINK_DATA[hs_key][conv_type]['ambient_temperature']
    average_convection = scaled_convection(simulation_data["baseplate_height"])
    print(f"Rth_cv={round(average_convection,4)} K/W")
    tamb_list = np.array(simulation_data["ambient_temperature"])
    telm_list = tamb_list + 60
    conductivity_yz = HEATSINK_DATA[simulation_data["heatsink"]]["thermal_conductivity"]*HEATSINK_DATA[simulation_data["heatsink"]]["baseplate_thickness"]
    conductivity_x = HEATSINK_DATA[simulation_data["heatsink"]]["thermal_conductivity"]/HEATSINK_DATA[simulation_data["heatsink"]]["baseplate_thickness"]
    # loop
    last_stamp = datetime.now()
    map_partition = 0
    ryz_cnt = np.ones(2) # 1 row vertically, 1 column horizontally
    while map_partition < 20 and (datetime.now() - last_stamp).total_seconds() < SIMULATION_SECONDS:
        print(f"PARTITION #{map_partition} " + 60 * "#")
        # loop variables update and reshape
        last_stamp = datetime.now()
        if map_partition > 0:
            if mm_step[0] < mm_step[1]:
                # split vertically, upper and lower part
                # TEMPERATURE reshape : shape is 1 x matrix_size
                mm_step /= (1,2) # split cell height
                ryz_cnt *= (2,1) # 2 rows, 1 column
                tamb_list = np.repeat(tamb_list,2, axis=0).reshape(ryz_cnt)
                telm_list = np.repeat(telm_list,2, axis=0).reshape(ryz_cnt)
            else:
                # split horizontally, left and right part
                mm_step /= (2,1) # split cell width
                ryz_cnt *= (1,2) # 1 row, 2 columns
                tamb_list = np.repeat(tamb_list,2).reshape(ryz_cnt)
                telm_list = np.repeat(telm_list,2).reshape(ryz_cnt)
        ryz_cnt = ryz_cnt.astype(int)
        matrix_size = np.prod(ryz_cnt)
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
        # CONVECTION LIST : shape is 1 x matrix_size
        # compute for z: row 1 to last row
        z_list = (np.arange(ryz_cnt[0]) + 1) * mm_step[1]
        convection_list = scaled_convection(z_list) * np.maximum(z_list, 100)
        conv_adm_list = convection_list**-1*ryz_cnt[0]*mm_step[1]
        # admittance of first row to last row
        conv_adm_list *= average_convection**-1/np.average(conv_adm_list)/ryz_cnt[0]
        # repeat for all columns
        conv_adm_list = np.repeat(conv_adm_list, ryz_cnt[1])/ryz_cnt[1]
        conv_adm_list = conv_adm_list.reshape(ryz_cnt)
        #print(f'Total convection sum {average_convection * np.sum(conv_adm_list)}')
        # repeat for all y, no need to reshape
        rel_th = np.round((np.min(conv_adm_list), np.max(conv_adm_list)), 2)*matrix_size*average_convection
        print(
            f"Rth_cv avg is {round(np.sum(conv_adm_list)**-1, 4)} K/W, in range {rel_th}"
        )
        # loop until error gets below 10C divided by all the elements
        simulation_count = 0
        dt_error = 100  #
        while dt_error > 3 and simulation_count < MAXIMUM_ITERATIONS:
            #print(f"LOOP {simulation_count} t_error={round(dt_error,6)}") # + 60 * "#"
            # RADIATION ADMITTANCE LIST: shape is 1 x matrix_size W/K
            rad_adm_list = (
                0.9
                * 5.67e-8
                * np.prod(mm_step)
                * 1e-6
                * ((telm_list + 273.15) ** 4 - (273.15 + tamb_list) ** 4)
            ) / (telm_list - tamb_list)
            # rel_th = np.min(rad_adm_list) / np.max(rad_adm_list)
            # print(f"Rth_rad avg is {round(np.sum(rad_adm_list)**-1, 4)} K/W, from {int(rel_th*100)}% to 100%")
            cond_res = (conductivity_x*np.prod(mm_step)*1E-6)**-1
            # CONVECTION + RADIATION admittance in W/K
            if conv_type == 'natural_convection':
                conv_new_list = conv_adm_list*((telm_list-tamb_list)/dt_experimental)**.25
            else:
                conv_new_list = conv_adm_list
            resistance_list = cond_res+(rad_adm_list.ravel() + conv_new_list.ravel())**-1
            admittance_matrix = np.eye(matrix_size) * resistance_list.ravel()**-1
            # CONDUCTION singular matrix in W/K
            conduction_matrix = 0 #matrix_size x matrix_size
            if map_partition > 0:
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
                    #dtr = gt/(telm_list/2 +t_loop/2 - tamb_list)
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
            tnew_list = tamb_list + (resistance_matrix @ (pyz_list.ravel())).reshape(ryz_cnt)
            # limit
            tnew_list = np.maximum(tamb_list + 1, tnew_list)
            dt_error = np.max(np.abs(tnew_list - telm_list))
            tnew_list = np.minimum(tamb_list + 199, tnew_list)
            # recalculate error
            simulation_count += 1
            #print(telm_list.astype(np.uint8))
            telm_list = np.round(tnew_list)
            if np.any(np.isnan(telm_list)):
                print(f"T max is {np.max(telm_list)}ºC")
                return False
        if print_help and (np.min(ryz_cnt)>1.5):
            print('POWER GENERATION MATRIX (W)')
            print(pyz_list.astype(int))
            print('POWER TROUGH CONVECTION (W)')
            print(conv_adm_list.reshape(ryz_cnt)*(telm_list-tamb_list).astype(int))
            print('POWER TROUGH RADIATION (W)')
            print(rad_adm_list.reshape(ryz_cnt)*(telm_list-tamb_list).astype(int))
            print('POWER TROUGH CONDUCTION (W)')
            print(rad_adm_list.reshape(ryz_cnt)*(telm_list-tamb_list).astype(int))
            #print('POWER TRANSFER TROUGH CONDUCTION / CONVECTION AND RADIATION')
            #print(np.round(np.reshape(np.diag(conduction_matrix)/np.diag(admittance_matrix), ryz_cnt), 1))
            print('TEMPERATURE OF ELEMENTS')
            print(telm_list.astype(np.uint8))
        rth_avg = np.average(resistance_matrix)
        print(f"Rth loop is {round(rth_avg, 4)} K/W, Pyz loop = {round(np.sum(pyz_list))} W")
        print(f"T in {int(np.min(telm_list))}ºC ... {np.max(telm_list)}ºC")
        map_partition += 1
        # plot
        plot_path = out_dir / f"loop{map_partition}.png"
        if np.min(ryz_cnt)>=2:
            plot_array(
                np.reshape(telm_list, ryz_cnt),
                plot_path
            )
        if simulation_count + 1 > MAXIMUM_ITERATIONS:
            print(f"MAXIMUM ITERATIONS REACH WITHOUT CONVERGING: dt_error={dt_error}")
            return False
    if plot_path.is_file():
        startfile(plot_path)
    print("END OF SIMULATION" + 60 * "#")
    return True


def open_simulation(prj_path=None):
    if prj_path is None:
        # open
        prj_path = Path(
            tk_filedialog.askopenfilename(
                initialdir=WORSPACE_DIR,
                title="Select project 0XXXXX.json file",
                filetypes=(("json files", "*.json"), ("all files", "*.*")),
            )
        )
    if not prj_path.is_file():
        return False
    global simulation_data, LAST_DESIGN
    with prj_path.open("r") as reader:
        simulation_data = json.load(reader)
    LAST_DESIGN = prj_path
    reset_main()
    title = simulation_data["title"]
    ttk.Label(main_frame, text=f"HEATSINK THERMAL SIMULATION {title}").pack(
        padx=20, pady=20
    )
    grid_frame = ttk.LabelFrame(main_frame, text=f"Heatsink configuration")
    grid_frame.pack(fill=tk.BOTH, padx=20, pady=20)

    row_id = 0
    ttk.Label(grid_frame, text="Model").grid(row=row_id, column=0, padx=10)
    model_combo = ttk.Combobox(grid_frame)
    model_combo["state"] = "readonly"
    model_list = list(HEATSINK_DATA)
    model_combo["values"] = model_list
    model_combo.set(model_list[0])
    model_combo.grid(row=row_id, column=1, padx=10)

    row_id += 1
    ttk.Label(grid_frame, text="Width").grid(row=row_id, column=0, padx=10)
    y_combo = ttk.Combobox(grid_frame)
    y_combo["values"] = list(range(300, 1000, 50))
    y_combo.set(simulation_data["baseplate_width"])
    y_combo.grid(row=row_id, column=1, padx=10, pady=10)
    ttk.Label(grid_frame, text="mm").grid(row=row_id, column=2, padx=10)

    row_id += 1
    ttk.Label(grid_frame, text="Height").grid(row=row_id, column=0, padx=10)
    z_combo = ttk.Combobox(grid_frame)
    z_combo["values"] = list(range(300, 1000, 50))
    z_combo.set(simulation_data["baseplate_height"])
    z_combo.grid(row=row_id, column=1, padx=10, pady=10)
    ttk.Label(grid_frame, text="mm").grid(row=row_id, column=2, padx=10)

    row_id += 1
    ttk.Label(grid_frame, text="Ambient").grid(row=row_id, column=0, padx=10)
    tamb_combo = ttk.Combobox(grid_frame)
    tamb_combo["values"] = list(range(50, 90, 10))
    tamb_combo.set(simulation_data["ambient_temperature"])
    tamb_combo.grid(row=row_id, column=1, padx=10, pady=10)
    ttk.Label(grid_frame, text="ºC").grid(row=row_id, column=2, padx=10)

    row_id += 1
    ttk.Label(grid_frame, text="Source layout").grid(row=row_id, column=0, padx=10)
    ttk.Button(grid_frame, text="EDIT", command=edit_layout).grid(
        row=row_id, column=1, padx=10, pady=10
    )

    row_id += 1
    ttk.Label(grid_frame, text="Color palette").grid(row=row_id, column=0, padx=10)
    color_combo = ttk.Combobox(grid_frame)
    palette_list = [p.name for p in COLOR_DIR.glob('*.*')]
    if "color_palette" in simulation_data:
        color_combo.set(simulation_data['color_palette'])
    elif len(palette_list)>0:
        color_combo.set(palette_list[0])
    color_combo["values"] = [p.name.split('.')[0] for p in COLOR_DIR.glob('*.*')]
    color_combo['state']='readonly'
    color_combo.grid(row=row_id, column=1, padx=10, pady=10)
    def open_palette():
        startfile(COLOR_DIR/color_combo.get())
    ttk.Button(grid_frame, text="OPEN", command=open_palette).grid(
        row=row_id, column=2, padx=10, pady=10
    )

    def run_sim():
        global simulation_data
        simulation_data["baseplate_width"] = int(y_combo.get())
        simulation_data["baseplate_height"] = int(z_combo.get())
        simulation_data["heatsink"] = model_combo.get()
        simulation_data["ambient_temperature"] = int(tamb_combo.get())
        simulation_data['color_palette'] = color_combo.get()
        return run_simulation()

    ttk.Button(main_frame, text="RUN SIMULATION", command=run_sim).pack(
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
        global simulation_data
        simulation_data["baseplate_width"] = int(y_combo.get())
        simulation_data["baseplate_height"] = int(z_combo.get())
        simulation_data["heatsink"] = model_combo.get()
        simulation_data["ambient_temperature"] = int(tamb_combo.get())
        simulation_data['color_palette'] = color_combo.get()
        with open(file_path, "w") as writer:
            json.dump(simulation_data, writer)
        return True

    ttk.Button(main_frame, text="SAVE CONFIGURATION", command=save_simulation).pack(
        padx=20, pady=20
    )
    ttk.Button(
        main_frame, text="OPEN A DIFFERENT CONFIGURATION", command=open_simulation
    ).pack(padx=20, pady=20)


if __name__ == "__main__":
    tk_mgr = tk.Tk()
    tk_mgr.geometry("900x600")
    tk_mgr.config(bg="#F8F8FF")
    tk_mgr.title(f"HEATSINK SIMULATOR (version {__version__}). {__copyright__}")

    main_frame = ttk.Frame(tk_mgr)
    main_frame.pack(fill=tk.BOTH, expand=1)

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
        return open_simulation(Path(path_combo.get()))
    
    ttk.Button(grid_frame, text="OPEN", command=open_selected).grid(row=row_id, column=2, padx=10, pady=10)


    ttk.Button(main_frame, text="BROWSE CONFIGURATION", command=open_simulation).pack(
        padx=20, pady=20
    )
    def new_simulation():
        pass
    ttk.Button(main_frame, text="NEW CONFIGURATION", command=new_simulation).pack(
        padx=20, pady=20
    )

    tk_mgr.mainloop()
