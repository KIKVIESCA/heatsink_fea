import json
import tkinter as tk
import tkinter.filedialog as tk_filedialog
import tkinter.ttk as ttk
from datetime import datetime
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

WORSPACE_DIR = Path.home() / "Desktop"
LAST_DESIGN = WORSPACE_DIR / "hola.json"
MAXIMUM_ITERATIONS = 999
FIN_FACTOR = 3.2 / 12.5 * 50 / 20 * 12.5

simulation_data = {}

# mm, K/W
HEATSINK_DATA = {
    "HKS150RH120": {
        "natural_convection": (
            (400, 100, 0.209),
            (400, 200, 0.130),
            (400, 300, 0.096),
            (400, 400, 0.076),
            (400, 500, 0.069),
            (400, 600, 0.063),
            (400, 700, 0.059),
            (500, 100, 0.166),
            (500, 200, 0.103),
            (500, 300, 0.076),
            (500, 400, 0.064),
            (500, 500, 0.055),
            (500, 600, 0.049),
            (500, 700, 0.045),
            (750, 100, 0.118),
            (750, 200, 0.069),
            (750, 300, 0.051),
            (750, 400, 0.042),
            (750, 500, 0.036),
            (750, 600, 0.032),
            (750, 700, 0.030),
        ),  # K/W
        "thermal_conductivity": 237 * 20e-3,  # W/K
    }
}


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
    closest_x = min(abs(yzR[0] - y_loop) for yzR in HEATSINK_DATA[hs_key][conv_type])
    # scale for y
    convection_z = [0]
    convection_R = []
    for y, z, R in HEATSINK_DATA[hs_key][conv_type]:
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
    range_list = (np.min(array2d), np.max(array2d))
    # print(range_list)
    scaled_array = np.interp(array2d, range_list, (0, 255)).astype(np.uint8)
    thermal_img = Image.fromarray(scaled_array, mode="L")

    if True:
        thermal_img = thermal_img.resize(
            (simulation_data["baseplate_width"], simulation_data["baseplate_height"]),
            resample=Image.Resampling.NEAREST,
        ).convert("RGB")
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
                outline=(255, 0, 0),
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
    mm_sim = np.array(
        (simulation_data["baseplate_width"], simulation_data["baseplate_height"])
    )
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
    average_convection = scaled_convection(simulation_data["baseplate_height"])
    print(f"Rth_cv={round(average_convection,4)} K/W")
    tamb_list = np.array(simulation_data["ambient_temperature"])
    telm_list = tamb_list + 60
    # loop
    last_stamp = datetime.now()
    map_partition = 0
    while map_partition < 7 and (datetime.now() - last_stamp).total_seconds() < 30:
        print(f"PARTITION 2**{map_partition}" + 60 * "#")
        # loop variables
        last_stamp = datetime.now()
        ryz_cnt = (np.ones(2) * 2**map_partition).astype(int)
        mm_step = mm_sim / ryz_cnt
        matrix_size = np.prod(ryz_cnt)
        ry, rz = np.meshgrid(np.arange(ryz_cnt[0]), np.arange(ryz_cnt[1]))

        # heat dissipation matrix
        y_rav = ry.ravel() * mm_step[0]
        z_rav = rz.ravel() * mm_step[1]
        yz_list = np.transpose(np.vstack((y_rav, z_rav)))
        pyz_list = np.zeros(matrix_size)
        for heat_dict in simulation_data["source_layout"].values():
            rect_data = np.array(heat_dict["rect"])
            heat_density = heat_dict["power"] / np.prod(rect_data[2:])  # W/mm2
            yz_bl_h = rect_data[:2]
            yz_tr_h = yz_bl_h + rect_data[2:]
            yz_list_bl = yz_list
            yz_list_tr = yz_list + mm_step
            dyz = np.minimum(yz_list_tr, yz_tr_h) - np.maximum(yz_list_bl, yz_bl_h)
            pyz_list += np.prod(np.maximum(dyz, 0), axis=1) * heat_density
        # print(f"Pyz={round(np.sum(pyz_list))} W")
        # TEMPERATURE reshape : shape is 1 x matrix_size
        if map_partition > 0:
            # shape is 1xmatrix_size, instead of ryz_cnt
            tamb_list = np.repeat(tamb_list, 4)
            telm_list = np.repeat(telm_list, 4)
        # CONVECTION LIST : shape is 1 x matrix_size
        # compute for z
        z_list = (np.arange(ryz_cnt[1]) + 1) * mm_step[1]
        convection_list = scaled_convection(z_list) * np.maximum(z_list, 100)
        # print(average_convection * np.average(convection_list**-1))
        convection_list *= average_convection * np.average(convection_list**-1)
        # repeat for all y, no need to reshape
        convection_list = np.repeat(convection_list, ryz_cnt[0]) * matrix_size
        rel_th = np.min(convection_list) / np.max(convection_list)
        print(
            f"Rth_cv avg is {round(np.sum(convection_list**-1)**-1, 4)} K/W, from {int(rel_th*100)}% to 100%"
        )
        # loop until error gets below 10C divided by all the elements
        simulation_count = 0
        dt_error = 100  #
        while dt_error > 3 and simulation_count < MAXIMUM_ITERATIONS:
            print(f"LOOP {simulation_count} t_error={round(dt_error)} " + 60 * "#")
            # RADIATION ADMITTANCE LIST: shape is 1 x matrix_size W/K
            rad_adm_list = (
                0.9
                * 5.67e-8
                * np.prod(mm_step)
                * 1e-6
                * ((telm_list + 273.15) ** 4 - (273.15 + tamb_list) ** 4)
            ) / (telm_list - tamb_list)
            rel_th = np.min(rad_adm_list) / np.max(rad_adm_list)
            # print(f"Rth_rad avg is {round(np.sum(rad_adm_list)**-1, 4)} K/W, from {int(rel_th*100)}% to 100%")
            # CONVECTION + RADIATION admittance in W/K
            admittance_matrix = np.eye(matrix_size) * (
                rad_adm_list + convection_list**-1 * (telm_list / 85) ** 1.2
            )
            if map_partition > 0:
                # CONDUCTION singular matrix in W/K
                conduction_matrix = np.zeros((matrix_size, matrix_size))
                # conduction with adjacent roll and next roll
                # kk for di in (1, ryz_cnt[0]):  # np.arange(1, matrix_size):
                temperature_gradient = []
                for roll_axis in (0, 1):
                    ry2 = np.copy(ry)
                    rz2 = np.copy(rz)
                    if roll_axis == 0:
                        ry2 -= 1
                    else:
                        rz2 -= 1
                    boundary_condition = np.logical_and(ry2 >= 0, rz2 >= 0)
                    if False:
                        print(
                            f"Boundary sum is {np.sum(boundary_condition)}, should be {2**map_partition*(2**map_partition-1)}"
                        )
                    # roll to the next coordinate
                    di = (1 - roll_axis) + roll_axis * ryz_cnt[0]
                    # print(f'Roll elements = {di}')
                    dy = (ry - np.roll(ry, di)) * mm_step[0]
                    dz = (rz - np.roll(rz, di)) * mm_step[1]
                    t_loop = np.roll(telm_list, di)
                    # temperature difference to ambient
                    # np.where(dtr > 1 temperature gradient over difference to ambient, when at least 1 C of difference to ambient is present
                    gt = np.abs(telm_list - t_loop)
                    temperature_gradient = np.concatenate((temperature_gradient, gt))
                    dt = telm_list / 2 + t_loop / 2 - tamb_list
                    dtr = gt / dt
                    if np.any(dtr < -1e-6):
                        print(f"Minimum relative temperature {np.min(dtr)}")
                        return False
                    # compute
                    # dt * contact length (=np.cross product) / distance, values in K
                    dyzr = (mm_step[1] * dy + mm_step[0] * dz) / (dy**2 + dz**2)
                    dyzr = np.where(boundary_condition, dyzr, 0)
                    if np.any(dyzr < -1e-6):
                        print(f"Minimum relative distance {np.min(dyzr)}")
                        assert True
                    if False:  # boudary condition verification: removes negative
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
                        print("EYE MATRIX")
                        print(np.roll(eye1, -di) - eye1 + np.roll(eye2, di) - eye2)
                    # last di elements are excluded
                    dyzr_eye = np.eye(matrix_size, k=di) * dyzr.ravel() * dtr
                    eye2 = np.transpose(dyzr_eye)
                    conduction_matrix += (
                        np.roll(dyzr_eye, -di) - dyzr_eye + np.roll(eye2, di) - eye2
                    )
                if False:
                    print(
                        f"Avg temperature grad between adjacent elements = {np.average(temperature_gradient)} C"
                    )
                # values in W/K
                conduction_matrix *= HEATSINK_DATA[simulation_data["heatsink"]][
                    "thermal_conductivity"
                ]
                # avg_grad = np.average(np.diag(conduction_matrix))
                # print(f"Ycond/Ycvrad = {round(avg_grad/np.average(admittance_matrix), 4)}")
                # CONVECTION + RADIATION admittance in W/K
                admittance_matrix += conduction_matrix
            # RESISTANCE in K/W
            resistance_matrix = np.linalg.inv(admittance_matrix)
            # rth_avg = np.average(resistance_matrix)
            # print(f"Rth loop is {round(rth_avg, 4)} K/W, Pyz loop = {round(np.sum(pyz_list))} W")
            tnew_list = tamb_list + resistance_matrix @ pyz_list
            # limit
            tnew_list = np.maximum(tamb_list + 1, tnew_list)
            dt_error = np.max(np.abs(tnew_list - telm_list))
            tnew_list = np.minimum(tamb_list + 199, tnew_list)
            # recalculate error
            simulation_count += 1
            telm_list = tnew_list
            if np.any(np.isnan(telm_list)):
                print(f"T max is {np.max(telm_list)}ºC")
                return False
        print(f"T in {int(np.min(telm_list))}ºC ... {np.max(telm_list)}ºC")
        map_partition += 1
        # plot
        if map_partition > 1:
            out_dir = WORSPACE_DIR / "figures"
            out_dir.mkdir(exist_ok=True)
            plot_array(
                np.reshape(telm_list, ryz_cnt),
                out_dir / f"loop{map_partition}.png",
            )
        if simulation_count + 1 > MAXIMUM_ITERATIONS:
            print("SIMULATION IS OVER")
            return False
    print("END OF SIMULATION" + 60 * "#")
    return True


def open_simulation():
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

    def run_sim():
        global simulation_data
        simulation_data["baseplate_width"] = int(y_combo.get())
        simulation_data["baseplate_height"] = int(z_combo.get())
        simulation_data["heatsink"] = model_combo.get()
        simulation_data["ambient_temperature"] = int(tamb_combo.get())
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

    ttk.Button(main_frame, text="OPEN CONFIGURATION", command=open_simulation).pack(
        padx=20, pady=20
    )

    tk_mgr.mainloop()
