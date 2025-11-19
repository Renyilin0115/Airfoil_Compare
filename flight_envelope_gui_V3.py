import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Physical constants (SI)
g = 9.80665
R = 287.058
gamma = 1.4
T0 = 288.15
p0 = 101325.0
L = -0.0065  # K/m
rho0 = 1.225

# Unit conversions
PA_PER_PSF = 47.880258      # 1 psf = 47.880258 Pa
MPS_PER_KT = 0.514444       # 1 kt  = 0.514444 m/s
M_PER_FT = 0.3048           # 1 ft  = 0.3048 m
LBF_TO_N = 4.4482216153     # 1 lbf = 4.448... N
FT2_TO_M2 = 0.09290304      # 1 ft^2 = 0.09290304 m^2
HP_TO_W = 745.699872        # 1 hp (incl. shp) ≈ 745.7 W


def isa_atmosphere(h):
    """ISA troposphere up to 11 km, then simple isothermal extension.
    h [m] -> T [K], p [Pa], rho [kg/m3], a [m/s]
    """
    if h < 0.0:
        h = 0.0
    if h <= 11000.0:
        T = T0 + L * h
        p = p0 * (T / T0) ** (-g / (L * R))
    else:
        T11 = T0 + L * 11000.0
        p11 = p0 * (T11 / T0) ** (-g / (L * R))
        T = T11
        p = p11 * np.exp(-g * (h - 11000.0) / (R * T11))
    rho = p / (R * T)
    a = np.sqrt(gamma * R * T)
    return T, p, rho, a


class NotesGUI:
    def __init__(self, root):
        self.root = root
        root.title("Chapter 4 Flight Envelope (V3)")

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)

        # Tab 1: M-h + stall + Tmax/Pmax + qmax
        self.tab_qmh = ttk.Frame(nb)
        nb.add(self.tab_qmh, text="M–h flight envelope")

        # Tab 2: Tools (q_max, P0, K, CLmax)
        self.tab_tools = ttk.Frame(nb)
        nb.add(self.tab_tools, text="Tools")

        # Tab 3: Heating vs Mach
        self.tab_heat = ttk.Frame(nb)
        nb.add(self.tab_heat, text="Heating vs M")

        self.build_tab_qmh()
        self.build_tab_tools()
        self.build_tab_heat()

    # ---------- Tab 1 : M–h + stall + Tmax/Pmax + qmax ----------

    def build_tab_qmh(self):
        frame = self.tab_qmh
        left = ttk.Frame(frame, padding=8)
        left.pack(side="left", fill="y")

        right = ttk.Frame(frame, padding=8)
        right.pack(side="right", fill="both", expand=True)

        row = 0

        def add(label, default):
            nonlocal row
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w")
            var = tk.StringVar(value=str(default))
            ttk.Entry(left, textvariable=var, width=12).grid(
                row=row, column=1, sticky="w", pady=2
            )
            row += 1
            return var

        # Structural q limit and altitude range (q in Pa)
        self.qmax_pa_var = add("q_max [Pa]", 2.5e4)
        self.hmax_ft_var = add("h_max [ft]", 40000)
        self.nh_var = add("Number of alt points", 80)

        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4
        )
        row += 1

        # Aero + weight
        self.W_lbf_var = add("Weight W [lbf]", 20000)
        self.S_ft2_var = add("Wing area S [ft²]", 300)
        self.CLmax_var = add("CL_max", 1.5)
        self.CD0_var = add("CD0", 0.02)
        self.K_var = add("K", 0.045)

        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4
        )
        row += 1

        # Optional AR (mainly for tools tab)
        self.AR_ft_var = add("Aspect ratio AR (optional)", "")

        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4
        )
        row += 1

        # Engine type
        ttk.Label(left, text="Engine type").grid(row=row, column=0, sticky="w")
        self.engine_type_var = tk.StringVar(value="Turboprop")
        engine_combo = ttk.Combobox(
            left,
            textvariable=self.engine_type_var,
            values=["Turbojet", "Turbofan", "Turboprop", "Piston prop"],
            state="readonly",
            width=12,
        )
        engine_combo.grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        # Jet/Fan parameters
        self.jet_frame_row = row
        self.jet_frame = ttk.LabelFrame(left, text="Jet/Fan parameters")
        jf_row = 0
        ttk.Label(self.jet_frame, text="Sea-level T0 [lbf]").grid(
            row=jf_row, column=0, sticky="w"
        )
        self.T0_lbf_var = tk.StringVar(value="20000")
        ttk.Entry(self.jet_frame, textvariable=self.T0_lbf_var, width=10).grid(
            row=jf_row, column=1, sticky="w", pady=2
        )
        jf_row += 1

        ttk.Label(self.jet_frame, text="Thrust exponent α").grid(
            row=jf_row, column=0, sticky="w"
        )
        self.alphaT_var = tk.StringVar(value="1.0")
        ttk.Entry(self.jet_frame, textvariable=self.alphaT_var, width=10).grid(
            row=jf_row, column=1, sticky="w", pady=2
        )

        # Prop parameters (Turboprop / Piston)
        self.prop_frame = ttk.LabelFrame(left, text="Prop parameters")
        pf_row = 0
        ttk.Label(self.prop_frame, text="Sea-level P0 [shp]").grid(
            row=pf_row, column=0, sticky="w"
        )
        self.P0_shp_var = tk.StringVar(value="4000")
        ttk.Entry(self.prop_frame, textvariable=self.P0_shp_var, width=10).grid(
            row=pf_row, column=1, sticky="w", pady=2
        )
        pf_row += 1

        ttk.Label(self.prop_frame, text="Prop efficiency η_p").grid(
            row=pf_row, column=0, sticky="w"
        )
        self.eta_p_var = tk.StringVar(value="0.8")
        ttk.Entry(self.prop_frame, textvariable=self.eta_p_var, width=10).grid(
            row=pf_row, column=1, sticky="w", pady=2
        )
        pf_row += 1

        ttk.Label(self.prop_frame, text="Power exponent β").grid(
            row=pf_row, column=0, sticky="w"
        )
        self.betaP_var = tk.StringVar(value="1.0")
        ttk.Entry(self.prop_frame, textvariable=self.betaP_var, width=10).grid(
            row=pf_row, column=1, sticky="w", pady=2
        )

        # Place frames; visibility controlled by update_engine_mode
        self.jet_frame.grid(
            row=self.jet_frame_row, column=0, columnspan=2, sticky="ew", pady=4
        )
        self.prop_frame.grid(
            row=self.jet_frame_row, column=0, columnspan=2, sticky="ew", pady=4
        )

        row = self.jet_frame_row + 1

        ttk.Button(
            left, text="Compute & Plot", command=self.compute_qmh
        ).grid(row=row, column=0, columnspan=2, pady=8)

        # Update engine-specific UI
        self.engine_type_var.trace_add("write", self.update_engine_mode)
        self.update_engine_mode()

        # Figure: x-axis Mach
        fig = Figure(figsize=(5, 4), dpi=100)
        self.ax_qmh = fig.add_subplot(111)
        self.ax_qmh.grid(True, linestyle=":")
        self.ax_qmh.set_xlabel("Mach number")
        self.ax_qmh.set_ylabel("Altitude [ft]")

        self.canvas_qmh = FigureCanvasTkAgg(fig, master=right)
        self.canvas_qmh.get_tk_widget().pack(fill="both", expand=True)

    def update_engine_mode(self, *args):
        mode = self.engine_type_var.get()
        if mode in ("Turbojet", "Turbofan"):
            self.jet_frame.grid(
                row=self.jet_frame_row, column=0, columnspan=2, sticky="ew", pady=4
            )
            self.prop_frame.grid_remove()
        else:
            self.prop_frame.grid(
                row=self.jet_frame_row, column=0, columnspan=2, sticky="ew", pady=4
            )
            self.jet_frame.grid_remove()

    def compute_qmh(self):
        # Common inputs
        try:
            qmax_pa = float(self.qmax_pa_var.get())
            hmax_ft = float(self.hmax_ft_var.get())
            nh = int(self.nh_var.get())

            W_lbf = float(self.W_lbf_var.get())
            S_ft2 = float(self.S_ft2_var.get())
            CLmax = float(self.CLmax_var.get())
            CD0 = float(self.CD0_var.get())
            K = float(self.K_var.get())
        except ValueError:
            print("Input error in M–h tab (common).")
            return

        mode = self.engine_type_var.get()

        # Jet/Fan
        if mode in ("Turbojet", "Turbofan"):
            try:
                T0_lbf = float(self.T0_lbf_var.get())
                alpha_T = float(self.alphaT_var.get())
            except ValueError:
                print("Input error in jet/fan parameters.")
                return
            T0_engine = T0_lbf * LBF_TO_N
        else:
            # Prop
            try:
                P0_shp = float(self.P0_shp_var.get())
                eta_p = float(self.eta_p_var.get())
                beta_P = float(self.betaP_var.get())
            except ValueError:
                print("Input error in prop parameters.")
                return
            P0_shaft_W = P0_shp * HP_TO_W

        # Convert
        W = W_lbf * LBF_TO_N
        S = S_ft2 * FT2_TO_M2
        q_pa = qmax_pa

        # Altitude grid
        h_m = np.linspace(0.0, hmax_ft * M_PER_FT, nh)

        M_q = np.zeros_like(h_m)              # qmax line
        M_stall = np.zeros_like(h_m)          # stall line
        M_max = np.full_like(h_m, np.nan)     # max level-speed

        for i, h in enumerate(h_m):
            T, p, rho, a = isa_atmosphere(h)

            if rho <= 0.0 or a <= 0.0:
                M_q[i] = np.nan
            else:
                V_q = np.sqrt(2.0 * q_pa / rho)
                M_q[i] = V_q / a

            if rho <= 0.0:
                M_stall[i] = np.nan
                continue

            # Stall
            V_stall = np.sqrt(2.0 * W / (rho * S * CLmax))
            M_stall[i] = V_stall / a

            # Velocity sweep for max level
            Vmin = max(V_stall * 1.05, 40.0)
            if np.isfinite(M_q[i]):
                Vmax_search = min(1.8 * a, 2.0 * V_q)
            else:
                Vmax_search = 1.8 * a

            if not np.isfinite(Vmax_search) or Vmax_search <= Vmin:
                continue

            V_grid = np.linspace(Vmin, Vmax_search, 300)
            q_grid = 0.5 * rho * V_grid ** 2
            CL_grid = W / (q_grid * S)
            CD_grid = CD0 + K * CL_grid ** 2
            D_grid = q_grid * S * CD_grid

            if mode in ("Turbojet", "Turbofan"):
                T_avail = T0_engine * (rho / rho0) ** alpha_T
                margin = T_avail - D_grid
            else:
                P_avail = eta_p * P0_shaft_W * (rho / rho0) ** beta_P
                P_req = D_grid * V_grid
                margin = P_avail - P_req

            idx = np.where(margin >= 0.0)[0]
            if idx.size > 0:
                V_max = V_grid[idx[-1]]
                M_max[i] = V_max / a

        # Absolute ceiling
        finite = np.isfinite(M_max)
        h_ac_ft = None
        if finite.any():
            h_ac_ft = h_m[finite][-1] / M_PER_FT

        # For plot
        h_ft = h_m / M_PER_FT

        self.ax_qmh.clear()
        self.ax_qmh.grid(True, linestyle=":")
        self.ax_qmh.set_xlabel("Mach number")
        self.ax_qmh.set_ylabel("Altitude [ft]")

        self.ax_qmh.plot(M_stall, h_ft,
                         label="Stall (CL_max)", linewidth=2)
        label_max = "Max level speed (T=D)" if mode in ("Turbojet", "Turbofan") \
                    else "Max level speed (P_avail = P_req)"
        self.ax_qmh.plot(M_max, h_ft,
                         label=label_max, linewidth=2)
        self.ax_qmh.plot(M_q, h_ft,
                         label=f"q = {qmax_pa:.0f} Pa", linestyle="--")

        if h_ac_ft is not None:
            self.ax_qmh.axhline(
                h_ac_ft, color="red", linestyle=":",
                label=f"Absolute ceiling ≈ {h_ac_ft:.0f} ft"
            )
            self.ax_qmh.set_title(
                f"Absolute ceiling ≈ {h_ac_ft:.0f} ft"
            )
        else:
            self.ax_qmh.set_title(
                "No level-flight solution with given engine parameters"
            )

        self.ax_qmh.legend(loc="lower right")
        self.canvas_qmh.draw()

    # ---------- Tab 2 : Tools (q_max, P0, K, CLmax) ----------

    def build_tab_tools(self):
        frame = self.tab_tools
        main = ttk.Frame(frame, padding=8)
        main.pack(fill="both", expand=True)

        main.rowconfigure((0, 1, 2, 3), weight=1)
        main.columnconfigure(0, weight=1)

        # --- q_max helper (h + Vmax -> q[Pa]) ---
        qf = ttk.LabelFrame(main, text="q_max helper (ISA, output in Pa)")
        qf.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        r = 0
        ttk.Label(qf, text="Altitude [ft]").grid(row=r, column=0, sticky="w")
        self.qtool_alt_ft_var = tk.StringVar(value="0")
        ttk.Entry(qf, textvariable=self.qtool_alt_ft_var, width=10).grid(
            row=r, column=1, sticky="w", pady=2
        )
        r += 1

        ttk.Label(qf, text="V_max [m/s]").grid(row=r, column=0, sticky="w")
        self.qtool_Vmax_var = tk.StringVar(value="200")
        ttk.Entry(qf, textvariable=self.qtool_Vmax_var, width=10).grid(
            row=r, column=1, sticky="w", pady=2
        )
        r += 1

        self.qtool_result_var = tk.StringVar(value="q = -- Pa")
        ttk.Button(qf, text="Compute q",
                   command=self.compute_q_tool).grid(
            row=r, column=0, pady=4, sticky="w"
        )
        ttk.Button(qf, text="Send to main q_max",
                   command=self.send_q_to_main).grid(
            row=r, column=1, pady=4, sticky="w"
        )
        r += 1
        ttk.Label(qf, textvariable=self.qtool_result_var).grid(
            row=r, column=0, columnspan=2, sticky="w"
        )
        self.qtool_last_q_pa = None

        # --- P0 helper (T + Vmax + eta_p -> P0[shp]) ---
        pf = ttk.LabelFrame(main, text="P0 helper (from thrust, V_max, η_p)")
        pf.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        r = 0
        ttk.Label(pf, text="Thrust T [N]").grid(row=r, column=0, sticky="w")
        self.ptool_T_N_var = tk.StringVar(value="17400")
        ttk.Entry(pf, textvariable=self.ptool_T_N_var, width=10).grid(
            row=r, column=1, sticky="w", pady=2
        )
        r += 1

        ttk.Label(pf, text="V_max [m/s]").grid(row=r, column=0, sticky="w")
        self.ptool_Vmax_var = tk.StringVar(value="200")
        ttk.Entry(pf, textvariable=self.ptool_Vmax_var, width=10).grid(
            row=r, column=1, sticky="w", pady=2
        )
        r += 1

        ttk.Label(pf, text="Prop efficiency η_p").grid(row=r, column=0, sticky="w")
        self.ptool_eta_var = tk.StringVar(value="0.8")
        ttk.Entry(pf, textvariable=self.ptool_eta_var, width=10).grid(
            row=r, column=1, sticky="w", pady=2
        )
        r += 1

        self.ptool_result_var = tk.StringVar(value="P0 = -- shp")
        ttk.Button(pf, text="Compute P0",
                   command=self.compute_P0_tool).grid(
            row=r, column=0, pady=4, sticky="w"
        )
        ttk.Button(pf, text="Send to main P0",
                   command=self.send_P0_to_main).grid(
            row=r, column=1, pady=4, sticky="w"
        )
        r += 1
        ttk.Label(pf, textvariable=self.ptool_result_var).grid(
            row=r, column=0, columnspan=2, sticky="w"
        )
        self.ptool_last_P0_shp = None

        # --- K helper ---
        kf = ttk.LabelFrame(main, text="K helper (K = 1 / (π e AR))")
        kf.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        r = 0
        ttk.Label(kf, text="Oswald e").grid(row=r, column=0, sticky="w")
        self.ktool_e_var = tk.StringVar(value="0.8")
        ttk.Entry(kf, textvariable=self.ktool_e_var, width=10).grid(
            row=r, column=1, sticky="w", pady=2
        )
        r += 1

        ttk.Label(kf, text="AR (direct, optional)").grid(row=r, column=0, sticky="w")
        self.ktool_AR_var = tk.StringVar(value="")
        ttk.Entry(kf, textvariable=self.ktool_AR_var, width=10).grid(
            row=r, column=1, sticky="w", pady=2
        )
        r += 1

        ttk.Label(kf, text="Span b [ft] (if AR not given)").grid(
            row=r, column=0, sticky="w"
        )
        self.ktool_span_ft_var = tk.StringVar(value="")
        ttk.Entry(kf, textvariable=self.ktool_span_ft_var, width=10).grid(
            row=r, column=1, sticky="w", pady=2
        )
        r += 1

        self.ktool_result_var = tk.StringVar(value="K = --")
        ttk.Button(kf, text="Compute K",
                   command=self.compute_K_tool).grid(
            row=r, column=0, pady=4, sticky="w"
        )
        ttk.Button(kf, text="Send to main K",
                   command=self.send_K_to_main).grid(
            row=r, column=1, pady=4, sticky="w"
        )
        r += 1
        ttk.Label(kf, text="(If AR empty, program uses b and S from main tab.)").grid(
            row=r, column=0, columnspan=2, sticky="w"
        )
        r += 1
        ttk.Label(kf, textvariable=self.ktool_result_var).grid(
            row=r, column=0, columnspan=2, sticky="w"
        )
        self.ktool_last_K = None

        # --- CL_max helper ---
        cf = ttk.LabelFrame(main, text="CL_max helper (load polar DAT if available)")
        cf.grid(row=3, column=0, sticky="nsew", padx=4, pady=4)
        r = 0
        ttk.Label(
            cf,
            text="DAT format: numeric columns, CL in 2nd column (e.g. α, CL).",
        ).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        self.cltool_file_var = tk.StringVar(value="No file loaded")
        ttk.Button(cf, text="Load DAT file",
                   command=self.load_airfoil_dat).grid(
            row=r, column=0, pady=4, sticky="w"
        )
        ttk.Button(cf, text="Send CL_max to main",
                   command=self.send_CL_to_main).grid(
            row=r, column=1, pady=4, sticky="w"
        )
        r += 1
        ttk.Label(cf, textvariable=self.cltool_file_var).grid(
            row=r, column=0, columnspan=2, sticky="w"
        )
        r += 1
        self.cltool_result_var = tk.StringVar(value="CL_max = --")
        ttk.Label(cf, textvariable=self.cltool_result_var).grid(
            row=r, column=0, columnspan=2, sticky="w"
        )
        self.cltool_last_CL = None

    def compute_q_tool(self):
        try:
            alt_ft = float(self.qtool_alt_ft_var.get())
            Vmax = float(self.qtool_Vmax_var.get())  # m/s
        except ValueError:
            print("q tool: input error.")
            return
        h = alt_ft * M_PER_FT
        T, p, rho, a = isa_atmosphere(h)
        q_pa = 0.5 * rho * Vmax * Vmax
        self.qtool_last_q_pa = q_pa
        self.qtool_result_var.set(f"q = {q_pa:.1f} Pa")

    def send_q_to_main(self):
        if self.qtool_last_q_pa is not None:
            self.qmax_pa_var.set(f"{self.qtool_last_q_pa:.3f}")

    def compute_P0_tool(self):
        try:
            T_N = float(self.ptool_T_N_var.get())
            Vmax = float(self.ptool_Vmax_var.get())  # m/s
            eta_p = float(self.ptool_eta_var.get())
        except ValueError:
            print("P0 tool: input error.")
            return
        if eta_p <= 0.0:
            print("P0 tool: η_p must be > 0.")
            return
        P_shaft = T_N * Vmax / eta_p          # W
        P0_shp = P_shaft / HP_TO_W
        self.ptool_last_P0_shp = P0_shp
        self.ptool_result_var.set(f"P0 = {P0_shp:.1f} shp")

    def send_P0_to_main(self):
        if self.ptool_last_P0_shp is not None:
            self.P0_shp_var.set(f"{self.ptool_last_P0_shp:.3f}")

    def compute_K_tool(self):
        try:
            e = float(self.ktool_e_var.get())
        except ValueError:
            print("K tool: e input error.")
            return
        AR_val = None
        # Prefer direct AR input
        if self.ktool_AR_var.get().strip():
            try:
                AR_val = float(self.ktool_AR_var.get())
            except ValueError:
                print("K tool: AR input error.")
                return
        else:
            # Use span + S from main
            if not self.ktool_span_ft_var.get().strip():
                print("K tool: need AR or span.")
                return
            try:
                b_ft = float(self.ktool_span_ft_var.get())
                S_ft2 = float(self.S_ft2_var.get())
            except ValueError:
                print("K tool: span or S input error.")
                return
            if S_ft2 <= 0.0:
                print("K tool: S must be > 0.")
                return
            AR_val = b_ft ** 2 / S_ft2

        if e <= 0.0 or AR_val <= 0.0:
            print("K tool: e and AR must be > 0.")
            return

        K = 1.0 / (np.pi * e * AR_val)
        self.ktool_last_K = K
        self.ktool_result_var.set(f"K = {K:.5f}  (AR = {AR_val:.3f})")

    def send_K_to_main(self):
        if self.ktool_last_K is not None:
            self.K_var.set(f"{self.ktool_last_K:.5f}")

    def load_airfoil_dat(self):
        path = filedialog.askopenfilename(
            title="Select airfoil polar DAT (alpha, CL, ...)",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            cl_vals = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty or header lines
                    if not line:
                        continue
                    # Header lines often contain letters
                    if any(char.isalpha() for char in line):
                        continue

                    parts = line.split()
                    # Need at least 2 numeric columns
                    if len(parts) < 2:
                        continue

                    # Try parsing numeric values
                    try:
                        nums = [float(x) for x in parts]
                    except ValueError:
                        continue

                    # Default CL is second column
                    cl_candidate = nums[1]

                    # Filter out unreasonable CL values
                    if abs(cl_candidate) < 5.0:
                        cl_vals.append(cl_candidate)

            if not cl_vals:
                self.cltool_file_var.set("No valid CL data found")
                self.cltool_result_var.set("CL_max = --")
                return

            CLmax = max(cl_vals)
            self.cltool_last_CL = CLmax
            self.cltool_file_var.set(path)
            self.cltool_result_var.set(f"CL_max = {CLmax:.3f}")

        except Exception as e:
            print("Error reading DAT:", e)


    def send_CL_to_main(self):
        if self.cltool_last_CL is not None:
            self.CLmax_var.set(f"{self.cltool_last_CL:.3f}")

    # ---------- Tab 3 : Heating vs Mach ----------

    def build_tab_heat(self):
        frame = self.tab_heat
        left = ttk.Frame(frame, padding=8)
        left.pack(side="left", fill="y")

        right = ttk.Frame(frame, padding=8)
        right.pack(side="right", fill="both", expand=True)

        row = 0

        def add(label, default):
            nonlocal row
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w")
            var = tk.StringVar(value=str(default))
            ttk.Entry(left, textvariable=var, width=12).grid(
                row=row, column=1, sticky="w", pady=2
            )
            row += 1
            return var

        self.alt_ft_heat_var = add("Altitude [ft]", 30000)
        self.Mmin_heat_var = add("M_min", 2.0)
        self.Mmax_heat_var = add("M_max", 6.0)
        self.nM_heat_var = add("Number of points", 50)
        self.sweep_deg_var = add("Sweep angle λ [deg]", 0.0)
        self.qconv_lim_var = add("q_conv,limit [Btu/ft²·s]", 10.0)

        ttk.Button(left, text="Compute & Plot",
                   command=self.compute_heat).grid(
            row=row, column=0, columnspan=2, pady=6
        )

        fig = Figure(figsize=(5, 4), dpi=100)
        self.ax_heat = fig.add_subplot(111)
        self.ax_heat.grid(True, linestyle=":")
        self.ax_heat.set_xlabel("Mach number")
        self.ax_heat.set_ylabel("q_conv [Btu/ft²·s]")

        self.canvas_heat = FigureCanvasTkAgg(fig, master=right)
        self.canvas_heat.get_tk_widget().pack(fill="both", expand=True)

    def compute_heat(self):
        try:
            alt_ft = float(self.alt_ft_heat_var.get())
            Mmin = float(self.Mmin_heat_var.get())
            Mmax = float(self.Mmax_heat_var.get())
            nM = int(self.nM_heat_var.get())
            sweep_deg = float(self.sweep_deg_var.get())
            qconv_lim = float(self.qconv_lim_var.get())
        except ValueError:
            print("Input error in heating tab.")
            return

        h = alt_ft * M_PER_FT
        T, p, rho, a = isa_atmosphere(h)

        lam = np.deg2rad(sweep_deg)
        cos_lam_15 = np.cos(lam) ** 1.5

        Ms = np.linspace(Mmin, Mmax, nM)
        qconv = []

        for M in Ms:
            V = M * a          # m/s
            V_ft = V / M_PER_FT  # ft/s
            rho_ratio = rho / rho0
            qc = 15.0 * np.sqrt(rho_ratio) * (V_ft / 1000.0) ** 3 * cos_lam_15
            qconv.append(qc)

        qconv = np.array(qconv)

        self.ax_heat.clear()
        self.ax_heat.grid(True, linestyle=":")
        self.ax_heat.set_xlabel("Mach number")
        self.ax_heat.set_ylabel("q_conv [Btu/ft²·s]")
        self.ax_heat.plot(Ms, qconv, label="q_conv(M)")
        self.ax_heat.axhline(qconv_lim, color="red", linestyle="--",
                             label="limit")
        self.ax_heat.legend(loc="upper left")
        self.canvas_heat.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = NotesGUI(root)
    root.mainloop()
