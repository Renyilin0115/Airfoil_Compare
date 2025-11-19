import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 物理常數（SI）
g = 9.80665
R = 287.058
gamma = 1.4
T0 = 288.15
p0 = 101325.0
L = -0.0065  # K/m
rho0 = 1.225

# 單位轉換
PA_PER_PSF = 47.880258      # 1 psf = 47.880258 Pa
MPS_PER_KT = 0.514444       # 1 kt  = 0.514444 m/s
M_PER_FT = 0.3048           # 1 ft  = 0.3048 m
LBF_TO_N = 4.4482216153     # 1 lbf = 4.448... N
FT2_TO_M2 = 0.09290304      # 1 ft^2 = 0.09290304 m^2
HP_TO_W = 745.699872        # 1 hp (含 shp) ≈ 745.7 W


def isa_atmosphere(h):
    """
    ISA 對流層到 11 km，以上簡單等溫延伸
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
        root.title("Chapter 4 Flight Envelope (Notes Tool)")

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)

        # Tab 1: V-h + stall + max thrust/power + qmax
        self.tab_qvh = ttk.Frame(nb)
        nb.add(self.tab_qvh, text="V–h (stall + Tmax/Pmax + qmax)")

        # Tab 2: q & p0 vs Mach
        self.tab_qp = ttk.Frame(nb)
        nb.add(self.tab_qp, text="q, p0 vs M")

        # Tab 3: Heating vs Mach
        self.tab_heat = ttk.Frame(nb)
        nb.add(self.tab_heat, text="Heating vs M")

        self.build_tab_qvh()
        self.build_tab_qp()
        self.build_tab_heat()

    # ---------- Tab 1 : V–h + stall + Tmax/Pmax + qmax ----------

    def build_tab_qvh(self):
        frame = self.tab_qvh
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

        # 結構 q 限制與高度範圍
        self.qmax_psf_var = add("q_max [psf]", 1800)
        self.hmax_ft_var = add("h_max [ft]", 40000)
        self.nh_var = add("Number of alt points", 80)

        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4
        )
        row += 1

        # 氣動與重量
        self.W_lbf_var = add("Weight W [lbf]", 20000)
        self.S_ft2_var = add("Wing area S [ft²]", 300)
        self.CLmax_var = add("CL_max", 1.5)
        self.CD0_var = add("CD0", 0.02)
        self.K_var = add("K", 0.045)

        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4
        )
        row += 1

        # 發動機型式下拉選單
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

        # Jet/Fan 參數
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

        # Prop 參數（Turboprop / Piston）
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

        # 放置位置（先隨便 grid，後面用 update_engine_mode 控制顯示/隱藏）
        self.jet_frame.grid(
            row=self.jet_frame_row, column=0, columnspan=2, sticky="ew", pady=4
        )
        self.prop_frame.grid(
            row=self.jet_frame_row, column=0, columnspan=2, sticky="ew", pady=4
        )

        row = self.jet_frame_row + 1

        ttk.Button(
            left, text="Compute & Plot", command=self.compute_qvh
        ).grid(row=row, column=0, columnspan=2, pady=8)

        # engine type 改變時更新顯示
        self.engine_type_var.trace_add("write", self.update_engine_mode)
        self.update_engine_mode()  # 一開始根據預設型式更新一次

        # 圖
        fig = Figure(figsize=(5, 4), dpi=100)
        self.ax_qvh = fig.add_subplot(111)
        self.ax_qvh.grid(True, linestyle=":")
        self.ax_qvh.set_xlabel("True Airspeed [kt]")
        self.ax_qvh.set_ylabel("Altitude [ft]")

        self.canvas_qvh = FigureCanvasTkAgg(fig, master=right)
        self.canvas_qvh.get_tk_widget().pack(fill="both", expand=True)

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

    def compute_qvh(self):
        # 共通部分
        try:
            qmax_psf = float(self.qmax_psf_var.get())
            hmax_ft = float(self.hmax_ft_var.get())
            nh = int(self.nh_var.get())

            W_lbf = float(self.W_lbf_var.get())
            S_ft2 = float(self.S_ft2_var.get())
            CLmax = float(self.CLmax_var.get())
            CD0 = float(self.CD0_var.get())
            K = float(self.K_var.get())
        except ValueError:
            print("Input error in V–h tab (common).")
            return

        mode = self.engine_type_var.get()

        # Jet/Fan 參數
        if mode in ("Turbojet", "Turbofan"):
            try:
                T0_lbf = float(self.T0_lbf_var.get())
                alpha_T = float(self.alphaT_var.get())
            except ValueError:
                print("Input error in jet/fan parameters.")
                return
            T0_engine = T0_lbf * LBF_TO_N
        else:
            # Prop 參數（Turboprop / Piston）
            try:
                P0_shp = float(self.P0_shp_var.get())
                eta_p = float(self.eta_p_var.get())
                beta_P = float(self.betaP_var.get())
            except ValueError:
                print("Input error in prop parameters.")
                return
            P0_shaft_W = P0_shp * HP_TO_W

        # 單位轉換
        W = W_lbf * LBF_TO_N
        S = S_ft2 * FT2_TO_M2
        q_pa = qmax_psf * PA_PER_PSF

        # 高度網格
        h_m = np.linspace(0.0, hmax_ft * M_PER_FT, nh)

        V_q = np.zeros_like(h_m)              # qmax 線
        V_stall = np.zeros_like(h_m)          # 失速線
        V_max = np.full_like(h_m, np.nan)     # Max level speed（依發動機型式）

        for i, h in enumerate(h_m):
            T, p, rho, a = isa_atmosphere(h)

            # q_max 對應的真空速
            V_q[i] = np.sqrt(2.0 * q_pa / rho)

            # 失速速度
            V_stall[i] = np.sqrt(2.0 * W / (rho * S * CLmax))

            # 若沒有密度或 a 問題，才往下算
            if rho <= 0.0:
                continue

            # 速度掃描範圍
            Vmin = max(V_stall[i] * 1.05, 40.0)
            Vmax_search = min(1.8 * a, 2.0 * V_q[i])  # 上限避免太誇張
            if not np.isfinite(Vmax_search) or Vmax_search <= Vmin:
                continue

            V_grid = np.linspace(Vmin, Vmax_search, 300)
            q_grid = 0.5 * rho * V_grid ** 2
            CL_grid = W / (q_grid * S)
            CD_grid = CD0 + K * CL_grid ** 2
            D_grid = q_grid * S * CD_grid  # 阻力

            if mode in ("Turbojet", "Turbofan"):
                # 推力模式：T_avail(h) 與 D(V)
                T_avail = T0_engine * (rho / rho0) ** alpha_T
                margin = T_avail - D_grid
            else:
                # 功率模式：P_avail(h) 與 P_req(V,h)
                P_avail = eta_p * P0_shaft_W * (rho / rho0) ** beta_P
                P_req = D_grid * V_grid
                margin = P_avail - P_req

            idx = np.where(margin >= 0.0)[0]
            if idx.size > 0:
                V_max[i] = V_grid[idx[-1]]

        # 絕對升限：最後一個有解的高度
        finite = np.isfinite(V_max)
        h_ac_ft = None
        if finite.any():
            h_ac_ft = h_m[finite][-1] / M_PER_FT

        # 轉成繪圖單位
        h_ft = h_m / M_PER_FT
        V_q_kt = V_q / MPS_PER_KT
        V_stall_kt = V_stall / MPS_PER_KT
        V_max_kt = V_max / MPS_PER_KT

        # 繪圖
        self.ax_qvh.clear()
        self.ax_qvh.grid(True, linestyle=":")
        self.ax_qvh.set_xlabel("True Airspeed [kt]")
        self.ax_qvh.set_ylabel("Altitude [ft]")

        self.ax_qvh.plot(V_stall_kt, h_ft,
                         label="Stall (CL_max)", linewidth=2)
        label_max = "Max level speed (T=D)" if mode in ("Turbojet", "Turbofan") \
                    else "Max level speed (P_avail = P_req)"
        self.ax_qvh.plot(V_max_kt, h_ft,
                         label=label_max, linewidth=2)
        self.ax_qvh.plot(V_q_kt, h_ft,
                         label=f"q = {qmax_psf:.0f} psf", linestyle="--")

        if h_ac_ft is not None:
            self.ax_qvh.axhline(
                h_ac_ft, color="red", linestyle=":",
                label=f"Absolute ceiling ≈ {h_ac_ft:.0f} ft"
            )
            self.ax_qvh.set_title(
                f"Absolute ceiling ≈ {h_ac_ft:.0f} ft"
            )
        else:
            self.ax_qvh.set_title(
                "No level-flight solution with given engine parameters"
            )

        self.ax_qvh.legend(loc="lower right")
        self.canvas_qvh.draw()

    # ---------- Tab 2 : q & p0 vs Mach ----------

    def build_tab_qp(self):
        frame = self.tab_qp
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

        self.alt_ft_qp_var = add("Altitude [ft]", 30000)
        self.Mmin_var = add("M_min", 0.2)
        self.Mmax_var = add("M_max", 3.0)
        self.nM_qp_var = add("Number of points", 60)
        self.qmax_psf_qp_var = add("q_max [psf]", 1800)

        ttk.Button(left, text="Compute & Plot",
                   command=self.compute_qp).grid(
            row=row, column=0, columnspan=2, pady=6
        )

        fig = Figure(figsize=(5, 4), dpi=100)
        self.ax_qp = fig.add_subplot(111)
        self.ax_qp.grid(True, linestyle=":")
        self.ax_qp.set_xlabel("Mach number")
        self.ax_qp.set_ylabel("Dynamic pressure q [psf]")

        self.canvas_qp = FigureCanvasTkAgg(fig, master=right)
        self.canvas_qp.get_tk_widget().pack(fill="both", expand=True)

    def compute_qp(self):
        try:
            alt_ft = float(self.alt_ft_qp_var.get())
            Mmin = float(self.Mmin_var.get())
            Mmax = float(self.Mmax_var.get())
            nM = int(self.nM_qp_var.get())
            qmax_psf = float(self.qmax_psf_qp_var.get())
        except ValueError:
            print("Input error in q,p0 tab.")
            return

        h = alt_ft * M_PER_FT
        T, p, rho, a = isa_atmosphere(h)
        Ms = np.linspace(Mmin, Mmax, nM)

        q_psf = []
        p0_psi = []
        for M in Ms:
            V = M * a
            q_pa = 0.5 * rho * V * V
            q_psf.append(q_pa / PA_PER_PSF)

            p0_pa = p * (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (gamma / (gamma - 1.0))
            p0_psi.append(p0_pa / 6894.757)  # Pa -> psi

        q_psf = np.array(q_psf)
        p0_psi = np.array(p0_psi)

        self.ax_qp.clear()
        self.ax_qp.grid(True, linestyle=":")
        self.ax_qp.set_xlabel("Mach number")
        self.ax_qp.set_ylabel("Dynamic pressure q [psf]")
        self.ax_qp.plot(Ms, q_psf, label="q(M)")
        self.ax_qp.axhline(qmax_psf, color="red", linestyle="--", label="q_max")

        ax2 = self.ax_qp.twinx()
        ax2.set_ylabel("Stagnation pressure p0 [psi]")
        ax2.plot(Ms, p0_psi, color="green", linestyle="-.", label="p0(M)")

        lines1, labels1 = self.ax_qp.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        self.ax_qp.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        self.canvas_qp.draw()

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
