# AC_Designer.py
# Aircraft & Airfoil Concept Designer
# - Mission-based airfoil evaluator with score rank
# - Supports CSV polars, DAT geometry (Lednicer / Selig style), DAT paste
# - Estimates Mcrit and CD_wave(M), includes CD0_airframe and high-lift modes
# - Auto-logs imported airfoils into Airfoil_log/

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import math
import csv
import datetime
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# -----------------------------
# Data classes
# -----------------------------

@dataclass
class Phase:
    name: str
    altitude_ft: float
    speed_kt: float

@dataclass
class PhaseResult:
    name: str
    h_m: float
    tas: float
    M: float
    Re: float
    q: float
    CL_req: float

@dataclass
class AirfoilModel:
    name: str
    kind: str            # 'geom' or 'model'
    tc: float            # thickness ratio
    clmax_clean: float
    cm0: float
    cd0: float
    k_ind: float         # induced drag factor
    source: str          # 'builtin','csv','dat','paste'
    csv_rows: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

@dataclass
class AirfoilScore:
    name: str
    score: float
    ld_cruise: float
    stall_min: float
    tc: float
    cm0_abs: float


# -----------------------------
# Utility functions
# -----------------------------

def isa_atmosphere(h_m: float):
    """Simple ISA up to ~11 km."""
    T0 = 288.15
    p0 = 101325.0
    rho0 = 1.225
    a0 = 340.3
    lapse = -0.0065
    g = 9.80665
    R = 287.05

    if h_m < 11000.0:
        T = T0 + lapse * h_m
        p = p0 * (T / T0) ** (-g / (lapse * R))
    else:
        T = 216.65
        p11 = p0 * (T / T0) ** (-g / (lapse * R))
        p = p11 * math.exp(-g * (h_m - 11000.0) / (R * T))
    rho = p / (R * T)
    a = math.sqrt(1.4 * R * T)
    return T, p, rho, a


def ft_to_m(x): return x * 0.3048
def kt_to_mps(v_kt): return v_kt * 0.514444
def lb_to_N(x): return x * 4.4482216
def ft2_to_m2(x): return x * 0.092903
def ft_to_m_chord(x): return x * 0.3048


def now_stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_log_dir():
    d = Path("Airfoil_log")
    d.mkdir(exist_ok=True)
    return d


def log_airfoil_raw(name: str, raw_text: str, meta: dict):
    base = ensure_log_dir()
    safe_name = "".join(c for c in name if c.isalnum() or c in " _-")[:40]
    folder = base / f"{now_stamp()}_{safe_name}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "airfoil.dat").write_text(raw_text, encoding="utf-8")
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def log_airfoil_file(src_path: Path, name: str, meta: dict):
    base = ensure_log_dir()
    safe_name = "".join(c for c in name if c.isalnum() or c in " _-")[:40]
    folder = base / f"{now_stamp()}_{safe_name}"
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / src_path.name
    try:
        shutil.copy2(src_path, dst)
    except Exception:
        pass
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


# -----------------------------
# Aerodynamic models
# -----------------------------

def phase_flap_mode(phase_name: str) -> str:
    n = phase_name.lower()
    if "takeoff" in n or "t/o" in n or "to " in n:
        return "TO"
    if "approach" in n or "landing" in n or "ldg" in n:
        return "LDG"
    return "clean"


def clmax_for_phase(af: AirfoilModel, phase_name: str) -> float:
    base = af.clmax_clean or 1.6
    mode = phase_flap_mode(phase_name)
    if mode == "clean":
        mult = 1.0
    elif mode == "TO":
        mult = 1.4
    elif mode == "LDG":
        mult = 1.8
    else:
        mult = 1.0
    return base * mult


def flap_drag_increment(mode: str) -> float:
    if mode == "clean":
        return 0.0
    if mode == "TO":
        return 0.010
    if mode == "LDG":
        return 0.040
    return 0.0


def estimate_mcrit(tc: float, cl: float, sweep_deg: float) -> float:
    tc = tc if tc is not None and tc > 0 else 0.12
    cl = cl if cl is not None else 0.0
    sweep_rad = math.radians(sweep_deg or 0.0)
    Mcrit_unswept = 0.94 - 1.9 * tc - 0.3 * cl
    Mcrit_unswept = max(0.5, Mcrit_unswept)
    beta = abs(math.cos(sweep_rad))
    if beta < 0.2:
        beta = 0.2
    return Mcrit_unswept / beta


def wave_drag_increment(M: float, Mcrit: float, tc: float) -> float:
    tc = tc if tc is not None and tc > 0 else 0.12
    if M <= Mcrit:
        return 0.0
    k_wave = 4.0 * (tc ** 2)
    return k_wave * (M - Mcrit) ** 2


def cd_from_csv(csv_rows, cl_target):
    if not csv_rows:
        return None
    best = None
    best_d = 1e9
    for row in csv_rows:
        try:
            cl = float(row.get("Cl") or row.get("CL") or row.get("cl"))
            cd = float(row.get("Cd") or row.get("CD") or row.get("cd"))
        except Exception:
            continue
        d = abs(cl - cl_target)
        if d < best_d:
            best_d = d
            best = cd
    return best


def cd_decomposition(af: AirfoilModel,
                     cl: float,
                     M: float,
                     Re: float,
                     AR: float,
                     e_oswald: float,
                     CD0_airframe: float,
                     mode: str,
                     tc: float,
                     sweep_deg: float):
    cl = cl or 0.0
    AR = AR if AR > 0 else 8.0
    e_oswald = e_oswald if e_oswald > 0 else 0.8

    cd0_wing = af.cd0 or 0.010
    k_ind = af.k_ind or (1.0 / (math.pi * e_oswald * AR))
    cd_ind = k_ind * cl ** 2

    dcd0_flap = flap_drag_increment(mode)

    Mcrit = estimate_mcrit(tc, cl, sweep_deg)
    cd_wave = wave_drag_increment(M, Mcrit, tc)

    cm0 = af.cm0 or 0.0
    cd_trim = 0.0  # placeholder

    cd_total = (cd0_wing + cd_ind +
                CD0_airframe + dcd0_flap +
                cd_wave + cd_trim)

    return {
        "total": cd_total,
        "cd0_wing": cd0_wing,
        "cd0_airframe": CD0_airframe,
        "cd0_flap": dcd0_flap,
        "cd_ind": cd_ind,
        "cd_wave": cd_wave,
        "cd_trim": cd_trim,
        "Mcrit": Mcrit,
        "cm0": cm0,
    }


# -----------------------------
# Geometry estimators from DAT
# -----------------------------

def parse_dat_geometry(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None, []
    name = lines[0]
    pts = []
    for l in lines[1:]:
        parts = l.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except Exception:
            continue
        pts.append((x, y))
    return name, pts


def estimate_tc_from_pts(pts):
    if not pts:
        return 0.12
    ys = [p[1] for p in pts]
    t = max(ys) - min(ys)
    if t <= 0:
        t = 0.12
    return t


def build_surrogate_from_geom(name: str, pts):
    tc = estimate_tc_from_pts(pts)
    clmax_clean = 1.4 + 1.5 * (tc - 0.12)  # crude trend
    clmax_clean = max(1.2, min(2.2, clmax_clean))
    cd0 = 0.008 + 0.10 * (tc - 0.12) ** 2
    cd0 = max(0.006, cd0)
    cm0 = -0.05 - 0.5 * (tc - 0.12)
    k_ind = 0.045
    return AirfoilModel(
        name=name,
        kind="geom",
        tc=tc,
        clmax_clean=clmax_clean,
        cm0=cm0,
        cd0=cd0,
        k_ind=k_ind,
        source="dat",
        csv_rows=[],
        meta={"tc": tc}
    )


# -----------------------------
# GUI Application
# -----------------------------

class ACDesignerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AC Designer (Aircraft & Airfoil Planner)")
        self.geometry("1280x720")

        # Aircraft inputs (Imperial)
        self.var_weight_lb = tk.StringVar(value="70000")
        self.var_S_ft2 = tk.StringVar(value="861")
        self.var_c_ref_ft = tk.StringVar(value="8.7")
        self.var_AR_hint = tk.StringVar(value="14")
        self.var_tc = tk.StringVar(value="0.12")
        self.var_sweep_deg = tk.StringVar(value="25")
        self.var_CD0_airframe = tk.StringVar(value="0.015")
        self.var_e_oswald = tk.StringVar(value="0.80")

        self.phases = []
        self.phase_results = []
        self.airfoils = []
        self.scores = []

        self._build_ui()

    # ---------------- UI layout ----------------

    def _build_ui(self):
        main = ttk.Notebook(self)
        main.pack(fill="both", expand=True)

        self.page_mission = ttk.Frame(main)
        self.page_geom = ttk.Frame(main)
        main.add(self.page_mission, text="Mission & Airfoil Planner")
        main.add(self.page_geom, text="Geometry & Layout (stub)")

        self._build_mission_page(self.page_mission)
        self._build_geom_page(self.page_geom)

    def _build_mission_page(self, parent):
        top = ttk.Frame(parent)
        top.pack(side="top", fill="x")

        ttk.Label(top, text="Weight [lb]").grid(row=0, column=0, padx=2, pady=2, sticky="e")
        ttk.Entry(top, textvariable=self.var_weight_lb, width=10).grid(row=0, column=1, padx=2)
        ttk.Label(top, text="Wing area [ft²]").grid(row=0, column=2, padx=2, sticky="e")
        ttk.Entry(top, textvariable=self.var_S_ft2, width=8).grid(row=0, column=3, padx=2)
        ttk.Label(top, text="Ref chord [ft]").grid(row=0, column=4, padx=2, sticky="e")
        ttk.Entry(top, textvariable=self.var_c_ref_ft, width=8).grid(row=0, column=5, padx=2)
        ttk.Label(top, text="AR (hint)").grid(row=0, column=6, padx=2, sticky="e")
        ttk.Entry(top, textvariable=self.var_AR_hint, width=6).grid(row=0, column=7, padx=2)

        ttk.Label(top, text="t/c").grid(row=1, column=0, padx=2, sticky="e")
        ttk.Entry(top, textvariable=self.var_tc, width=6).grid(row=1, column=1, padx=2)
        ttk.Label(top, text="Λ25 [deg]").grid(row=1, column=2, padx=2, sticky="e")
        ttk.Entry(top, textvariable=self.var_sweep_deg, width=6).grid(row=1, column=3, padx=2)
        ttk.Label(top, text="CD0_airframe").grid(row=1, column=4, padx=2, sticky="e")
        ttk.Entry(top, textvariable=self.var_CD0_airframe, width=8).grid(row=1, column=5, padx=2)
        ttk.Label(top, text="Oswald e").grid(row=1, column=6, padx=2, sticky="e")
        ttk.Entry(top, textvariable=self.var_e_oswald, width=6).grid(row=1, column=7, padx=2)

        # Phases + buttons
        frame_mid = ttk.Frame(parent)
        frame_mid.pack(side="top", fill="both", expand=True)

        left = ttk.Frame(frame_mid)
        left.pack(side="top", fill="both", expand=True)

        ttk.Label(left, text="Phases (Imperial)").pack(anchor="w")

        self.list_phases = tk.Listbox(left, height=6)
        self.list_phases.pack(fill="x", padx=4)

        btn_row = ttk.Frame(left)
        btn_row.pack(fill="x", padx=4, pady=2)
        ttk.Button(btn_row, text="Add Phase", command=self.add_phase).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Edit Phase", command=self.edit_phase).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Delete Phase", command=self.delete_phase).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Compute Envelope", command=self.compute_envelope).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Export Mission CSV", command=self.export_mission_csv).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Export Report", command=self.export_report).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Save Mission JSON", command=self.save_mission_json).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Load Mission JSON", command=self.load_mission_json).pack(side="left", padx=2)

        # Phase results tree
        ttk.Label(left, text="Phase Results (SI)").pack(anchor="w")
        cols = ("name", "h_m", "V", "M", "Re", "q", "CL_req")
        self.tree_phases = ttk.Treeview(left, columns=cols, show="headings", height=6)
        headings = {
            "name": "name",
            "h_m": "h[m]",
            "V": "V[m/s]",
            "M": "M",
            "Re": "Re",
            "q": "q[Pa]",
            "CL_req": "CL_req"
        }
        for c in cols:
            self.tree_phases.heading(c, text=headings[c])
        self.tree_phases.column("name", width=80)
        self.tree_phases.column("h_m", width=80)
        self.tree_phases.column("V", width=80)
        self.tree_phases.column("M", width=60)
        self.tree_phases.column("Re", width=100)
        self.tree_phases.column("q", width=80)
        self.tree_phases.column("CL_req", width=80)
        self.tree_phases.pack(fill="x", padx=4, pady=2)

        # Airfoil evaluator
        bottom = ttk.Frame(parent)
        bottom.pack(side="top", fill="both", expand=True)

        ttk.Label(bottom, text="Airfoil Evaluator").pack(anchor="w")

        mid_bottom = ttk.Frame(bottom)
        mid_bottom.pack(fill="both", expand=True)

        # airfoil list
        cols_af = ("name", "type")
        self.tree_airfoils = ttk.Treeview(
            mid_bottom, columns=cols_af, show="headings", height=7
        )
        self.tree_airfoils.heading("name", text="name")
        self.tree_airfoils.heading("type", text="type")
        self.tree_airfoils.column("name", width=180)
        self.tree_airfoils.column("type", width=60)
        self.tree_airfoils.pack(side="left", fill="both", expand=True, padx=4, pady=2)

        # airfoil buttons
        btn_af = ttk.Frame(mid_bottom)
        btn_af.pack(side="left", fill="y", padx=4)
        ttk.Button(btn_af, text="Add Built-in Models", command=self.add_builtin_models).pack(fill="x", pady=2)
        ttk.Button(btn_af, text="Import Polar CSV...", command=self.import_polar_csv).pack(fill="x", pady=2)
        ttk.Button(btn_af, text="Import Airfoil DAT...", command=self.import_airfoil_dat).pack(fill="x", pady=2)
        ttk.Button(btn_af, text="Paste DAT text...", command=self.paste_dat_text).pack(fill="x", pady=2)
        ttk.Button(btn_af, text="Remove Selected", command=self.remove_selected_airfoil).pack(fill="x", pady=2)
        ttk.Button(btn_af, text="Score Rank (from Data)", command=self.score_rank).pack(fill="x", pady=2)

        # score table
        cols_sc = ("rank", "name", "score", "ld_cruise", "stall_min", "tc", "cm0")
        self.tree_scores = ttk.Treeview(
            bottom, columns=cols_sc, show="headings", height=7
        )
        headers_sc = {
            "rank": "rank",
            "name": "name",
            "score": "score(0-100)",
            "ld_cruise": "L/D@cruise",
            "stall_min": "stall_min",
            "tc": "t/c",
            "cm0": "|Cm0|"
        }
        for c in cols_sc:
            self.tree_scores.heading(c, text=headers_sc[c])
        self.tree_scores.column("rank", width=50)
        self.tree_scores.column("name", width=150)
        self.tree_scores.column("score", width=100)
        self.tree_scores.column("ld_cruise", width=90)
        self.tree_scores.column("stall_min", width=80)
        self.tree_scores.column("tc", width=60)
        self.tree_scores.column("cm0", width=60)
        self.tree_scores.pack(fill="x", padx=4, pady=2)

        # notes
        ttk.Label(parent, text="Notes").pack(anchor="w")
        self.txt_notes = tk.Text(parent, height=4)
        self.txt_notes.pack(fill="both", expand=False, padx=4, pady=2)

        self.append_note("Initialized AC Designer.\n")

    def _build_geom_page(self, parent):
        # stub: just a placeholder text for now
        ttk.Label(parent, text="Geometry & Layout page (to be implemented)").pack(
            padx=10, pady=10, anchor="w"
        )

    # ---------------- Phase management ----------------

    def refresh_phase_list(self):
        self.list_phases.delete(0, tk.END)
        for ph in self.phases:
            self.list_phases.insert(
                tk.END, f"{ph.name}: alt={ph.altitude_ft} ft, V={ph.speed_kt} kt"
            )

    def add_phase(self):
        dlg = PhaseDialog(self, title="Add Phase")
        self.wait_window(dlg)
        if dlg.result is not None:
            self.phases.append(dlg.result)
            self.refresh_phase_list()

    def edit_phase(self):
        idx = self.list_phases.curselection()
        if not idx:
            messagebox.showinfo("Edit Phase", "Please select a phase.")
            return
        i = idx[0]
        ph = self.phases[i]
        dlg = PhaseDialog(self, title="Edit Phase", phase=ph)
        self.wait_window(dlg)
        if dlg.result is not None:
            self.phases[i] = dlg.result
            self.refresh_phase_list()

    def delete_phase(self):
        idx = self.list_phases.curselection()
        if not idx:
            return
        i = idx[0]
        del self.phases[i]
        self.refresh_phase_list()

    # ---------------- Envelope computation ----------------

    def compute_envelope(self):
        try:
            W_lb = float(self.var_weight_lb.get())
            S_ft2 = float(self.var_S_ft2.get())
            c_ft = float(self.var_c_ref_ft.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid aircraft inputs.")
            return
        if not self.phases:
            messagebox.showinfo("Compute", "Please add at least one phase.")
            return

        W = lb_to_N(W_lb)
        S = ft2_to_m2(S_ft2)
        c = ft_to_m_chord(c_ft)

        self.phase_results = []
        for ph in self.phases:
            h_m = ft_to_m(ph.altitude_ft)
            V = kt_to_mps(ph.speed_kt)
            T, p, rho, a = isa_atmosphere(h_m)
            q = 0.5 * rho * V ** 2
            CL_req = W / (q * S) if q > 0 else 0.0
            M = V / a if a > 0 else 0.0
            mu = 1.7894e-5  # ~ sea-level; acceptable for this level
            Re = rho * V * c / mu if mu > 0 else 0.0
            self.phase_results.append(
                PhaseResult(
                    name=ph.name,
                    h_m=h_m,
                    tas=V,
                    M=M,
                    Re=Re,
                    q=q,
                    CL_req=CL_req,
                )
            )

        # refresh tree
        for i in self.tree_phases.get_children():
            self.tree_phases.delete(i)
        for res in self.phase_results:
            self.tree_phases.insert(
                "",
                tk.END,
                values=(
                    res.name,
                    f"{res.h_m:.0f}",
                    f"{res.tas:.1f}",
                    f"{res.M:.3f}",
                    f"{res.Re:.2e}",
                    f"{res.q:.1f}",
                    f"{res.CL_req:.3f}",
                ),
            )
        self.append_note("Envelope computed.\n")

    # ---------------- Mission import/export ----------------

    def export_mission_csv(self):
        if not self.phase_results:
            messagebox.showinfo("Export", "Compute envelope first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "h[m]", "V[m/s]", "M", "Re", "q[Pa]", "CL_req"])
            for r in self.phase_results:
                w.writerow(
                    [
                        r.name,
                        r.h_m,
                        r.tas,
                        r.M,
                        r.Re,
                        r.q,
                        r.CL_req,
                    ]
                )
        self.append_note(f"Mission CSV exported to {path}\n")

    def export_report(self):
        if not self.phase_results or not self.scores:
            messagebox.showinfo("Report", "Compute envelope and scores first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text", "*.txt")]
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("AC Designer Report\n\n")
            f.write("Aircraft Inputs:\n")
            f.write(f"Weight_lb={self.var_weight_lb.get()}\n")
            f.write(f"Wing area ft2={self.var_S_ft2.get()}\n")
            f.write(f"Ref chord ft={self.var_c_ref_ft.get()}\n")
            f.write(f"AR hint={self.var_AR_hint.get()}\n")
            f.write(f"t/c={self.var_tc.get()}, sweep={self.var_sweep_deg.get()} deg\n")
            f.write(
                f"CD0_airframe={self.var_CD0_airframe.get()}, e={self.var_e_oswald.get()}\n\n"
            )
            f.write("Phases:\n")
            for ph, res in zip(self.phases, self.phase_results):
                f.write(
                    f"{ph.name}: alt={ph.altitude_ft} ft, V={ph.speed_kt} kt, "
                    f"CL_req={res.CL_req:.3f}, M={res.M:.3f}, Re={res.Re:.2e}\n"
                )
            f.write("\nScores:\n")
            for i, sc in enumerate(self.scores, start=1):
                f.write(
                    f"{i}. {sc.name}: score={sc.score:.1f}, "
                    f"L/D@cruise={sc.ld_cruise:.1f}, "
                    f"stall_min={sc.stall_min:.2f}, t/c={sc.tc:.3f}, |Cm0|={sc.cm0_abs:.3f}\n"
                )
        self.append_note(f"Report exported to {path}\n")

    def save_mission_json(self):
        data = {
            "aircraft": {
                "weight_lb": self.var_weight_lb.get(),
                "S_ft2": self.var_S_ft2.get(),
                "c_ref_ft": self.var_c_ref_ft.get(),
                "AR_hint": self.var_AR_hint.get(),
                "tc": self.var_tc.get(),
                "sweep_deg": self.var_sweep_deg.get(),
                "CD0_airframe": self.var_CD0_airframe.get(),
                "e": self.var_e_oswald.get(),
            },
            "phases": [ph.__dict__ for ph in self.phases],
        }
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")]
        )
        if not path:
            return
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.append_note(f"Mission JSON saved to {path}\n")

    def load_mission_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        data = json.load(open(path, "r", encoding="utf-8"))
        ac = data.get("aircraft", {})
        self.var_weight_lb.set(ac.get("weight_lb", "70000"))
        self.var_S_ft2.set(ac.get("S_ft2", "861"))
        self.var_c_ref_ft.set(ac.get("c_ref_ft", "8.7"))
        self.var_AR_hint.set(ac.get("AR_hint", "14"))
        self.var_tc.set(ac.get("tc", "0.12"))
        self.var_sweep_deg.set(ac.get("sweep_deg", "25"))
        self.var_CD0_airframe.set(ac.get("CD0_airframe", "0.015"))
        self.var_e_oswald.set(ac.get("e", "0.80"))
        self.phases = [Phase(**p) for p in data.get("phases", [])]
        self.refresh_phase_list()
        self.append_note(f"Mission JSON loaded from {path}\n")

    # ---------------- Airfoil management ----------------

    def refresh_airfoil_list(self):
        for i in self.tree_airfoils.get_children():
            self.tree_airfoils.delete(i)
        for af in self.airfoils:
            self.tree_airfoils.insert(
                "", tk.END, values=(af.name, af.kind)
            )

    def add_builtin_models(self):
        # simple NACA set with approximate parameters
        builtin = [
            ("NACA 23015 (geom)", 0.15, 1.9, -0.05, 0.010),
            ("NACA 2412 (geom)", 0.12, 1.6, -0.05, 0.010),
            ("NACA 2415 (geom)", 0.15, 1.7, -0.05, 0.011),
            ("NACA 23024 (geom)", 0.24, 2.0, -0.06, 0.012),
            ("NACA 0006 (geom)", 0.06, 1.2, 0.00, 0.008),
        ]
        for name, tc, clmax, cm0, cd0 in builtin:
            af = AirfoilModel(
                name=name,
                kind="geom",
                tc=tc,
                clmax_clean=clmax,
                cm0=cm0,
                cd0=cd0,
                k_ind=0.045,
                source="builtin",
            )
            self.airfoils.append(af)
        self.refresh_airfoil_list()
        self.append_note("Built-in models added.\n")

    def import_polar_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        name = Path(path).stem
        af = AirfoilModel(
            name=name + " (csv)",
            kind="model",
            tc=float(self.var_tc.get() or 0.12),
            clmax_clean=1.8,
            cm0=-0.05,
            cd0=0.010,
            k_ind=0.045,
            source="csv",
            csv_rows=rows,
        )
        self.airfoils.append(af)
        self.refresh_airfoil_list()
        meta = {"source": "csv_polar", "path": path}
        log_airfoil_file(Path(path), name, meta)
        self.append_note(f"Imported CSV polar {path}\n")

    def import_airfoil_dat(self):
        path = filedialog.askopenfilename(
            filetypes=[("DAT", "*.dat"), ("All files", "*.*")]
        )
        if not path:
            return
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        name, pts = parse_dat_geometry(text)
        if name is None:
            messagebox.showerror("Error", "Failed to parse DAT file.")
            return
        af = build_surrogate_from_geom(name, pts)
        self.airfoils.append(af)
        self.refresh_airfoil_list()
        meta = {"source": "dat_file", "path": path, "tc": af.tc}
        log_airfoil_file(Path(path), name, meta)
        self.append_note(f"Imported DAT {path}, t/c≈{af.tc:.3f}\n")
        self.var_tc.set(f"{af.tc:.3f}")

    def paste_dat_text(self):
        dlg = PasteDatDialog(self)
        self.wait_window(dlg)
        if dlg.result is None:
            return
        text = dlg.result
        name, pts = parse_dat_geometry(text)
        if name is None:
            messagebox.showerror("Error", "Failed to parse DAT text.")
            return
        af = build_surrogate_from_geom(name, pts)
        af.source = "paste"
        self.airfoils.append(af)
        self.refresh_airfoil_list()
        meta = {"source": "dat_paste", "tc": af.tc}
        folder = log_airfoil_raw(name, text, meta)
        self.append_note(f"Pasted DAT for {name}, logged to {folder}\n")
        self.var_tc.set(f"{af.tc:.3f}")

    def remove_selected_airfoil(self):
        sel = self.tree_airfoils.selection()
        if not sel:
            return
        idx = self.tree_airfoils.index(sel[0])
        del self.airfoils[idx]
        self.refresh_airfoil_list()

    # ---------------- Scoring ----------------

    def score_rank(self):
        if not self.phase_results:
            messagebox.showinfo("Score", "Compute envelope first.")
            return
        if not self.airfoils:
            messagebox.showinfo("Score", "Add at least one airfoil.")
            return

        try:
            AR_hint = float(self.var_AR_hint.get())
            CD0_airframe = float(self.var_CD0_airframe.get())
            e_oswald = float(self.var_e_oswald.get())
            tc_global = float(self.var_tc.get())
            sweep_deg = float(self.var_sweep_deg.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid AR/CD0/e/tc/sweep.")
            return

        self.scores = []
        notes_lines = []

        # define cruise phase as first one named 'cruise' or index 0
        idx_cruise = 0
        for i, res in enumerate(self.phase_results):
            if "cruise" in res.name.lower():
                idx_cruise = i
                break

        for af in self.airfoils:
            tc = af.tc if af.tc is not None else tc_global
            stall_min = 1e9
            ld_cruise = 0.0
            hard_fail = False

            for ip, (ph, res) in enumerate(zip(self.phases, self.phase_results)):
                cl = res.CL_req
                mode = phase_flap_mode(ph.name)
                clmax_phase = clmax_for_phase(af, ph.name)
                stall_margin = clmax_phase - cl
                stall_min = min(stall_min, stall_margin)

                cd_parts = cd_decomposition(
                    af=af,
                    cl=cl,
                    M=res.M,
                    Re=res.Re,
                    AR=AR_hint,
                    e_oswald=e_oswald,
                    CD0_airframe=CD0_airframe,
                    mode=mode,
                    tc=tc,
                    sweep_deg=sweep_deg,
                )
                cd_total = cd_parts["total"]
                if cd_total <= 0:
                    continue
                LD = cl / cd_total
                if ip == idx_cruise:
                    ld_cruise = LD

                if stall_margin < 0:
                    hard_fail = True

                if ip == idx_cruise:
                    notes_lines.append(
                        f"{af.name} cruise: CL={cl:.3f}, CD0w={cd_parts['cd0_wing']:.3f}, "
                        f"CD0_air={cd_parts['cd0_airframe']:.3f}, CDi={cd_parts['cd_ind']:.3f}, "
                        f"CDwave={cd_parts['cd_wave']:.3f}, L/D={LD:.1f}, Mcrit={cd_parts['Mcrit']:.3f}"
                    )

            if hard_fail:
                score = 0.0
            else:
                # simple scoring: L/D and stall margin
                score_ld = max(0.0, min(50.0, 50.0 * ld_cruise / 25.0))
                score_stall = max(
                    0.0, min(50.0, 50.0 * stall_min / 0.5)
                )  # 0.5 margin -> full
                score = score_ld + score_stall

            sc = AirfoilScore(
                name=af.name,
                score=score,
                ld_cruise=ld_cruise,
                stall_min=stall_min,
                tc=tc,
                cm0_abs=abs(af.cm0 or 0.0),
            )
            self.scores.append(sc)

        self.scores.sort(key=lambda s: s.score, reverse=True)
        for i in self.tree_scores.get_children():
            self.tree_scores.delete(i)
        for rank, sc in enumerate(self.scores, start=1):
            self.tree_scores.insert(
                "",
                tk.END,
                values=(
                    rank,
                    sc.name,
                    f"{sc.score:.1f}",
                    f"{sc.ld_cruise:.1f}",
                    f"{sc.stall_min:.2f}",
                    f"{sc.tc:.3f}",
                    f"{sc.cm0_abs:.3f}",
                ),
            )

        self.append_note("Score rank computed.\n")
        if notes_lines:
            self.append_note(
                "Cruise CD breakdown (top few):\n" + "\n".join(notes_lines[:5]) + "\n"
            )

    # ---------------- Helpers ----------------

    def append_note(self, text):
        self.txt_notes.insert(tk.END, text)
        self.txt_notes.see(tk.END)


# -----------------------------
# Dialogs
# -----------------------------

class PhaseDialog(tk.Toplevel):
    def __init__(self, parent, title="Phase", phase: Phase = None):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.result = None

        self.var_name = tk.StringVar(value=phase.name if phase else "cruise")
        self.var_alt = tk.StringVar(
            value=str(phase.altitude_ft) if phase else "0"
        )
        self.var_speed = tk.StringVar(
            value=str(phase.speed_kt) if phase else "200"
        )

        ttk.Label(self, text="Name").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        ttk.Entry(self, textvariable=self.var_name, width=16).grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(self, text="Altitude [ft]").grid(row=1, column=0, padx=4, pady=4, sticky="e")
        ttk.Entry(self, textvariable=self.var_alt, width=12).grid(row=1, column=1, padx=4, pady=4)

        ttk.Label(self, text="Speed [kt]").grid(row=2, column=0, padx=4, pady=4, sticky="e")
        ttk.Entry(self, textvariable=self.var_speed, width=12).grid(row=2, column=1, padx=4, pady=4)

        btns = ttk.Frame(self)
        btns.grid(row=3, column=0, columnspan=2, pady=6)
        ttk.Button(btns, text="OK", command=self.on_ok).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=4)

        self.grab_set()
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def on_ok(self):
        try:
            name = self.var_name.get().strip() or "phase"
            alt = float(self.var_alt.get())
            spd = float(self.var_speed.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid altitude or speed.")
            return
        self.result = Phase(name=name, altitude_ft=alt, speed_kt=spd)
        self.destroy()


class PasteDatDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Paste DAT text")
        self.resizable(True, True)
        self.result = None

        ttk.Label(self, text="Paste airfoil DAT text below:").pack(anchor="w", padx=4, pady=2)
        self.txt = tk.Text(self, width=80, height=25)
        self.txt.pack(fill="both", expand=True, padx=4, pady=2)

        btns = ttk.Frame(self)
        btns.pack(pady=4)
        ttk.Button(btns, text="OK", command=self.on_ok).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=4)

        self.grab_set()
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def on_ok(self):
        text = self.txt.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Empty DAT text.")
            return
        self.result = text
        self.destroy()


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    app = ACDesignerApp()
    app.mainloop()
