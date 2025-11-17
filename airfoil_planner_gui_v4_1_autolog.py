
import json, math, csv, datetime, re, shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "Airfoil Planner GUI v4.1 (Score Rank + DAT Paste + AutoLog)"
LOG_DIR_NAME = "Airfoil_log"

# ----------------- Unit helpers (Imperial ➜ SI) -----------------
def lb_to_N(lb): return lb * 4.4482216153
def ft_to_m(ft): return ft * 0.3048
def kt_to_mps(kt): return kt * 0.514444
def ft2_to_m2(ft2): return ft2 * 0.09290304

# ----------------- Atmosphere & physics -----------------
def isa_props(h_m):
    T0, p0, R, g, L = 288.15, 101325.0, 287.058, 9.80665, -0.0065
    if h_m <= 11000:
        T = T0 + L * h_m
        p = p0 * (T / T0) ** (-g / (L * R))
    else:
        T11 = T0 + L * 11000.0
        p11 = p0 * (T11 / T0) ** (-g / (L * R))
        p = p11 * math.exp(-g * (h_m - 11000.0) / (R * T11))
        T = T11
    rho = p / (R * T)
    C1, S = 1.458e-6, 110.4
    mu = C1 * T ** 1.5 / (T + S)
    a = math.sqrt(1.4 * R * T)
    return T, a, rho, mu

def phase_calc_SI(W_N, S_m2, c_m, h_m, V_mps):
    T, a, rho, mu = isa_props(h_m)
    q = 0.5 * rho * V_mps * V_mps
    CL = W_N / (q * S_m2) if q > 0 else float("nan")
    Re = rho * V_mps * c_m / mu if mu > 0 else float("nan")
    M = V_mps / a if a > 0 else float("nan")
    return dict(T=T, a=a, rho=rho, mu=mu, q=q, CL=CL, Re=Re, M=M)

# ----------------- Simple polar models -----------------
class ParabolicPolar:
    def __init__(self, name, cd0, k, clmax_clean, cm0=-0.08):
        self.name = name
        self.cd0 = cd0
        self.k = k
        self.clmax_clean = clmax_clean
        self.cm0 = cm0
    def cd(self, cl):
        return self.cd0 + self.k * (cl ** 2)

BUILTIN_MODELS = [
    ParabolicPolar("NACA 2412 (12%)", cd0=0.0085, k=0.045, clmax_clean=1.5, cm0=-0.05),
    ParabolicPolar("NACA 23012 (12%)", cd0=0.0090, k=0.040, clmax_clean=1.6, cm0=-0.08),
    ParabolicPolar("Selig S1223 (14%)", cd0=0.0100, k=0.055, clmax_clean=1.9, cm0=-0.10),
    ParabolicPolar("SC(2)-0612 Supercritical (12%)", cd0=0.0095, k=0.038, clmax_clean=1.4, cm0=-0.06),
    ParabolicPolar("Biconvex 6% (supersonic)", cd0=0.0120, k=0.030, clmax_clean=1.2, cm0=-0.02),
]

# ----------------- CSV polar loader -----------------
def load_polar_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                cl = float(row.get("CL") or row.get("Cl") or row.get("cl"))
                cd = float(row.get("CD") or row.get("Cd") or row.get("cd"))
                cm = row.get("CM") or row.get("Cm") or row.get("cm")
                cm = float(cm) if cm is not None and str(cm) != "" else None
                Re = row.get("Re") or row.get("RE") or row.get("re")
                Re = float(Re) if Re not in (None, "") else None
                Ma = row.get("Mach") or row.get("M") or row.get("Ma")
                Ma = float(Ma) if Ma not in (None, "") else None
                rows.append({"CL": cl, "CD": cd, "CM": cm, "Re": Re, "Mach": Ma})
            except Exception:
                continue
    rows.sort(key=lambda x: x["CL"])
    return rows

def interp_cd_from_csv(rows, cl_target):
    if not rows: return None
    lo, hi = None, None
    for r in rows:
        if r["CL"] <= cl_target: lo = r
        if r["CL"] >= cl_target:
            hi = r; break
    if lo is None: lo = rows[0]
    if hi is None: hi = rows[-1]
    if hi["CL"] == lo["CL"]: return lo["CD"]
    t = (cl_target - lo["CL"]) / (hi["CL"] - lo["CL"])
    return lo["CD"] + t * (hi["CD"] - lo["CD"])

# ----------------- DAT parser (file & pasted text) -----------------
def parse_airfoil_dat_lines(lines):
    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines: raise ValueError("Empty DAT")
    name = lines[0]
    pts = []
    for ln in lines[1:]:
        parts = ln.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                x = float(parts[0]); y = float(parts[1])
                pts.append((x, y))
            except ValueError:
                continue
    if len(pts) < 10: raise ValueError("Too few coordinate points")
    xs = [p[0] for p in pts]; xmax = max(xs); xmin = min(xs); span = xmax - xmin
    if span == 0: raise ValueError("Invalid coordinates")
    pts = [((p[0] - xmin) / span, p[1] / span) for p in pts]
    pts_sorted = sorted(pts, key=lambda p: p[0])
    bins = {}
    for x, y in pts_sorted:
        k = round(x, 4)
        if k not in bins: bins[k] = {"x": x, "ymax": y, "ymin": y}
        else:
            bins[k]["ymax"] = max(bins[k]["ymax"], y)
            bins[k]["ymin"] = min(bins[k]["ymin"], y)
    xs = sorted(bins.keys())
    thickness, camber = [], []
    for k in xs:
        u = bins[k]["ymax"]; l = bins[k]["ymin"]
        thickness.append((bins[k]["x"], u - l))
        camber.append((bins[k]["x"], 0.5*(u + l)))
    tc = max(t for _, t in thickness)
    mcam = max(abs(c) for _, c in camber)
    # Rough LE radius
    le_pts = [(x, y) for x, y in pts_sorted if x <= 0.05][:5]
    r_le = None
    if len(le_pts) >= 3:
        (x1, y1) = le_pts[0]; (x2, y2) = le_pts[len(le_pts)//2]; (x3, y3) = le_pts[-1]
        def circle_from(p1, p2, p3):
            (x1, y1) = p1; (x2, y2) = p2; (x3, y3) = p3
            a = x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2
            if abs(a) < 1e-9: return None
            b = (x1**2+y1**2)*(y3-y2) + (x2**2+y2**2)*(y1-y3) + (x3**2+y3**2)*(y2-y1)
            c = (x1**2+y1**2)*(x2-x3) + (x2**2+y2**2)*(x3-x1) + (x3**2+y3**2)*(x1-x2)
            xc = -b/(2*a); yc = -c/(2*a)
            r = math.hypot(x1-xc, y1-yc); return r
        r_try = circle_from((x1, y1), (x2, y2), (x3, y3))
        if r_try and r_try > 0: r_le = r_try
    return {"name": name.strip(), "xy": pts_sorted, "tc": tc, "mcam": mcam, "rle": r_le, "raw_lines": lines}

def parse_airfoil_dat(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return parse_airfoil_dat_lines(f.readlines())

def build_surrogate_from_geometry(name, tc, mcam, AR_hint=8.0, e=0.8):
    cd0 = 0.005 + 0.015 * tc
    k = 1.0 / (math.pi * e * AR_hint) if AR_hint and AR_hint > 0 else 0.04
    clmax = min(2.2, 1.2 + 1.8 * tc + 1.0 * mcam)
    return ParabolicPolar(f"{name} (geom)", cd0=cd0, k=k, clmax_clean=clmax, cm0=-0.08)

# ----------------- Scoring (0–100) -----------------
def score_from_data(phases, airfoil, csv_rows=None, Lref=30.0, SMref=0.30, w_ld=0.7, w_sm=0.3):
    total = 0.0
    hard_fail = False
    for ph in phases:
        cl = ph["CL"]; M = ph["M"]
        wave_pen = 0.0
        if M >= 0.72: wave_pen = 0.004 * (M - 0.72) * 10.0
        if csv_rows:
            cd = interp_cd_from_csv(csv_rows, cl)
            if cd is None and airfoil: cd = airfoil.cd(cl)
        else:
            cd = airfoil.cd(cl)
        cd_eff = cd + wave_pen
        l_over_d = cl / max(cd_eff, 1e-9)
        stall_margin = None
        if hasattr(airfoil, "clmax_clean"):
            stall_margin = airfoil.clmax_clean - cl
            if stall_margin < 0: hard_fail = True
        s_ld = min(l_over_d / Lref, 1.5)
        s_sm = max(0.0, (stall_margin or 0.0) / SMref)
        total += w_ld * s_ld + w_sm * s_sm
    if hard_fail:
        return 0.0
    return 100.0 * total / max(len(phases), 1)

# ----------------- Logging helpers -----------------
def ensure_log_dir():
    # Create Airfoil_log in current working directory
    logdir = Path.cwd() / LOG_DIR_NAME
    logdir.mkdir(parents=True, exist_ok=True)
    return logdir

def slugify(name):
    name = re.sub(r'[^A-Za-z0-9_\\-]+', '_', name.strip())
    return name.strip('_')[:80] or "airfoil"

def log_dat(name, lines, meta):
    logdir = ensure_log_dir()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = logdir / f"{ts}_{slugify(name)}"
    folder.mkdir(parents=True, exist_ok=True)
    # save dat
    (folder / "airfoil.dat").write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    # save meta
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(folder)

def log_csv(name, src_path, rows, meta):
    logdir = ensure_log_dir()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = logdir / f"{ts}_{slugify(name)}"
    folder.mkdir(parents=True, exist_ok=True)
    # copy csv
    try:
        shutil.copy2(src_path, folder / Path(src_path).name)
    except Exception:
        pass
    # save a compact csv summary (first 10 points)
    import io
    summary = io.StringIO()
    w = csv.writer(summary)
    w.writerow(["CL","CD","CM","Re","Mach"])
    for r in rows[:10]:
        w.writerow([r["CL"], r["CD"], r.get("CM",""), r.get("Re",""), r.get("Mach","")])
    (folder / "preview.csv").write_text(summary.getvalue(), encoding="utf-8")
    # save meta
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(folder)

# ----------------- GUI -----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1220x840"); self.minsize(1100, 720)
        self.airfoils = []; self.results = []
        self.create_widgets()

    def create_widgets(self):
        # Aircraft inputs
        frm = ttk.LabelFrame(self, text="Aircraft Inputs (Imperial)")
        frm.pack(fill="x", padx=10, pady=6)
        self.var_W_lb = tk.StringVar(value="220")
        self.var_S_ft2 = tk.StringVar(value="26.9")
        self.var_c_ft = tk.StringVar(value="1.97")
        self.var_AR = tk.StringVar(value="8")
        def add(label, var, col):
            ttk.Label(frm, text=label).grid(row=0, column=col*2, sticky="e", padx=4, pady=4)
            ttk.Entry(frm, textvariable=var, width=12).grid(row=0, column=col*2+1, sticky="w", padx=4, pady=4)
        add("Weight [lb]", self.var_W_lb, 0)
        add("Wing area [ft²]", self.var_S_ft2, 1)
        add("Ref chord [ft]", self.var_c_ft, 2)
        add("AR (hint)", self.var_AR, 3)

        # Mission phases
        phases_frame = ttk.LabelFrame(self, text="Phases (Imperial)")
        phases_frame.pack(fill="both", expand=False, padx=10, pady=6)
        cols = ("name", "altitude_ft", "speed_kt")
        self.tree = ttk.Treeview(phases_frame, columns=cols, show="headings", height=5)
        for c in cols: self.tree.heading(c, text=c); self.tree.column(c, width=130, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(phases_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=sb.set); sb.pack(side="right", fill="y")
        for row in [("takeoff", "0", "50"), ("cruise", "6500", "107"), ("approach", "0", "43")]:
            self.tree.insert("", "end", values=row)

        btns = ttk.Frame(self); btns.pack(fill="x", padx=10, pady=4)
        ttk.Button(btns, text="Add Phase", command=self.add_phase).pack(side="left", padx=2)
        ttk.Button(btns, text="Edit Phase", command=self.edit_phase).pack(side="left", padx=2)
        ttk.Button(btns, text="Delete Phase", command=self.del_phase).pack(side="left", padx=2)
        ttk.Button(btns, text="Compute Envelope", command=self.compute_envelope).pack(side="left", padx=10)
        ttk.Button(btns, text="Export Mission CSV", command=self.export_csv).pack(side="left", padx=2)
        ttk.Button(btns, text="Export Report", command=self.export_md).pack(side="left", padx=2)
        ttk.Button(btns, text="Save Mission JSON", command=self.save_mission).pack(side="right", padx=2)
        ttk.Button(btns, text="Load Mission JSON", command=self.load_mission).pack(side="right", padx=2)

        # Results table
        res_frame = ttk.LabelFrame(self, text="Phase Results (SI)")
        res_frame.pack(fill="both", expand=False, padx=10, pady=6)
        self.res_cols = ("name", "h[m]", "V[m/s]", "M", "Re", "q[Pa]", "CL_req")
        self.res_tree = ttk.Treeview(res_frame, columns=self.res_cols, show="headings", height=6)
        for c in self.res_cols: self.res_tree.heading(c, text=c); self.res_tree.column(c, width=120, anchor="center")
        self.res_tree.pack(side="left", fill="both", expand=True)
        rsb = ttk.Scrollbar(res_frame, orient="vertical", command=self.res_tree.yview)
        self.res_tree.configure(yscroll=rsb.set); rsb.pack(side="right", fill="y")

        # Airfoil Evaluator
        af_frame = ttk.LabelFrame(self, text="Airfoil Evaluator")
        af_frame.pack(fill="both", expand=True, padx=10, pady=6)
        self.af_cols = ("name", "type")
        self.af_tree = ttk.Treeview(af_frame, columns=self.af_cols, show="headings", height=6)
        for c in self.af_cols:
            self.af_tree.heading(c, text=c)
            self.af_tree.column(c, width=240 if c == "name" else 100, anchor="center")
        self.af_tree.grid(row=0, column=0, rowspan=9, sticky="nsew", padx=4, pady=4)
        af_frame.grid_columnconfigure(0, weight=1); af_frame.grid_rowconfigure(0, weight=1)
        ttk.Button(af_frame, text="Add Built-in Models", command=self.add_builtin).grid(row=0, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(af_frame, text="Import Polar CSV…", command=self.import_csv).grid(row=1, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(af_frame, text="Import Airfoil DAT…", command=self.import_dat).grid(row=2, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(af_frame, text="Paste DAT text…", command=self.paste_dat_text).grid(row=3, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(af_frame, text="Remove Selected", command=self.remove_selected_af).grid(row=4, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(af_frame, text="Score Rank (from Data)", command=self.score_rank).grid(row=5, column=1, sticky="ew", padx=6, pady=2)

        # Ranking table
        rank_cols = ("rank", "name", "score(0-100)", "L/D@cruise", "stall_min", "t/c", "|Cm0|")
        self.rank_tree = ttk.Treeview(af_frame, columns=rank_cols, show="headings", height=10)
        for c in rank_cols:
            self.rank_tree.heading(c, text=c)
            self.rank_tree.column(c, width=130 if c != "rank" else 60, anchor="center")
        self.rank_tree.grid(row=8, column=0, columnspan=2, sticky="nsew", padx=4, pady=6)
        af_frame.grid_rowconfigure(8, weight=1)

        # Notes
        rec_frame = ttk.LabelFrame(self, text="Notes")
        rec_frame.pack(fill="both", expand=True, padx=10, pady=6)
        self.txt_rec = tk.Text(rec_frame, height=8, wrap="word")
        self.txt_rec.pack(fill="both", expand=True)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(fill="x", padx=10, pady=4)

    # ---- Phase ops ----
    def add_phase(self): PhaseEditor(self, title="Add Phase")
    def edit_phase(self):
        sel = self.tree.selection()
        if not sel: return
        iid = sel[0]; vals = self.tree.item(iid, "values")
        PhaseEditor(self, title="Edit Phase", iid=iid, values=vals)
    def del_phase(self):
        for iid in self.tree.selection(): self.tree.delete(iid)

    def gather_phases_SI(self):
        try:
            W_lb = float(self.var_W_lb.get()); W_N = lb_to_N(W_lb)
            S_ft2 = float(self.var_S_ft2.get()); S_m2 = ft2_to_m2(S_ft2)
            c_ft = float(self.var_c_ft.get()); c_m = ft_to_m(c_ft)
        except ValueError:
            messagebox.showerror("Error", "Aircraft inputs must be numbers."); return None
        phases = []
        for iid in self.tree.get_children():
            name, h_ft, v_kt = self.tree.item(iid, "values")
            try:
                h_m = ft_to_m(float(h_ft)); v_mps = kt_to_mps(float(v_kt))
            except ValueError:
                messagebox.showerror("Error", "Phase values must be numbers."); return None
            out = phase_calc_SI(W_N, S_m2, c_m, h_m, v_mps)
            out.update({"name": name, "alt_ft": float(h_ft), "spd_kt": float(v_kt)})
            phases.append(out)
        return dict(W_N=W_N, S_m2=S_m2, c_m=c_m, phases=phases)

    def compute_envelope(self):
        dat = self.gather_phases_SI()
        if not dat: return
        for iid in self.res_tree.get_children(): self.res_tree.delete(iid)
        for r in dat["phases"]:
            self.res_tree.insert("", "end",
                values=(r["name"], f"{ft_to_m(r['alt_ft']):.0f}", f"{kt_to_mps(r['spd_kt']):.1f}",
                        f"{r['M']:.3f}", f"{r['Re']:.2e}", f"{r['q']:.1f}", f"{r['CL']:.3f}"))
        Ms = [r["M"] for r in dat["phases"]]; Res = [r["Re"] for r in dat["phases"]]
        self.txt_rec.delete("1.0", "end")
        self.txt_rec.insert("1.0", f"Envelope: M∈[{min(Ms):.2f},{max(Ms):.2f}], Re∈[{min(Res):.2e},{max(Res):.2e}]\n")
        self.results = dat; self.status.set("Envelope computed.")

    # ---- Airfoil ops ----
    def add_builtin(self):
        count = 0
        for m in BUILTIN_MODELS:
            self.airfoils.append({"type": "model", "name": m.name, "obj": m})
            self.af_tree.insert("", "end", values=(m.name, "model")); count += 1
        self.status.set(f"Added {count} built-in models.")

    def import_csv(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")], title="Import Polar CSV")
        if not fp: return
        rows = load_polar_csv(fp)
        if not rows: messagebox.showerror("Error", "No valid CL/CD rows found."); return
        name = Path(fp).stem
        self.airfoils.append({"type": "csv", "name": name, "rows": rows})
        self.af_tree.insert("", "end", values=(name, "csv"))
        # Auto-log
        meta = {"source":"csv","name":name,"points":len(rows)}
        folder = log_csv(name, fp, rows, meta)
        self.txt_rec.insert("end", f"Logged CSV to: {folder}\n")
        self.status.set(f"Imported polar: {name} ({len(rows)} pts)")

    def import_dat(self):
        fp = filedialog.askopenfilename(filetypes=[("DAT", "*.dat")], title="Import Airfoil DAT")
        if not fp: return
        try:
            af = parse_airfoil_dat(fp)
        except Exception as e:
            messagebox.showerror("Error", f"DAT parse failed: {e}"); return
        try: AR_hint = float(self.var_AR.get())
        except Exception: AR_hint = 8.0
        model = build_surrogate_from_geometry(af["name"], af["tc"], af["mcam"], AR_hint=AR_hint, e=0.8)
        self.airfoils.append({"type": "geom", "name": model.name, "obj": model, "geom": af})
        self.af_tree.insert("", "end", values=(model.name, "geom"))
        # Auto-log (save original lines from file)
        try:
            raw = Path(fp).read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            raw = af.get("raw_lines", [])
        meta = {"source":"dat_file","name":af["name"],"tc":af["tc"],"mcam":af["mcam"],"rle":af["rle"],
                "surrogate":{"cd0":model.cd0,"k":model.k,"clmax":model.clmax_clean,"cm0":model.cm0}}
        folder = log_dat(af["name"], raw, meta)
        self.txt_rec.insert("end", f"Logged DAT to: {folder}\n")
        self.status.set(f"Imported DAT: {af['name']}")

    def paste_dat_text(self):
        dlg = DATPasteDialog(self, title="Paste DAT text")
        self.wait_window(dlg)
        if not getattr(dlg, "ok", False): return
        lines = dlg.text.splitlines()
        try:
            af = parse_airfoil_dat_lines(lines)
        except Exception as e:
            messagebox.showerror("Error", f"DAT parse failed: {e}")
            return
        try: AR_hint = float(self.var_AR.get())
        except Exception: AR_hint = 8.0
        model = build_surrogate_from_geometry(af["name"], af["tc"], af["mcam"], AR_hint=AR_hint, e=0.8)
        self.airfoils.append({"type": "geom", "name": model.name, "obj": model, "geom": af})
        self.af_tree.insert("", "end", values=(model.name, "geom"))
        # Auto-log pasted dat
        meta = {"source":"dat_paste","name":af["name"],"tc":af["tc"],"mcam":af["mcam"],"rle":af["rle"],
                "surrogate":{"cd0":model.cd0,"k":model.k,"clmax":model.clmax_clean,"cm0":model.cm0}}
        folder = log_dat(af["name"], af["raw_lines"], meta)
        self.txt_rec.insert("end", f"Logged DAT to: {folder}\n")
        self.status.set(f"Pasted DAT: {af['name']}")

    def remove_selected_af(self):
        for iid in self.af_tree.selection():
            name = self.af_tree.item(iid, "values")[0]; self.af_tree.delete(iid)
            self.airfoils = [a for a in self.airfoils if a["name"] != name]
        self.status.set("Removed selected.")

    # ---- Score rank ----
    def score_rank(self):
        if not self.results:
            messagebox.showwarning("Warning", "Compute envelope first."); return
        if not self.airfoils:
            messagebox.showwarning("Warning", "Add airfoils (built-in, CSV, or DAT)."); return
        phases = [{"name": ph["name"], "CL": ph["CL"], "M": ph["M"], "Re": ph["Re"]} for ph in self.results["phases"]]
        rows = []
        for af in self.airfoils:
            if af["type"] == "csv":
                backbone = ParabolicPolar(af["name"] + " (csv)", cd0=0.009, k=0.045, clmax_clean=1.5, cm0=-0.08)
                score = score_from_data(phases, backbone, csv_rows=af["rows"])
                cm0 = backbone.cm0; tc = None
                ld_cruise, stall_min = extract_ld_stall(phases, backbone, csv_rows=af["rows"])
            else:
                model = af["obj"]
                score = score_from_data(phases, model, csv_rows=None)
                cm0 = getattr(model, "cm0", None)
                tc = af.get("geom", {}).get("tc", None)
                ld_cruise, stall_min = extract_ld_stall(phases, model, csv_rows=None)
            rows.append({"name": af["name"], "score": score, "ld": ld_cruise or 0.0, "stall": stall_min or 0.0,
                         "tc": tc, "cm0": cm0})
        rows.sort(key=lambda r: (-r["score"], -r["ld"], -r["stall"]))
        for iid in self.rank_tree.get_children(): self.rank_tree.delete(iid)
        for i, r in enumerate(rows, start=1):
            self.rank_tree.insert("", "end",
                values=(i, r["name"], f"{r['score']:.1f}", f"{r['ld']:.1f}", f"{r['stall']:.2f}",
                        f"{r['tc']:.3f}" if r["tc"] is not None else "", f"{abs(r['cm0']):.3f}" if r["cm0"] is not None else ""))
        self.txt_rec.insert("end", "Score rank uses L/D and stall margin across phases (0–100). CSV polars provide higher fidelity.\n")
        self.status.set("Score ranking complete.")

    # ---- Export & mission load/save ----
    def export_csv(self):
        if not self.results:
            messagebox.showwarning("Warning", "Compute first."); return
        fp = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], title="Save CSV")
        if not fp: return
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["phase", "h_m", "V_mps", "M", "Re", "q_Pa", "CL_req"])
            for ph in self.results["phases"]:
                w.writerow([ph["name"], f"{ft_to_m(ph['alt_ft']):.0f}", f"{kt_to_mps(ph['spd_kt']):.3f}",
                            f"{ph['M']:.6f}", f"{ph['Re']:.6e}", f"{ph['q']:.3f}", f"{ph['CL']:.6f}"])
        self.status.set(f"Saved CSV: {fp}")

    def export_md(self):
        if not self.results:
            messagebox.showwarning("Warning", "Compute first."); return
        fp = filedialog.asksaveasfilename(defaultextension=".md", filetypes=[("Markdown", "*.md")], title="Save Report")
        if not fp: return
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        W_lb = float(self.var_W_lb.get()); S_ft2 = float(self.var_S_ft2.get()); c_ft = float(self.var_c_ft.get())
        lines = ["# Airfoil Planner Report (Imperial)", "", f"Generated: {ts}", "",
                 f"**Inputs**  W={W_lb:.1f} lb, S={S_ft2:.2f} ft², c_ref={c_ft:.2f} ft.", "", "## Phases"]
        for ph in self.results["phases"]:
            lines.append(f"- {ph['name']}: h={ft_to_m(ph['alt_ft']):.0f} m, V={kt_to_mps(ph['spd_kt']):.1f} m/s, "
                         f"M={ph['M']:.3f}, Re={ph['Re']:.2e}, CL_req={ph['CL']:.3f}")
        with open(fp, "w", encoding="utf-8") as f: f.write("\n".join(lines))
        self.status.set(f"Saved Report: {fp}")

    def load_mission(self):
        fp = filedialog.askopenfilename(filetypes=[("JSON", "*.json")], title="Load Mission JSON")
        if not fp: return
        try:
            with open(fp, "r", encoding="utf-8") as f: cfg = json.load(f)
            self.var_W_lb.set(str(cfg["aircraft"]["weight_lb"]))
            self.var_S_ft2.set(str(cfg["aircraft"]["wing_area_ft2"]))
            self.var_c_ft.set(str(cfg["aircraft"]["ref_chord_ft"]))
            for iid in self.tree.get_children(): self.tree.delete(iid)
            for ph in cfg["phases"]:
                self.tree.insert("", "end", values=(ph["name"], str(ph["altitude_ft"]), str(ph["speed_kt"])))
            self.status.set("Mission loaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Load failed: {e}")

    def save_mission(self):
        try:
            W_lb = float(self.var_W_lb.get()); S_ft2 = float(self.var_S_ft2.get()); c_ft = float(self.var_c_ft.get())
        except ValueError:
            messagebox.showerror("Error", "Aircraft inputs must be numbers."); return
        phases = []
        for iid in self.tree.get_children():
            name, h, V = self.tree.item(iid, "values")
            phases.append({"name": name, "altitude_ft": float(h), "speed_kt": float(V)})
        cfg = {"aircraft": {"weight_lb": W_lb, "wing_area_ft2": S_ft2, "ref_chord_ft": c_ft}, "phases": phases}
        fp = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], title="Save Mission JSON")
        if not fp: return
        with open(fp, "w", encoding="utf-8") as f: json.dump(cfg, f, ensure_ascii=False, indent=2)
        self.status.set(f"Mission saved: {fp}")

# Helpers to show LD/stall
def extract_ld_stall(phases, airfoil, csv_rows=None):
    cruise_name = None
    for p in phases:
        if p["name"].lower() == "cruise":
            cruise_name = "cruise"
            break
    if cruise_name is None and phases:
        cruise_name = phases[0]["name"]
    ld_cruise = None
    stall_min = None
    for ph in phases:
        cl = ph["CL"]; M = ph["M"]
        wave_pen = 0.0
        if M >= 0.72: wave_pen = 0.004 * (M - 0.72) * 10.0
        if csv_rows:
            cd = interp_cd_from_csv(csv_rows, cl)
            if cd is None and airfoil: cd = airfoil.cd(cl)
        else:
            cd = airfoil.cd(cl)
        cd_eff = cd + wave_pen
        l_over_d = cl / max(cd_eff, 1e-9)
        if ph["name"].lower() == cruise_name: ld_cruise = l_over_d
        stall_margin = None
        if hasattr(airfoil, "clmax_clean"):
            stall_margin = airfoil.clmax_clean - cl
        if stall_min is None or (stall_margin is not None and stall_margin < stall_min):
            stall_min = stall_margin
    return ld_cruise, stall_min

# ---- Phase editor ----
class PhaseEditor(tk.Toplevel):
    def __init__(self, master, title, iid=None, values=None):
        super().__init__(master)
        self.title(title); self.resizable(False, False); self.iid = iid
        ttk.Label(self, text="name").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        ttk.Label(self, text="altitude_ft").grid(row=1, column=0, padx=6, pady=6, sticky="e")
        ttk.Label(self, text="speed_kt").grid(row=2, column=0, padx=6, pady=6, sticky="e")
        self.var_name = tk.StringVar(value=values[0] if values else "")
        self.var_h = tk.StringVar(value=values[1] if values else "0")
        self.var_v = tk.StringVar(value=values[2] if values else "50")
        ttk.Entry(self, textvariable=self.var_name, width=24).grid(row=0, column=1, padx=6, pady=6)
        ttk.Entry(self, textvariable=self.var_h, width=24).grid(row=1, column=1, padx=6, pady=6)
        ttk.Entry(self, textvariable=self.var_v, width=24).grid(row=2, column=1, padx=6, pady=6)
        btns = ttk.Frame(self); btns.grid(row=3, column=0, columnspan=2, pady=6)
        ttk.Button(btns, text="OK", command=self.on_ok).pack(side="left", padx=8)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=8)
        self.grab_set(); self.protocol("WM_DELETE_WINDOW", self.destroy)
    def on_ok(self):
        name = self.var_name.get().strip()
        try:
            h = float(self.var_h.get()); v = float(self.var_v.get())
        except ValueError:
            messagebox.showerror("Error", "Values must be numeric."); return
        if not name:
            messagebox.showerror("Error", "Name required."); return
        if self.iid is None: self.master.tree.insert("", "end", values=(name, str(h), str(v)))
        else: self.master.tree.item(self.iid, values=(name, str(h), str(v)))
        self.destroy()

# ---- Paste DAT dialog ----
class DATPasteDialog(tk.Toplevel):
    def __init__(self, master, title="Paste DAT text"):
        super().__init__(master)
        self.title(title); self.resizable(True, True)
        ttk.Label(self, text="貼上 DAT 內容：").pack(anchor="w", padx=8, pady=6)
        self.txt = tk.Text(self, width=80, height=24, wrap="none")
        self.txt.pack(fill="both", expand=True, padx=8, pady=6)
        btns = ttk.Frame(self); btns.pack(pady=6)
        ttk.Button(btns, text="OK", command=self.on_ok).pack(side="left", padx=8)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=8)
        self.grab_set()
    def on_ok(self):
        self.text = self.txt.get("1.0", "end").strip()
        if not self.text:
            messagebox.showerror("Error", "No text provided."); return
        self.ok = True
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
