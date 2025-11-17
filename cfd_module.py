# cfd_module.py
# 簡易「CFD 介面」：之後要將計算換成 XFOIL / OpenFOAM / 自訂程式

import math

def run_2d_airfoil_case(airfoil_name: str,
                        Re: float,
                        M: float,
                        alpha_deg: float,
                        tc: float,
                        sweep_deg: float):
    """
    輸入：
      airfoil_name : 翼型名稱
      Re           : 雷諾數
      M            : 馬赫數
      alpha_deg    : 迎角 [deg]
      tc           : 厚度比
      sweep_deg    : 後掠角 [deg]（這裡暫時不用，可留著之後擴充）

    回傳一個 dict：
      {
        'CL': ...,
        'CDp': ...,
        'CDi': ...,
        'CD': ...,
        'Cm': ...,
        'notes': '說明文字'
      }
    """

    # 這裡現在是「假 CFD」，之後改成真正的 solver
    alpha = math.radians(alpha_deg)

    # 用 thin airfoil 理論做個簡單 model 當 placeholder：
    Cla_2d = 2.0 * math.pi         # dCL/dα [per rad]
    CL = Cla_2d * alpha            # 單一 α 對應的 CL

    # 粗略 profile drag：Re 越大 CDp 越小
    Re_ref = max(Re, 1e5)
    CDp = 0.010 + 0.006 * (1e6 / Re_ref) ** 0.2
    CDp = max(0.007, min(0.040, CDp))

    # 簡單 induced drag，用 AR=8 當示意
    AR = 8.0
    e = 0.8
    CDi = CL ** 2 / (math.pi * AR * e)

    CD = CDp + CDi

    # 粗略 Cm0：假設厚一點 nose-down moment 大一點
    Cm0 = -0.02 - 0.3 * (tc - 0.12)

    notes = (
        f"Fake CFD for {airfoil_name}\n"
        f"alpha = {alpha_deg:.1f} deg, M = {M:.3f}, Re = {Re:.2e}\n"
        f"CL = {CL:.3f}, CDp = {CDp:.4f}, CDi = {CDi:.4f}, CD = {CD:.4f}, Cm ≈ {Cm0:.3f}"
    )

    return {
        "CL": CL,
        "CDp": CDp,
        "CDi": CDi,
        "CD": CD,
        "Cm": Cm0,
        "notes": notes,
    }
