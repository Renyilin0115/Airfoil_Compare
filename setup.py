# setup.py
from cx_Freeze import setup, Executable
import sys

script = "airfoil_planner_gui_v4_1_autolog.py"

# Python 3.14 暫時不要用 "win32gui"
base = None            # 等於 console base；要看訊息也方便除錯
# 若一定要隱藏主控台，請改用 Python 3.12 並把 base 改為 "win32gui"

build_exe_options = {
    "packages": ["tkinter", "json", "csv", "datetime", "re", "shutil", "pathlib", "math"],
    "include_msvcr": True,   # 內含 VC runtime
}

setup(
    name="AirfoilPlanner",
    version="4.1",
    description="Airfoil Planner GUI (v4.1, autolog)",
    options={"build_exe": build_exe_options},
    executables=[Executable(script=script, base=base, target_name="AirfoilPlanner.exe")],
)
