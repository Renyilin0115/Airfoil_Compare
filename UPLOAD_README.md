# Flight Envelope GUI – V3 Update Notes

## 1. Speed Input Changed to Mach Number
- TAS removed as direct input.
- Program now computes airspeed internally using:

V = M * sqrt(gamma * R * T)

## 2. Dynamic Pressure q_max Now Uses SI Units (Pa)
q_max = 0.5 * rho * V^2

## 3. Aspect Ratio (AR) Input Options
Users may input AR directly or let program compute AR from wing span and wing area.

## 4. q_max Calculator
Computes q_max using altitude and V_max (m/s).

## 5. P0 Power Calculator
Computes sea-level engine power from thrust, V_max, and propeller efficiency.

## 6. K Induced Drag Factor Calculator
K = 1 / (pi * e * AR)

## 7. Airfoil DAT Loader
Loads polar data (alpha, CL, ...) and extracts CL_max. Detects geometry files and rejects them.

## 8. Unified Tools Tab
All helper calculators are combined into a single tab for cleaner workflow.
