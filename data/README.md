# Data

The study cohort contains 1,358 patients aged 65 years or older who underwent PCI between 2015 and 2021. The patient-level dataset is not distributed in this repository. The article states that supporting data are available from the corresponding author upon reasonable request.

Place an authorized copy at `data/Data.xlsx` or provide any CSV/XLS/XLSX path to the command-line interface. The default analysis requires these columns:

| Column | Meaning | Expected type |
|---|---|---|
| `Pre PCI EF` | Pre-PCI ejection fraction (%) | numeric |
| `Age` | Age (years) | numeric |
| `BMI` | Body mass index (kg/m²) | numeric |
| `LDLtoHDL` | LDL/HDL ratio | numeric |
| `FBS` | Fasting blood sugar (mg/dL) | numeric |
| `Hemoglobin` | Hemoglobin | numeric |
| `Creatinine` | Serum creatinine (mg/dL) | numeric |
| `MACE` | One-year composite outcome | binary, 0/1 |

`demo.csv`, produced by `scripts/generate_demo_data.py`, is synthetic and must not be used for clinical inference.
