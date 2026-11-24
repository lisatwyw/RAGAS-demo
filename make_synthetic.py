"""Generate the SYNTHETIC longitudinal dataset (no real patients). Run: python make_synthetic.py

Writes data/patients.json (structured timeline), data/notes.json (clinical notes) and
data/guidelines.json (short illustrative reference snippets). Hand-authored, so the golden eval set can rely on it.
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "data"


def series(unit, dates, values):
    return {"unit": unit, "values": [[d, v] for d, v in zip(dates, values)]}


D1 = ["2023-01-12", "2023-07-20", "2024-01-18", "2024-07-25", "2025-01-30", "2025-07-17", "2026-01-22"]
D2 = ["2024-02-06", "2024-08-13", "2025-02-11", "2025-08-19", "2026-02-17"]
D3 = ["2024-03-05", "2024-06-04", "2024-10-01", "2025-10-07", "2026-04-14"]

PATIENTS = {
    "P001": {
        "name": "Jordan Ellis", "age": 66, "sex": "F", "mrn": "SYN-100001",
        "conditions": [
            {"name": "Type 2 diabetes mellitus", "date": "2019-06-10", "status": "active"},
            {"name": "Hypertension", "date": "2019-06-10", "status": "active"},
            {"name": "Chronic kidney disease stage G3a", "date": "2025-01-30", "status": "active"},
        ],
        "medications": [
            {"name": "metformin", "dose": "1000 mg BID", "start": "2019-06-10", "stop": "2026-01-22", "reason": "type 2 diabetes"},
            {"name": "metformin", "dose": "500 mg BID", "start": "2026-01-22", "stop": None, "reason": "dose reduced because eGFR fell below 45"},
            {"name": "lisinopril", "dose": "10 mg daily", "start": "2019-06-10", "stop": "2024-01-18", "reason": "hypertension"},
            {"name": "lisinopril", "dose": "20 mg daily", "start": "2024-01-18", "stop": None, "reason": "blood pressure above goal"},
            {"name": "empagliflozin", "dose": "10 mg daily", "start": "2025-07-17", "stop": None, "reason": "type 2 diabetes with CKD"},
        ],
        "observations": {
            "HbA1c": series("%", D1, [7.4, 7.6, 7.9, 8.2, 8.3, 8.1, 7.5]),
            "eGFR": series("mL/min/1.73m2", D1, [78, 74, 66, 58, 54, 46, 41]),
            "Creatinine": series("mg/dL", D1, [0.88, 0.93, 1.02, 1.14, 1.20, 1.35, 1.50]),
            "Potassium": series("mmol/L", D1, [4.3, 4.4, 4.6, 4.8, 4.9, 4.7, 4.8]),
            "SBP": series("mmHg", D1, [138, 136, 142, 134, 140, 132, 128]),
            "DBP": series("mmHg", D1, [84, 82, 86, 80, 84, 78, 76]),
            "Weight": series("kg", D1, [82.0, 82.5, 83.0, 83.5, 83.2, 82.8, 82.1]),
        },
    },
    "P002": {
        "name": "Morgan Reyes", "age": 71, "sex": "M", "mrn": "SYN-100002",
        "conditions": [
            {"name": "Heart failure with reduced ejection fraction", "date": "2022-09-01", "status": "active"},
            {"name": "Hypertension", "date": "2018-03-15", "status": "active"},
        ],
        "medications": [
            {"name": "carvedilol", "dose": "12.5 mg BID", "start": "2022-09-01", "stop": None, "reason": "heart failure"},
            {"name": "lisinopril", "dose": "10 mg daily", "start": "2022-09-01", "stop": None, "reason": "heart failure and hypertension"},
            {"name": "furosemide", "dose": "20 mg daily", "start": "2023-06-01", "stop": "2025-02-11", "reason": "fluid control"},
            {"name": "furosemide", "dose": "40 mg daily", "start": "2025-02-11", "stop": None, "reason": "weight gain and leg edema"},
            {"name": "spironolactone", "dose": "25 mg daily", "start": "2024-08-13", "stop": "2025-02-11", "reason": "heart failure"},
            {"name": "spironolactone", "dose": "12.5 mg daily", "start": "2025-02-11", "stop": None, "reason": "dose halved because potassium 5.3"},
        ],
        "observations": {
            "BNP": series("pg/mL", D2, [310, 420, 690, 380, 290]),
            "Potassium": series("mmol/L", D2, [4.4, 4.6, 5.3, 4.8, 4.6]),
            "Creatinine": series("mg/dL", D2, [1.1, 1.2, 1.4, 1.2, 1.1]),
            "Weight": series("kg", D2, [94.0, 95.5, 98.2, 96.0, 93.8]),
            "SBP": series("mmHg", D2, [128, 130, 146, 132, 126]),
            "DBP": series("mmHg", D2, [78, 80, 90, 82, 76]),
            "LVEF": series("%", ["2024-02-06", "2026-02-17"], [35, 40]),
        },
    },
    "P003": {
        "name": "Avery Chen", "age": 58, "sex": "F", "mrn": "SYN-100003",
        "conditions": [
            {"name": "Primary hypothyroidism", "date": "2024-03-05", "status": "active"},
            {"name": "Osteopenia", "date": "2025-12-10", "status": "active"},
        ],
        "medications": [
            {"name": "levothyroxine", "dose": "50 mcg daily", "start": "2024-03-05", "stop": "2024-06-04", "reason": "hypothyroidism"},
            {"name": "levothyroxine", "dose": "75 mcg daily", "start": "2024-06-04", "stop": None, "reason": "TSH still elevated"},
            {"name": "calcium carbonate", "dose": "500 mg daily", "start": "2026-01-20", "stop": None, "reason": "osteopenia"},
        ],
        "observations": {
            "TSH": series("mIU/L", D3, [9.8, 6.1, 3.2, 2.4, 6.8]),
            "Weight": series("kg", D3, [71.0, 71.4, 71.9, 72.5, 72.8]),
        },
    },
}

NOTES = [
    # ---- P001
    ("P001", "2023-07-20", "progress note", "MRN SYN-100001. Routine follow-up for type 2 diabetes and hypertension. HbA1c 7.6%, eGFR 74. Blood pressure 136/82. Stable on metformin 1000 mg BID and lisinopril 10 mg. Continue current therapy."),
    ("P001", "2024-01-18", "progress note", "MRN SYN-100001. Follow-up for diabetes and hypertension. HbA1c 7.9%, BP 142/86, eGFR 66. Patient admits missing evening metformin doses about twice a week because of shift work. Lisinopril increased from 10 mg to 20 mg daily for blood pressure above goal. Adherence strategies discussed."),
    ("P001", "2025-01-30", "progress note", "MRN SYN-100001. eGFR 54 today and 58 on 2024-07-25, both below 60 more than three months apart, so chronic kidney disease stage G3a is documented. Urine albumin testing has not yet been obtained. Potassium 4.9, upper normal on lisinopril 20 mg; recheck planned. NSAID avoidance and hydration reviewed."),
    ("P001", "2025-07-17", "progress note", "MRN SYN-100001. eGFR 46, HbA1c 8.1%. Empagliflozin 10 mg daily started for glycemic control and kidney protection. Patient counselled that a small early dip in eGFR can occur after starting; renal function to be rechecked at the next visit."),
    ("P001", "2026-01-22", "progress note", "MRN SYN-100001. eGFR 41, now in the G3b range. HbA1c improved to 7.5%. Metformin reduced from 1000 mg BID to 500 mg BID because eGFR fell below 45. Empagliflozin tolerated, no genital infections reported. Blood pressure 128/76."),
    # ---- P002
    ("P002", "2024-08-13", "cardiology note", "MRN SYN-100002. Heart failure with reduced ejection fraction, LVEF 35% on echo. Weight 95.5 kg, BNP 420, potassium 4.6. Spironolactone 25 mg daily started. Potassium and creatinine to be rechecked at the next visit."),
    ("P002", "2025-02-11", "cardiology note", "MRN SYN-100002. Weight up to 98.2 kg from 95.5 kg with mild leg edema; BNP 690. Furosemide increased from 20 mg to 40 mg daily. Potassium 5.3, which is elevated; spironolactone reduced from 25 mg to 12.5 mg daily. Dietary potassium, salt and NSAID avoidance reviewed."),
    ("P002", "2025-08-19", "cardiology note", "MRN SYN-100002. Weight 96.0 kg, BNP 380, potassium 4.8. Leg edema resolved. Continue furosemide 40 mg and spironolactone 12.5 mg."),
    ("P002", "2026-02-17", "cardiology note", "MRN SYN-100002. Weight 93.8 kg, BNP 290, potassium 4.6. Repeat echo shows LVEF 40%. Symptoms stable, NYHA class II. No medication changes."),
    # ---- P003
    ("P003", "2024-03-05", "endocrinology note", "MRN SYN-100003. Fatigue and cold intolerance. TSH 9.8 mIU/L. Primary hypothyroidism diagnosed. Levothyroxine 50 mcg daily started; advised to take on an empty stomach."),
    ("P003", "2024-06-04", "endocrinology note", "MRN SYN-100003. TSH 6.1, still above range. Levothyroxine increased from 50 mcg to 75 mcg daily. Recheck TSH in about 6 to 8 weeks."),
    ("P003", "2025-10-07", "endocrinology note", "MRN SYN-100003. TSH 2.4 on levothyroxine 75 mcg. Symptoms resolved. Continue current dose."),
    ("P003", "2025-12-10", "primary care note", "MRN SYN-100003. DXA shows osteopenia. Calcium carbonate 500 mg daily with vitamin D recommended, started 2026-01-20."),
    ("P003", "2026-04-14", "endocrinology note", "MRN SYN-100003. TSH 6.8, higher than 2.4 in October 2025. Patient reports taking the new calcium tablet together with levothyroxine in the morning. Counselled to separate them by about 4 hours. Repeat TSH in 6 to 8 weeks."),
]

NOTE = "Illustrative paraphrase for a demo, not clinical guidance; verify against the original guideline or label."
GUIDELINES = [
    ("ckd-staging", "CKD definition and GFR categories", "CKD is defined by abnormalities of kidney structure or function present for more than 3 months. GFR categories (mL/min/1.73m2): G1 90 or above, G2 60-89, G3a 45-59, G3b 30-44, G4 15-29, G5 below 15. Albuminuria categories A1-A3 add prognostic information."),
    ("metformin-renal", "Metformin and kidney function", "Metformin labeling: contraindicated when eGFR is below 30. Starting metformin is not recommended when eGFR is 30-45. If eGFR falls below 45 in a patient already taking it, the benefit and risk of continuing should be assessed; dose reduction is commonly considered."),
    ("sglt2-egfr-dip", "SGLT2 inhibitors and early eGFR change", "SGLT2 inhibitors commonly cause a small, early, usually reversible fall in eGFR after initiation (a hemodynamic effect). This alone is generally not a reason to stop, but renal function should be rechecked."),
    ("hba1c-targets", "HbA1c thresholds and targets", "HbA1c of 6.5% or above is a diagnostic threshold for diabetes. Many adults with diabetes have an individualized target, often around 7%, adjusted for age, comorbidities and hypoglycemia risk."),
    ("hyperkalemia-monitoring", "Hyperkalemia and RAAS drugs", "Serum potassium above the laboratory upper limit (often about 5.0-5.5 mmol/L) is hyperkalemia. ACE inhibitors, ARBs and mineralocorticoid receptor antagonists raise the risk, especially with reduced kidney function. Potassium and creatinine are usually checked soon after starting or changing these drugs."),
    ("hf-weight-bnp", "Heart failure weight and BNP", "Rapid weight gain (commonly cited as about 2 kg or more over a few days) can indicate fluid retention in heart failure. BNP and NT-proBNP rise with congestion and are also affected by age, kidney function and obesity."),
    ("levothyroxine-monitoring", "Levothyroxine monitoring and interactions", "After starting or changing a levothyroxine dose, TSH is typically rechecked after about 6-8 weeks. Calcium and iron supplements reduce levothyroxine absorption; doses are commonly separated by about 4 hours."),
    ("nsaids-ckd", "NSAIDs and kidney disease", "NSAIDs can reduce kidney function and raise potassium, especially alongside ACE inhibitors or diuretics; avoidance is generally advised in CKD."),
]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "patients.json").write_text(json.dumps(PATIENTS, indent=1), encoding="utf-8")
    (OUT / "notes.json").write_text(json.dumps(
        [{"patient_id": p, "date": d, "type": t, "text": x} for p, d, t, x in NOTES], indent=1), encoding="utf-8")
    (OUT / "guidelines.json").write_text(json.dumps(
        [{"id": i, "title": t, "text": f"{x} ({NOTE})"} for i, t, x in GUIDELINES], indent=1), encoding="utf-8")
    print("wrote data/patients.json, data/notes.json, data/guidelines.json")
