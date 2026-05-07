from pathlib import Path

import pandas as pd


def generate_summary(df: pd.DataFrame):
    by_department = df.groupby("department")["salary"].sum()
    summary = {
        "total_employees": len(df),
        "average_salary": df["salary"].mean(),
        "max_salary": df["salary"].max(),
        "min_salary": df["salary"].min(),
    }

    for department, total in by_department.items():
        summary[f"department_{str(department).lower().replace(" ", "_")}"] = total

    print(by_department)

    return summary


def save_report(summary: dict, out_path: str):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        pd.DataFrame([summary]).to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    print(f"Report saved to: {path.resolve()}")
