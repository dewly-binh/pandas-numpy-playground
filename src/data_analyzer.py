from typing import Any

import numpy as np
import pandas as pd


class DataAnalyzer:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def clean(self) -> "DataAnalyzer":
        self._df.columns = (
            self._df.columns.str.strip().str.lower().str.replace(" ", "_")
        )

        self._df["start_date"] = pd.to_datetime(self._df["start_date"], errors="coerce")

        self._df["senior_management"] = self._df["senior_management"].apply(
            lambda x: str(x).strip().lower() == "true" if pd.notna(x) else False
        )

        self._df["last_login_time"] = pd.to_datetime(
            self._df["last_login_time"], format="%I:%M %p"
        ).dt.time

        if "first_name" in self._df.columns:
            self._df["first_name"] = self._df["first_name"].fillna("Unknown")
        if "gender" in self._df.columns:
            self._df["gender"] = self._df["gender"].fillna("Unknown")
        if "department" in self._df.columns:
            self._df["department"] = self._df["department"].fillna("Unknown")

        self._df.rename(columns={"bonus_%": "bonus_pct"}, inplace=True)

        return self

    def filter_by(
        self, column: str, value: Any, operator: str = "=="
    ) -> pd.DataFrame | None:
        ops = {
            "==": lambda col, val: col == val,
            "!=": lambda col, val: col != val,
            ">": lambda col, val: col > val,
            "<": lambda col, val: col < val,
            ">=": lambda col, val: col >= val,
            "<=": lambda col, val: col <= val,
        }
        if operator not in ops:
            raise ValueError(f"Operator invalid: {operator}")

        lm_fnc = ops[operator]  # operator = "==" => lamda col, val: col == val
        mask = lm_fnc(self._df[column], value)  # <=> self._df[columns] == value
        result = self._df[mask]

        return result if not result.empty else None

    def group_by_department(
        self, agg_cols: dict[str, list[str] | str]
    ) -> pd.DataFrame | None:
        # agg_cols: dict[str, list[str] | str] : key tên cột, value các phương thức muốn thực hiện ở cột đó
        result = self._df.groupby("department").agg(agg_cols)

        return result if not result.empty else None

    def statistical_calculate(self) -> dict:
        salary = self._df["salary"].to_numpy()
        bonus = self._df["bonus_pct"].to_numpy()
        salary_mean = np.mean(salary)
        salary_std = np.std(salary)
        salary_max = np.max(salary)
        salary_min = np.min(salary)

        senior_salary = (
            self._df[self._df["senior_management"]]["salary"].to_numpy().mean()
        )
        non_senior_salary = (
            self._df[~self._df["senior_management"]]["salary"].to_numpy().mean()
        )
        gap = senior_salary - non_senior_salary
        male_salary = self._df[self._df["gender"] == "Male"]["salary"].to_numpy().mean()
        female_salary = (
            self._df[self._df["gender"] == "Female"]["salary"].to_numpy().mean()
        )
        gap_gender = male_salary - female_salary

        actual_bonus = salary * bonus / 100

        return {
            "salary_mean": salary_mean,
            "salary_std": salary_std,
            "salary_max": salary_max,
            "salary_min": salary_min,
            "senior_salary": senior_salary,
            "non_senior_salary": non_senior_salary,
            "gap": gap,
            "male_salary": male_salary,
            "female_salary": female_salary,
            "gap_gender": gap_gender,
            "actual_bonus": actual_bonus,
        }
