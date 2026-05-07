import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.report_generator import generate_summary, save_report


class TestGenerateSummary:
    def test_generate_summary_basic_stats(self):
        df = pd.DataFrame(
            {
                "department": ["Engineering", "Sales", "Engineering"],
                "salary": [100000, 80000, 120000],
            }
        )
        summary = generate_summary(df)

        assert summary["total_employees"] == 3
        assert summary["average_salary"] == 100000
        assert summary["max_salary"] == 120000
        assert summary["min_salary"] == 80000

    def test_generate_summary_department_totals(self):
        df = pd.DataFrame(
            {
                "department": ["Engineering", "Sales", "Engineering", "Marketing"],
                "salary": [100000, 80000, 120000, 90000],
            }
        )
        summary = generate_summary(df)

        assert summary["department_engineering"] == 220000
        assert summary["department_sales"] == 80000
        assert summary["department_marketing"] == 90000

    def test_generate_summary_returns_dict(self):
        df = pd.DataFrame({"department": ["HR"], "salary": [70000]})
        summary = generate_summary(df)
        assert isinstance(summary, dict)

    def test_generate_summary_single_employee(self):
        df = pd.DataFrame({"department": ["Finance"], "salary": [95000]})
        summary = generate_summary(df)

        assert summary["total_employees"] == 1
        assert summary["average_salary"] == 95000
        assert summary["max_salary"] == 95000
        assert summary["min_salary"] == 95000
        assert "department_finance" in summary

    def test_generate_summary_empty_department_key(self):
        df = pd.DataFrame(
            {"department": ["", "IT", ""], "salary": [50000, 60000, 55000]}
        )
        summary = generate_summary(df)

        assert "department_" in summary
        assert summary["department_"] == 105000


class TestSaveReport:
    def test_save_report_csv_creates_file(self):
        summary = {
            "total_employees": 5,
            "average_salary": 100000,
            "max_salary": 150000,
            "min_salary": 50000,
            "department_engineering": 500000,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "report.csv")
            save_report(summary, out_path)

            assert os.path.exists(out_path)

    def test_save_report_csv_content(self):
        summary = {
            "total_employees": 5,
            "average_salary": 100000,
            "max_salary": 150000,
            "min_salary": 50000,
            "department_engineering": 500000,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "report.csv")
            save_report(summary, out_path)

            df = pd.read_csv(out_path)
            assert len(df) == 1
            assert df.loc[0, "total_employees"] == 5
            assert df.loc[0, "average_salary"] == 100000

    def test_save_report_creates_parent_directories(self):
        summary = {"total_employees": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "subdir1", "subdir2", "report.csv")
            save_report(summary, out_path)

            assert os.path.exists(out_path)

    def test_save_report_unsupported_format_raises_error(self):
        summary = {"total_employees": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "report.txt")
            with pytest.raises(ValueError, match="Unsupported file format"):
                save_report(summary, out_path)

    def test_save_report_unsupported_uppercase_extension(self):
        summary = {"total_employees": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "report.CSV")
            save_report(summary, out_path)

            assert os.path.exists(out_path)
