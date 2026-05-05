import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import load_csv, validate_file
import tempfile
import os


class TestValidateFile:
    def test_validate_existing_csv_file(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("col1,col2\n1,2\n")
            temp_path = f.name
        try:
            validate_file(temp_path)
        except Exception:
            pytest.fail("validate_file raised an exception for valid file")
        finally:
            os.unlink(temp_path)

    def test_validate_nonexistent_file(self):
        with pytest.raises(FileNotFoundError, match="Path not found"):
            validate_file("nonexistent_file.csv")

    def test_validate_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="file not supported"):
                validate_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_validate_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="empty file"):
                validate_file(temp_path)
        finally:
            os.unlink(temp_path)


class TestLoadCsv:
    def test_load_csv_success(self):
        test_file = os.path.join(os.path.dirname(__file__), "..", "data", "test.csv")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("col1,col2\n1,2\n3,4\n")
            temp_path = f.name
        try:
            df = load_csv(temp_path)
            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == ["col1", "col2"]
            assert len(df) == 2
        finally:
            os.unlink(temp_path)

    def test_load_csv_nonexistent_file(self):
        with pytest.raises(FileNotFoundError, match="Path not found"):
            load_csv("nonexistent_file.csv")

    def test_load_csv_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="file not supported"):
                load_csv(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_csv_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="empty file"):
                load_csv(temp_path)
        finally:
            os.unlink(temp_path)