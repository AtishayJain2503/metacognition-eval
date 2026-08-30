"""
tests.unit.test_tools.test_sqlite_schema
----------------------------------------
Unit tests for SQLite schema introspection, DDL extraction, FKs, and BLOB sanitization.
"""

import pytest

from nemo_eval.tools.sqlite_engine import SQLiteEngine, SQLiteEngineConfig


class TestSQLiteSchema:
    """Tests for SQLite schema introspection and metadata profiling."""

    def test_schema_introspection_multi_table(self, sample_sqlite_engine):
        schema_resp = sample_sqlite_engine.get_schema()
        assert schema_resp.table_count == 2
        assert "categories" in schema_resp.tables
        assert "products" in schema_resp.tables

        products_tbl = schema_resp.tables["products"]
        assert products_tbl.name == "products"
        assert products_tbl.type == "table"
        assert products_tbl.row_count == 5
        assert "CREATE TABLE products" in products_tbl.ddl

        # Verify columns
        col_names = [c.name for c in products_tbl.columns]
        assert "product_id" in col_names
        assert "name" in col_names
        assert "price" in col_names
        assert "category_id" in col_names

        # Verify primary keys
        assert products_tbl.primary_keys == ["product_id"]

        # Verify foreign keys
        assert len(products_tbl.foreign_keys) == 1
        fk = products_tbl.foreign_keys[0]
        assert fk.from_column == "category_id"
        assert fk.referenced_table == "categories"
        assert fk.referenced_column == "category_id"

        # Verify sample rows
        assert len(products_tbl.sample_rows) == 3
        assert products_tbl.sample_rows[0]["name"] == "Laptop"

    def test_schema_introspection_single_table(self, sample_sqlite_engine):
        schema_resp = sample_sqlite_engine.get_schema(table_name="categories")
        assert schema_resp.table_count == 1
        assert "categories" in schema_resp.tables
        assert "products" not in schema_resp.tables

    def test_schema_blob_sanitization(self):
        engine = SQLiteEngine(SQLiteEngineConfig(read_only=False))
        engine.init_from_sql("""
            CREATE TABLE blob_table (id INT, raw_data BLOB);
            INSERT INTO blob_table VALUES (1, X'504B0304');
        """)
        schema_resp = engine.get_schema("blob_table")
        tbl = schema_resp.tables["blob_table"]
        assert len(tbl.sample_rows) == 1
        assert tbl.sample_rows[0]["raw_data"] == "<BLOB len=4>"
        engine.close()

    def test_schema_tool_wrapper(self, sample_sqlite_engine):
        res = sample_sqlite_engine.schema_tool()
        assert res.status == "success"
        assert res.data["table_count"] == 2
        assert "products" in res.data["tables"]
