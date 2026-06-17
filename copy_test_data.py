#!/usr/bin/env python3
"""
Script to copy schema and first N rows from chess_bot.db to test_chess_bot.db
"""
import sqlite3
import os

SOURCE_DB = "chess_bot.db"
TARGET_DB = "test_chess_bot.db"
ROWS_TO_COPY = 10  # Number of rows to copy from each table

def copy_database():
    # Connect to source database
    source_conn = sqlite3.connect(SOURCE_DB)
    source_conn.row_factory = sqlite3.Row
    source_cursor = source_conn.cursor()
    
    # Connect to target database
    target_conn = sqlite3.connect(TARGET_DB)
    target_cursor = target_conn.cursor()
    
    # Drop existing tables if they exist in target
    tables = ["games", "moves", "self_play_moves", "move_sequences"]
    for table in tables:
        target_cursor.execute(f"DROP TABLE IF EXISTS {table}")
    
    # Copy schema from source to target
    source_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    schema_rows = source_cursor.fetchall()
    
    for row in schema_rows:
        if row['sql']:
            target_cursor.execute(row['sql'])
    
    # Copy data from each table
    for table in tables:
        # Get first N rows from source
        source_cursor.execute(f"SELECT * FROM {table} LIMIT ?", (ROWS_TO_COPY,))
        rows = source_cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in source_cursor.description]
        
        # Insert into target
        placeholders = ', '.join(['?' for _ in column_names])
        columns_str = ', '.join(column_names)
        insert_sql = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"
        
        for row in rows:
            values = [row[col] for col in column_names]
            target_cursor.execute(insert_sql, values)
    
    # Copy indexes
    source_cursor.execute("SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")
    index_rows = source_cursor.fetchall()
    
    for row in index_rows:
        if row['sql']:
            try:
                target_cursor.execute(row['sql'])
            except sqlite3.Error as e:
                print(f"Warning: Could not create index: {e}")
    
    # Commit and close
    target_conn.commit()
    source_conn.close()
    target_conn.close()
    
    print(f"Successfully copied schema and {ROWS_TO_COPY} rows from each table to {TARGET_DB}")

if __name__ == "__main__":
    copy_database()
