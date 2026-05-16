#!/usr/bin/env python
"""
Initialize database tables
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import DatabaseManager, Base


def init_database():
    """Create all database tables"""
    print("Initializing database...")
    
    db_manager = DatabaseManager()
    db_manager.connect()
    
    print("Database tables created successfully!")
    print(f"Database type: {db_manager._config['db_type']}")
    
    if db_manager._config['db_type'] == 'sqlite':
        print(f"Database file: {db_manager._config['db_path']}")


def drop_database():
    """Drop all database tables (use with caution!)"""
    confirm = input("Are you sure you want to drop all tables? (yes/no): ")
    if confirm.lower() == 'yes':
        db_manager = DatabaseManager()
        db_manager.connect()
        Base.metadata.drop_all(db_manager._engine)
        print("All tables dropped!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop', action='store_true', help='Drop existing tables')
    args = parser.parse_args()
    
    if args.drop:
        drop_database()
    else:
        init_database()