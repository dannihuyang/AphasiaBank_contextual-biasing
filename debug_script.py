#!/usr/bin/env python3
import os
import pandas as pd
import re
import sys

def debug_excel_file(excel_file, audio_path):
    """Debug function to check Excel file contents and path issues"""
    print(f"\n=== DIAGNOSTIC INFO ===")
    
    # Check if files exist
    print(f"Checking if Excel file exists: {excel_file}")
    if os.path.exists(excel_file):
        print(f"✓ Excel file exists")
    else:
        print(f"✗ Excel file NOT found at path: {excel_file}")
        return
    
    print(f"Checking if audio file exists: {audio_path}")
    if os.path.exists(audio_path):
        print(f"✓ Audio file exists")
    else:
        print(f"✗ Audio file NOT found at path: {audio_path}")
        return
    
    # Try to read Excel file
    print(f"\nReading Excel file...")
    try:
        df = pd.read_excel(excel_file)
        print(f"✓ Excel file loaded successfully")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {', '.join(df.columns)}")
        
        # Check for timestamp column
        if 'timestamp' in df.columns:
            print(f"✓ 'timestamp' column found")
            
            # Check for values in timestamp column
            timestamp_count = 0
            
            for i, row in df.iterrows():
                timestamp = str(row.get('timestamp', ''))
                
                # Try to split by underscore, dash, or any non-digit character
                if '_' in timestamp:
                    parts = timestamp.split('_')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        timestamp_count += 1
                        if i < 3:  # Show max 3 examples
                            print(f"\nRow {i} has timestamp: {timestamp}")
                elif '-' in timestamp:
                    parts = timestamp.split('-')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        timestamp_count += 1
                        if i < 3:  # Show max 3 examples
                            print(f"\nRow {i} has timestamp: {timestamp}")
                elif re.search(r'\d+\D+\d+', timestamp):
                    timestamp_count += 1
                    if i < 3:  # Show max 3 examples
                        print(f"\nRow {i} has timestamp: {timestamp}")
            
            if timestamp_count == 0:
                print(f"✗ No valid timestamps found in timestamp column")
                print(f"   Expected format: START_END, START-END, or numbers separated by non-digits")
            else:
                print(f"\n✓ Found {timestamp_count} timestamps across {len(df)} rows")
        else:
            print(f"✗ 'timestamp' column NOT found in Excel file")
            print(f"   Available columns: {', '.join(df.columns)}")
            
    except Exception as e:
        print(f"✗ Error reading Excel file: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== END DIAGNOSTIC INFO ===")
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python debug_script.py excel_file audio_path")
        sys.exit(1)
        
    debug_excel_file(sys.argv[1], sys.argv[2])