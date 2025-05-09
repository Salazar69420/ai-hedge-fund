import os
import shutil
from pathlib import Path

def move_csv_files():
    # Define source and destination directories
    source_dir = 'data'
    dest_dir = os.path.join(source_dir, 'parsed_data')
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all CSV files in the source directory
    csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the data directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files to move.")
    
    # Move each CSV file
    for csv_file in csv_files:
        source_path = os.path.join(source_dir, csv_file)
        dest_path = os.path.join(dest_dir, csv_file)
        
        try:
            # Move the file
            shutil.move(source_path, dest_path)
            print(f"Moved: {csv_file}")
        except Exception as e:
            print(f"Error moving {csv_file}: {str(e)}")
    
    print(f"\nSuccessfully moved {len(csv_files)} CSV files to {dest_dir}")

if __name__ == "__main__":
    move_csv_files() 