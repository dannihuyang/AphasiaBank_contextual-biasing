import pandas as pd
import argparse
import re
import os

def parse_csv_and_write_to_txt(csv_file, output_folder, filename_start=None, error_start_letter='p', normalize=False):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Optionally filter by filenames starting with a specific string
    if filename_start:
        df = df[df['filename'].str.startswith(filename_start, na=False)]
    
    # Filter rows where 'error_type' starts with the specified letter
    filtered_df = df[df['error_type'].str.startswith(error_start_letter, na=False)]
    
    # Get unique filenames from the filtered DataFrame
    unique_filenames = filtered_df['filename'].unique()
    
    for unique_filename in unique_filenames:
        # Filter rows for the current unique filename
        filename_df = filtered_df[filtered_df['filename'] == unique_filename]
        
        # Extract unique 'target' phrases
        unique_targets = filename_df['target'].dropna().unique()
        
        # Collect all words from target phrases
        all_words = set()
        for target in unique_targets:
            words = target.split()
            for word in words:
                if normalize:
                    # Remove parentheses and content within them, e.g., "lef(t)" to "left"
                    word = re.sub(r'\((\w)\)', r'\1', word)
                word = word.strip()  # Remove any leading or trailing whitespace
                if word:  # Only add non-empty words
                    all_words.add(word)
        
        # Sort the words alphabetically
        sorted_words = sorted(all_words)
        
        # Determine the output file name
        output_filename = f"biasing_list_{unique_filename}.txt"
        output_path = os.path.join(output_folder, output_filename)
        
        # Write sorted words to a text file
        with open(output_path, 'w') as f:
            for word in sorted_words:
                if word:  # Ensure no empty words are written
                    f.write(f"{word}\n")
        
        # Remove any trailing newlines or whitespace from the file
        with open(output_path, 'r+') as f:
            lines = f.readlines()
            f.seek(0)
            f.writelines(line for line in lines if line.strip())
            f.truncate()

def main():
    parser = argparse.ArgumentParser(description="Parse a CSV file and extract unique target phrases.")
    parser.add_argument("csv_file", help="Path to the CSV file to parse")
    parser.add_argument("--filename-start", "-fs", help="Starting string of filename to filter by")
    parser.add_argument("--error-start", "-e", default='p', help="Starting letter of error_type to filter by")
    parser.add_argument("--normalize", "-n", action="store_true", help="Normalize words by removing parentheses and their contents")
    
    args = parser.parse_args()
    
    # Load the CSV file to determine the common prefix
    df = pd.read_csv(args.csv_file)
    if args.filename_start:
        df = df[df['filename'].str.startswith(args.filename_start, na=False)]
    
    # Extract the common prefix from the filename column
    filenames = df['filename'].dropna().unique().tolist()
    common_prefix = os.path.commonprefix(filenames)
    common_prefix = re.sub(r'\d+$', '', common_prefix)  # Remove trailing numbers
    
    # Create a directory for the biasing lists under the output folder
    output_folder = os.path.join("output", f"biasing_list_{common_prefix}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Process the specified CSV file
    parse_csv_and_write_to_txt(args.csv_file, output_folder, args.filename_start, args.error_start, args.normalize)

if __name__ == "__main__":
    main()