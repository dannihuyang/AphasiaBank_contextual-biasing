import re
import pandas as pd
import os
import glob
import sys

def extract_errors(filepath):
    """
    Extract errors from AphasiaBank .kwal.cex files
    
    Parameters:
    filepath (str): Path to the .kwal.cex file
    
    Returns:
    list: List of dictionaries containing error information
    """
    errors_list = []  # Renamed to avoid confusion with the variable used in the loop
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            # Read file content
            content = file.readlines()
            
            for line in content:
                # Check if the line contains utterances by the participant
                if line.startswith('*PAR:'):
                    # Find all timestamps in the line
                    timestamp_pattern = r'(\d+_\d+)'
                    timestamps = re.findall(timestamp_pattern, line)
                    
                    if not timestamps:
                        continue
                    
                    # Set the timestamp range for the entire utterance
                    if len(timestamps) >= 2:
                        timestamp = f"{timestamps[0]}_{timestamps[-1]}"
                    else:
                        timestamp = timestamps[0]
                    
                    # Find errors in the line
                    # Format: word@u [: target] [* error_type]
                    error_pattern = r'(\w+@u)\s+\[:\s+([^\]]+)\]\s+\[\*\s+([^\]]+)\]'
                    error_matches = re.findall(error_pattern, line)
                    
                    for error_match in error_matches:
                        pronunciation = error_match[0]  # This is a tuple, not a dict
                        # pronunciation = error_match[0].replace('@u', '')
                        target = error_match[1]
                        error_type = error_match[2]
                        
                        # Extract the surrounding context (the whole utterance)
                        utterance = line.strip().replace('*PAR:', '').strip()
                        
                        errors_list.append({
                            'timestamp': timestamp,
                            'pronunciation': pronunciation,
                            'target': target,
                            'error_type': error_type,
                            'utterance': utterance,
                            'filename': os.path.basename(filepath)
                        })
    
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
    
    return errors_list

def get_error_type_classification(error_type):
    """
    Classify the error type into more general categories
    
    Parameters:
    error_type (str): The error type code from the transcript
    
    Returns:
    str: A general classification of the error
    """
    # Define classification mapping
    classifications = {
        'p:n': 'phonological substitution',
        'p:m': 'phonemic paraphasia',
        'p:w': 'word-level phonological error',
        'n:k': 'neologism with known target',
        'n:uk': 'unknown neologism',
        's:r': 'semantic relation error',
        's:ur': 'unrelated semantic error',
        's:r:gc:pro': 'gender confusion pronoun error',
    }
    
    # Return the classification if found, otherwise return the original code
    return classifications.get(error_type, error_type)

def process_directory(directory_path, output_path=None):
    """
    Process all .kwal.cex files in a directory and extract errors with classifications
    
    Parameters:
    directory_path (str): Path to directory containing .kwal.cex files
    output_path (str, optional): Path to save the output CSV file
    
    Returns:
    pandas.DataFrame: DataFrame containing error information with classifications
    """
    all_errors = []
    
    # Find all .kwal.cex files in the directory
    kwal_files = glob.glob(os.path.join(directory_path, '*.kwal.cex'))
    
    if not kwal_files:
        print(f"No .kwal.cex files found in {directory_path}")
        return pd.DataFrame()
    
    print(f"Found {len(kwal_files)} .kwal.cex files in {directory_path}")
    
    for kwal_file in kwal_files:
        print(f"Processing {kwal_file}...")
        errors = extract_errors(kwal_file)
        
        # Add error type classification
        for error in errors:
            error['error_classification'] = get_error_type_classification(error['error_type'])
        
        all_errors.extend(errors)
        print(f"  Found {len(errors)} errors in this file")
    
    # Create DataFrame
    df = pd.DataFrame(all_errors)
    
    # Save to CSV if output path is provided
    if output_path:
        # Convert the output path to CSV if it's not already
        if not output_path.endswith('.csv'):
            output_path = output_path.rsplit('.', 1)[0] + '.csv'
        
        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Errors saved to {output_path}")
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")
    
    return df

if __name__ == "__main__":
    # Check if directory path is provided as command line argument
    if len(sys.argv) < 2:
        print("Usage: python -m get_biasing_list.parse_error <directory_path> [output_file.csv]")
        sys.exit(1)
    
    # Get directory path from command line
    directory_path = sys.argv[1]
    
    # Get output path from command line or use default
    output_path = sys.argv[2] if len(sys.argv) > 2 else "errors.csv"
    
    # Process the directory
    df = process_directory(directory_path, output_path)
    
    # Print summary
    print(f"\nSummary:")
    print(f"Processed {directory_path}")
    print(f"Found {len(df)} total errors")
    print(f"Results saved to {output_path}")