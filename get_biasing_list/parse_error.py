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
    errors_list = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            # Read file content
            content = file.readlines()
            
            for line in content:
                # Check if the line contains utterances by the participant
                if line.startswith('*PAR:'):
                    # Find all timestamps in the line
                    timestamp_pattern = r'\d+_\d+'
                    timestamps = re.findall(timestamp_pattern, line)
                    
                    if not timestamps:
                        continue
                    
                    # Set the timestamp for the utterance
                    timestamp = timestamps[-1]  # Use the last timestamp found
                    
                    # Extract the surrounding context (the whole utterance)
                    utterance = line.strip().replace('*PAR:', '').strip()
                    
                    # Remove the timestamp from the end of the utterance
                    # Split by the timestamp and take everything before it
                    parts = utterance.split(timestamp)
                    if len(parts) > 1:
                        utterance = parts[0].strip()
                    
                    # Find errors in the line - two patterns:
                    
                    # Pattern 1: For phonetic errors with @u notation
                    # Format: word@u [: target] [* error_type]
                    phonetic_pattern = r'([^\[\s]+@u)\s+\[:\s+([^\]]+)\]\s+\[\*\s+([^\]]+)\]'
                    phonetic_matches = re.findall(phonetic_pattern, line)
                    
                    # Pattern 2: For regular word errors
                    # Format: word [: target] [* error_type]
                    word_pattern = r'(\b[^\[\s]+\b)\s+\[:\s+([^\]]+)\]\s+\[\*\s+([^\]]+)\]'
                    word_matches = re.findall(word_pattern, line)
                    
                    # Process phonetic errors
                    for error_match in phonetic_matches:
                        pronunciation = error_match[0]
                        target = error_match[1]
                        error_type = error_match[2]
                        
                        errors_list.append({
                            'timestamp': timestamp,
                            'pronunciation': pronunciation,
                            'target': target,
                            'error_type': error_type,
                            'utterance': utterance,
                            'filename': os.path.basename(filepath).split('.')[0],
                            'error_notation': 'phonetic transcription'  
                        })
                    
                    # Process word errors
                    for error_match in word_matches:
                        pronunciation = error_match[0]
                        target = error_match[1]
                        error_type = error_match[2]
                        
                        # Skip if this is already captured as a phonetic error (has @u)
                        if '@u' in pronunciation:
                            continue
                        
                        errors_list.append({
                            'timestamp': timestamp,
                            'pronunciation': pronunciation,
                            'target': target,
                            'error_type': error_type,
                            'utterance': utterance,
                            'filename': os.path.basename(filepath).split('.')[0],
                            'error_notation': 'word-level'  # Indicate this is a word error
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
        'p:n': 'phonological error resulting in a nonword',
        'p:m': 'phonemic paraphasia',
        'p:w': 'phonemic paraphasia affecting a whole word',
        'n:k': 'neologism with known target',
        'n:uk': 'neologism with unknown target',
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
            error['error_explanation'] = get_error_type_classification(error['error_type'])
        
        all_errors.extend(errors)
        print(f"  Found {len(errors)} errors in this file")
    
    # Create DataFrame
    new_df = pd.DataFrame(all_errors)
    
    # Save to CSV if output path is provided
    if output_path:
        # Convert the output path to CSV if it's not already
        if not output_path.endswith('.csv'):
            output_path = output_path.rsplit('.', 1)[0] + '.csv'
        
        try:
            # Check if the file already exists
            if os.path.exists(output_path):
                print(f"File {output_path} already exists. Appending new data...")
                # Read existing CSV
                existing_df = pd.read_csv(output_path, encoding='utf-8')
                
                # Concatenate with new data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # Remove duplicates if needed (based on all columns)
                combined_df = combined_df.drop_duplicates()
                
                # Save the combined data
                combined_df.to_csv(output_path, index=False, encoding='utf-8')
                print(f"Appended {len(new_df)} new errors to {output_path}")
                return combined_df
            else:
                # If file doesn't exist, create a new one
                new_df.to_csv(output_path, index=False, encoding='utf-8')
                print(f"Errors saved to {output_path}")
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")
    
    return new_df

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