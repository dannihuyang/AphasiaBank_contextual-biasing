import re
import pandas as pd
import os
import glob
import sys

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
    
    # Clean up the error type
    error_type = error_type.strip()
    
    # Check for exact matches first
    if error_type in classifications:
        return classifications[error_type]
    
    # Check for partial matches (starts with)
    for prefix, classification in classifications.items():
        if error_type.startswith(prefix):
            return classification
    
    # Default classification
    return "other error type"

def process_file(filepath):
    """
    Process a .kwal.cex file and extract all utterances with error information
    
    Parameters:
    filepath (str): Path to the .kwal.cex file
    
    Returns:
    list: List of dictionaries containing utterance information
    """
    utterances_list = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.readlines()
            
            for line in content:
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
                    parts = utterance.split(timestamp)
                    if len(parts) > 1:
                        utterance = parts[0].strip()
                    
                    # Check if utterance contains errors
                    has_error = '[*' in utterance
                    
                    # Create base utterance entry
                    utterance_entry = {
                        'timestamp': timestamp,
                        'utterance': utterance,
                        'filename': os.path.basename(filepath).split('.')[0],
                        'has_error': has_error
                    }
                    
                    # If there are errors, extract them
                    if has_error:
                        # Pattern 1: For phonetic errors with @u notation
                        phonetic_pattern = r'([^\[\s]+@u)\s+\[:\s+([^\]]+)\]\s+\[\*\s+([^\]]+)\]'
                        phonetic_matches = re.findall(phonetic_pattern, line)
                        
                        # Pattern 2: For regular word errors
                        word_pattern = r'(\b[^\[\s]+\b)\s+\[:\s+([^\]]+)\]\s+\[\*\s+([^\]]+)\]'
                        word_matches = re.findall(word_pattern, line)
                        
                        # Process phonetic errors
                        for error_match in phonetic_matches:
                            pronunciation = error_match[0]
                            target = error_match[1]
                            error_type = error_match[2]
                            
                            # Create a copy of the base entry and add error details
                            entry = utterance_entry.copy()
                            entry.update({
                                'pronunciation': pronunciation,
                                'target': target,
                                'error_type': error_type,
                                'error_notation': 'phonetic',
                                'error_explanation': get_error_type_classification(error_type)
                            })
                            utterances_list.append(entry)
                        
                        # Process word errors
                        for error_match in word_matches:
                            pronunciation = error_match[0]
                            target = error_match[1]
                            error_type = error_match[2]
                            
                            # Skip if this is already captured as a phonetic error (has @u)
                            if '@u' in pronunciation:
                                continue
                            
                            # Create a copy of the base entry and add error details
                            entry = utterance_entry.copy()
                            entry.update({
                                'pronunciation': pronunciation,
                                'target': target,
                                'error_type': error_type,
                                'error_notation': 'word',
                                'error_explanation': get_error_type_classification(error_type)
                            })
                            utterances_list.append(entry)
                    else:
                        # If no errors, just add the base entry
                        utterances_list.append(utterance_entry)
    
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
    
    return utterances_list

def process_directory(directory_path, output_path=None):
    """
    Process all .kwal.cex files in a directory
    
    Parameters:
    directory_path (str): Path to directory containing .kwal.cex files
    output_path (str, optional): Path to save the output CSV file
    
    Returns:
    pandas.DataFrame: DataFrame containing utterance information with error details
    """
    all_utterances = []
    
    # Find all .kwal.cex files in the directory
    kwal_files = glob.glob(os.path.join(directory_path, '*.kwal.cex'))
    
    if not kwal_files:
        print(f"No .kwal.cex files found in {directory_path}")
        return pd.DataFrame()
    
    print(f"Found {len(kwal_files)} .kwal.cex files in {directory_path}")
    
    for kwal_file in kwal_files:
        print(f"Processing {kwal_file}...")
        
        # Process the file to get utterances with error information
        utterances = process_file(kwal_file)
        all_utterances.extend(utterances)
        
        print(f"  Found {len(utterances)} utterances in this file")
    
    # Create DataFrame
    df = pd.DataFrame(all_utterances)
    
    # Save to CSV if output path is provided
    if output_path:
        try:
            # Check if the file already exists
            if os.path.exists(output_path):
                print(f"File {output_path} already exists. Appending new data...")
                # Read existing CSV
                existing_df = pd.read_csv(output_path, encoding='utf-8')
                
                # Concatenate with new data
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                
                # Remove duplicates based on timestamp, filename, and pronunciation (if present)
                if 'pronunciation' in combined_df.columns:
                    duplicate_cols = ['timestamp', 'filename', 'pronunciation']
                else:
                    duplicate_cols = ['timestamp', 'filename']
                
                combined_df = combined_df.drop_duplicates(subset=duplicate_cols, keep='first')
                
                # Save the combined data
                combined_df.to_csv(output_path, index=False, encoding='utf-8')
                print(f"Appended new data to {output_path}")
                df = combined_df
            else:
                # If file doesn't exist, create a new one
                df.to_csv(output_path, index=False, encoding='utf-8')
                print(f"Data saved to {output_path}")
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")
    
    return df

if __name__ == "__main__":
    # Check if directory path is provided as command line argument
    if len(sys.argv) < 2:
        print("Usage: python -m get_biasing_list.parse_error <directory_path> [output.csv]")
        sys.exit(1)
    
    # Get directory path from command line
    directory_path = sys.argv[1]
    
    # Get output path from command line or use default
    output_path = sys.argv[2] if len(sys.argv) > 2 else "utterances_with_errors.csv"
    
    # Process the directory
    df = process_directory(directory_path, output_path)
    
    # Print summary
    print(f"\nSummary:")
    print(f"Processed {directory_path}")
    print(f"Found {len(df)} total utterances")
    print(f"Of which {df['has_error'].sum()} contain errors")
    print(f"Results saved to {output_path}")