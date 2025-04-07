import re
import pandas as pd
import argparse
import os
import sys

def extract_clean_transcription(utterance):
    """
    Extract the clean, intended transcription from an AphasiaBank utterance.
    
    This function removes all CLAN notation and extracts what the speaker
    intended to say, including replacing error words with their targets.
    
    Parameters:
    utterance (str): The raw AphasiaBank utterance with CLAN notation
    
    Returns:
    str: The clean transcription with all notation removed
    """
    if not utterance or not isinstance(utterance, str):
        return ''
    
    cleaned = utterance
    
    # STEP 1: Replace all error annotations with their targets
    # Format: word@u [: target] [* error_type]
    cleaned = re.sub(r'(\w+)@u\s+\[:\s+([^\]]+)\]\s+\[\*\s+[^\]]+\]', r'\2', cleaned)
    
    # Format: word [: target] [* error_type]
    cleaned = re.sub(r'(\b\w+\b)\s+\[:\s+([^\]]+)\]\s+\[\*\s+[^\]]+\]', r'\2', cleaned)
    
    # Handle special case for neologisms marked as x@n
    cleaned = re.sub(r'\w+\s+\[:\s+x@n\]\s+\[\*\s+[^\]]+\]', '', cleaned)
    
    # STEP 2: Remove non-linguistic elements
    # Remove gesture annotations &=gesture
    cleaned = re.sub(r'&=[^\s]+', '', cleaned)
    
    # Remove sounds and fillers &-sound, &+sound
    cleaned = re.sub(r'&[\-+][^\s]+', '', cleaned)
    
    # Remove pause notations (..), (...)
    cleaned = re.sub(r'\([\.]+\)', '', cleaned)
    cleaned = re.sub(r'\(\.\)', '', cleaned)
    
    # Remove other annotations like [+ gram], [+ exc], etc.
    cleaned = re.sub(r'\[\+\s*[\w:]+\]', '', cleaned)
    
    # STEP 3: Handle retracing and repetitions
    # Handle retracing with angle brackets <text> [//]
    previous = ""
    while previous != cleaned:
        previous = cleaned
        # Remove retraced material (corrections)
        cleaned = re.sub(r'<([^>]+)>\s*\[\/\/\]', '', cleaned)
    
    # Handle word-level retracing
    cleaned = re.sub(r'(\b\w+\b)\s*\[\/\/\]', '', cleaned)
    
    # Handle repetitions <text> [/]
    previous = ""
    while previous != cleaned:
        previous = cleaned
        # Keep content of repetitions but remove markers
        cleaned = re.sub(r'<([^>]+)>\s*\[\/\]', r'\1', cleaned)
    
    # Handle word-level repetitions
    cleaned = re.sub(r'(\b\w+\b)\s*\[\/\]', r'\1', cleaned)
    
    # STEP 4: Remove other CLAN notation
    # Remove discourse markers like ‡
    cleaned = re.sub(r'‡', '', cleaned)
    
    # Remove truncation/continuation markers
    cleaned = re.sub(r'\+\.\.\.', '', cleaned)
    cleaned = re.sub(r'\+\/\.', '.', cleaned)
    cleaned = re.sub(r'\+\/\?', '?', cleaned)
    
    # Handle direct speech notation
    cleaned = re.sub(r'\+\"', '"', cleaned)
    cleaned = re.sub(r'\"\\+', '"', cleaned)
    
    # STEP 5: Remove any remaining angle brackets
    cleaned = re.sub(r'<|>', '', cleaned)
    
    # STEP 6: Clean up final text
    # Fix spacing around punctuation
    cleaned = re.sub(r'\s+\.', '.', cleaned)
    cleaned = re.sub(r'\s+\,', ',', cleaned)
    cleaned = re.sub(r'\s+\?', '?', cleaned)
    cleaned = re.sub(r'\s+\!', '!', cleaned)
    
    # Remove consecutive duplicate words (common in transcription)
    words = cleaned.split()
    deduped_words = []
    for i, word in enumerate(words):
        if i == 0 or word.lower() != words[i-1].lower():
            deduped_words.append(word)
    
    cleaned = ' '.join(deduped_words)
    
    # Remove multiple spaces and trim
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def process_csv(input_file, output_file=None, utterance_column='utterance', clean_column='cleaned_utterance'):
    """
    Process a CSV file containing AphasiaBank utterances.
    
    Parameters:
    input_file (str): Path to the input CSV file
    output_file (str, optional): Path to save the output CSV file. If not provided,
                                will modify the input file name.
    utterance_column (str): Name of the column containing utterances (default: 'utterance')
    clean_column (str): Name of the column to store clean transcriptions (default: 'cleaned_utterance')
    
    Returns:
    pandas.DataFrame: DataFrame with cleaned transcriptions
    """
    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        
        # Check if specified utterance column exists
        if utterance_column not in df.columns:
            print(f"Error: CSV file does not have a '{utterance_column}' column")
            print(f"Available columns: {', '.join(df.columns)}")
            return None
        
        # Create a new column for clean transcriptions
        print(f"Extracting clean transcriptions from '{utterance_column}' column...")
        df[clean_column] = df[utterance_column].apply(extract_clean_transcription)
        
        # Save to output file
        if not output_file:
            # Create output filename based on input filename
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_clean.csv"
        
        print(f"Saving to {output_file}")
        df.to_csv(output_file, index=False)
        print(f"Processed {len(df)} utterances")
        
        # Print a few examples
        sample_size = min(5, len(df))
        print(f"\nExample of {sample_size} processed utterances:")
        
        for i, row in df.head(sample_size).iterrows():
            print(f"Original: {row[utterance_column]}")
            print(f"Cleaned : {row[clean_column]}")
            print("-" * 50)
        
        return df
    
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Clean AphasiaBank transcriptions in a CSV file')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('--output', '-o', default=None, 
                        help='Path to save the output CSV file (default: input_filename_with_clean.csv in the same directory)')
    parser.add_argument('--column', '-c', default='utterance', 
                        help='Name of the column containing utterances (default: utterance)')
    parser.add_argument('--clean-column', '-n', default='cleaned_utterance',
                        help='Name of the column to store clean transcriptions (default: cleaned_utterance)')
    
    args = parser.parse_args()
    
    # Process the CSV file
    process_csv(args.input_file, args.output, args.column, args.clean_column)

if __name__ == "__main__":
    main()