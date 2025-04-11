import re
import pandas as pd
import argparse
import os
import sys
import csv

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
    cleaned = re.sub(r'(\w+)@u\s+\[:\s+([^\]]+)\]\s+\[\*\s+[^\]]+\]', r'\2', cleaned)
    cleaned = re.sub(r'(\b\w+\b)\s+\[:\s+([^\]]+)\]\s+\[\*\s+[^\]]+\]', r'\2', cleaned)
    cleaned = re.sub(r'\w+\s+\[:\s+x@n\]\s+\[\*\s+[^\]]+\]', '', cleaned)
    
    # STEP 2: Handle parenthesized text (ENHANCED)
    cleaned = re.sub(r'\(([^)]*)\)', r'\1', cleaned)
    
    # STEP 2.1: Remove CLAN specific notations
    cleaned = re.sub(r'&=[^\s]+', '', cleaned)
    cleaned = re.sub(r'&[\-+][^\s]+', '', cleaned)
    cleaned = re.sub(r'\([\.]+\)', '', cleaned)
    cleaned = re.sub(r'\(\.\)', '', cleaned)
    cleaned = re.sub(r'\[\+\s*[\w:]+\]', '', cleaned)
    
    # STEP 3: Handle retracing and repetitions
    previous = ""
    while previous != cleaned:
        previous = cleaned
        cleaned = re.sub(r'<([^>]+)>\s*\[\/\/\]', '', cleaned)
    cleaned = re.sub(r'(\b\w+\b)\s*\[\/\/\]', '', cleaned)
    
    previous = ""
    while previous != cleaned:
        previous = cleaned
        cleaned = re.sub(r'<([^>]+)>\s*\[\/\]', r'\1', cleaned)
    cleaned = re.sub(r'(\b\w+\b)\s*\[\/\]', r'\1', cleaned)
    cleaned = re.sub(r'<|>', '', cleaned)
    
    # STEP 4: Remove other CLAN notation
    cleaned = re.sub(r'‡', '', cleaned)
    cleaned = re.sub(r'\+[\.\s]*\.\.', '', cleaned)
    cleaned = re.sub(r'\+\/\/\.?', '', cleaned)
    cleaned = re.sub(r'\+\s*\/\/', '', cleaned)
    cleaned = re.sub(r'\+\s*\.\.', '', cleaned)
    cleaned = re.sub(r'\+\s*\.\.\.', '', cleaned)
    cleaned = re.sub(r'\+\/', '', cleaned)
    cleaned = re.sub(r'\+\s*\/', '', cleaned)
    cleaned = re.sub(r'\+\/\.', '.', cleaned)
    cleaned = re.sub(r'\+\/\?', '?', cleaned)
    cleaned = re.sub(r'\s\+\s', ' ', cleaned)
    cleaned = re.sub(r'^\+\s', '', cleaned)
    cleaned = re.sub(r'\+\"', '', cleaned)
    cleaned = re.sub(r'\"\\+', '', cleaned)
    cleaned = re.sub(r'\"\/\.', '', cleaned)
    cleaned = re.sub(r'\/\.', '', cleaned)
    cleaned = re.sub(r'[""]', '', cleaned)
    
    # STEP 5: Remove any remaining angle brackets
    cleaned = re.sub(r'<|>', '', cleaned)
    
    # STEP 6: Clean up final text
    cleaned = re.sub(r'\s+\.', '.', cleaned)
    cleaned = re.sub(r'\s+\,', ',', cleaned)
    cleaned = re.sub(r'\s+\?', '?', cleaned)
    cleaned = re.sub(r'\s+\!', '!', cleaned)
    
    words = cleaned.split()
    deduped_words = []
    for i, word in enumerate(words):
        if i == 0 or word.lower() != words[i-1].lower():
            deduped_words.append(word)
    
    cleaned = ' '.join(deduped_words)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    phonetic_pattern = r'é|æ|ɑ|ɔ|ɕ|ç|ḏ|ḍ|ð|ə|ɚ|ɛ|ɝ|ḡ|ʰ|ḥ|ḫ|ḳ|ḵ|ḷ|ɬ|ɫ|ŋ|ṇ|ɲ|ɴ|ŏ|ɸ|θ|p̅|þ|ɹ|ɾ|ʀ|ʁ|ṛ|š|ś|ṣ|ʃ|ṭ|ṯ|ʨ|tʂ|ʊ|ŭ|ü|ʌ|ɣ|ʍ|χ|ʸ|ʎ|ẓ|ž|ʒ|'"'|'"'|ʔ|ʕ|∬|↫'
    cleaned = re.sub(phonetic_pattern, '', cleaned)
    
    if cleaned.strip() in ['.', '?', '!', ',', '']:
        cleaned = ''
    
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
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file, engine='python', on_bad_lines='skip')
        
        if utterance_column not in df.columns:
            print(f"Error: CSV file does not have a '{utterance_column}' column")
            print(f"Available columns: {', '.join(df.columns)}")
            return None
        
        print(f"Extracting clean transcriptions from '{utterance_column}' column...")
        df[clean_column] = df[utterance_column].apply(extract_clean_transcription)
        
        if not output_file:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_clean.csv"
        
        print(f"Saving to {output_file}")
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
        print(f"Processed {len(df)} utterances")
        
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

def process_examples(examples):
    """
    Process a list of example utterances and return their cleaned versions.
    Useful for testing the cleaning function.
    
    Parameters:
    examples (list): A list of utterance strings to clean
    
    Returns:
    list: A list of cleaned utterance strings
    """
    results = []
    
    for example in examples:
        cleaned = extract_clean_transcription(example)
        results.append((example, cleaned))
    
    for orig, clean in results:
        print(f"Original: {orig}")
        print(f"Cleaned : {clean}")
        print("-" * 50)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Clean AphasiaBank transcriptions in a CSV file')
    parser.add_argument('input_file', nargs='?', default=None,
                        help='Path to the input CSV file (required unless --test is used)')
    parser.add_argument('--output', '-o', default=None, 
                        help='Path to save the output CSV file (default: input_filename_with_clean.csv in the same directory)')
    parser.add_argument('--column', '-c', default='utterance', 
                        help='Name of the column containing utterances (default: utterance)')
    parser.add_argument('--clean-column', '-n', default='cleaned_utterance',
                        help='Name of the column to store clean transcriptions (default: cleaned_utterance)')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run with test examples instead of processing a file')
    
    args = parser.parse_args()
    
    if args.test:
        test_examples = [
            "(be)cause <I have just> [//] <I have> [/] I hafta talk.",
            "if I cant do it I say \"/.",
            "\" oh I cant get that.",
            "I went Wyomin(g).",
            "++ Wyoming.",
            "(be)cau(se) my sister is over here.",
            "and I go this way.",
            "and I woke and I woke up I was like I was like \"/.",
            "and I was like +//.",
            "+ what is like a +..?",
        ]
        process_examples(test_examples)
    else:
        if not args.input_file:
            print("Error: Input file is required when not using --test mode")
            parser.print_help()
            sys.exit(1)
            
        process_csv(args.input_file, args.output, args.column, args.clean_column)

if __name__ == "__main__":
    main()