import Levenshtein
import torch.nn.functional as F
import re
import pandas as pd
import argparse
import numpy as np
import os
import sys
from tabulate import tabulate

def process_transcript(text):
    if not isinstance(text, str):
        return ""
    # Lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def word_edit_distance(ref, hyp):
    if not isinstance(ref, str) or not isinstance(hyp, str):
        return 0
    ref_words = ref.split()
    hyp_words = hyp.split()
    return Levenshtein.distance(ref_words, hyp_words)

def char_edit_distance(ref, hyp):
    if not isinstance(ref, str) or not isinstance(hyp, str):
        return 0
    return Levenshtein.distance(ref, hyp)

def check_target_recognized(transcription, target):
    if not isinstance(transcription, str) or not isinstance(target, str) or target.strip() == "":
        return None
    return target.lower() in transcription.lower()

def calculate_target_recognition(df, hyp_columns, target_column, mask=None):
    """
    Calculates target recognition rates for each hypothesis column.
    Optionally, only considers rows where mask is True.
    Returns a dict: {folder: {hyp_column: recognition_rate, ...}, ...}
    """
    results = {}
    if mask is not None:
        df = df[mask]
    for hyp_column in hyp_columns:
        col_name = f"{hyp_column}_target_recognized"
        df[col_name] = df.apply(
            lambda row: check_target_recognized(row[hyp_column], row[target_column]), axis=1
        )
        recognized = df[col_name].mean() * 100 if len(df) > 0 else 0
        results[hyp_column] = recognized
    return results

def analyze_phonological_errors(df, hyp_columns, target_column):
    """
    Analyze how well each hypothesis recognizes target phrases specifically in phonological errors.
    Returns: dict of recognition rates for each hypothesis column, by folder and overall.
    """
    results = {}
    # Filter for rows with phonological errors
    mask = (df['has_error'] == True) & (df['error_type'].str.startswith('p', na=False))
    phonological_errors = df[mask]
    if len(phonological_errors) == 0:
        return results

    # By folder
    for folder in phonological_errors['folder'].unique():
        folder_mask = phonological_errors['folder'] == folder
        folder_df = phonological_errors[folder_mask]
        results[folder] = calculate_target_recognition(folder_df, hyp_columns, target_column)

    # Overall
    results['overall'] = calculate_target_recognition(phonological_errors, hyp_columns, target_column)
    return results

def get_short_name(hyp_column):
    if hyp_column == "whisper_transcription":
        return "W.Trans"
    elif hyp_column == "whisper_turbo_0.5":
        return "W.Boost"
    else:
        return hyp_column[:7]

def process_hypothesis_column(df, hyp_column, ref_column, target_column=None):
    hyp_processed_col = f"{hyp_column}_processed"
    word_edit_col = f"{hyp_column}_word_edit_distance"
    char_edit_col = f"{hyp_column}_char_edit_distance"
    wer_col = f"{hyp_column}_wer"
    cer_col = f"{hyp_column}_cer"

    df[hyp_processed_col] = df[hyp_column].apply(process_transcript)
    valid_rows = (df['ref_processed'].notna() & (df['ref_processed'] != '')) & \
                 (df[hyp_processed_col].notna() & (df[hyp_processed_col] != ''))

    df[word_edit_col] = df.apply(
        lambda row: word_edit_distance(row['ref_processed'], row[hyp_processed_col]) 
        if valid_rows[row.name] else 0, 
        axis=1
    )
    df[char_edit_col] = df.apply(
        lambda row: char_edit_distance(row['ref_processed'], row[hyp_processed_col])
        if valid_rows[row.name] else 0,
        axis=1
    )
    df[wer_col] = df.apply(
        lambda row: (row[word_edit_col] / len(row['ref_processed'].split())) * 100 
        if valid_rows[row.name] and len(row['ref_processed'].split()) > 0 
        else None, 
        axis=1
    )
    df[cer_col] = df.apply(
        lambda row: (row[char_edit_col] / len(row['ref_processed'])) * 100
        if valid_rows[row.name] and len(row['ref_processed']) > 0
        else None,
        axis=1
    )
    if target_column:
        target_recognized_col = f"{hyp_column}_target_recognized"
        df[target_recognized_col] = df.apply(
            lambda row: check_target_recognized(row[hyp_column], row[target_column]) if valid_rows[row.name] else None, 
            axis=1
        )
    return df

def calculate_metrics(df, hyp_column):
    hyp_processed_col = f"{hyp_column}_processed"
    word_edit_col = f"{hyp_column}_word_edit_distance"
    char_edit_col = f"{hyp_column}_char_edit_distance"
    valid_rows = (df['ref_processed'].notna() & (df['ref_processed'] != '')) & \
                 (df[hyp_processed_col].notna() & (df[hyp_processed_col] != ''))
    valid_df = df[valid_rows]
    valid_samples = len(valid_df)
    total_words = valid_df['word_count'].sum()
    total_word_errors = valid_df[word_edit_col].sum()
    total_chars = valid_df['char_count'].sum()
    total_char_errors = valid_df[char_edit_col].sum()
    overall_wer = (total_word_errors / total_words) * 100 if total_words > 0 else 0
    overall_cer = (total_char_errors / total_chars) * 100 if total_chars > 0 else 0
    return overall_wer, overall_cer, valid_samples

def calculate_folder_metrics(df, hyp_columns):
    folder_results = {}
    for folder in df['folder'].unique():
        row_data = {'Folder': folder}
        folder_df = df[df['folder'] == folder]
        for hyp_column in hyp_columns:
            overall_wer, overall_cer, count = calculate_metrics(folder_df, hyp_column)
            short_name = get_short_name(hyp_column)
            row_data[f"{short_name} WER"] = f"{overall_wer:.2f}%" if count > 0 else "N/A"
            row_data[f"{short_name} CER"] = f"{overall_cer:.2f}%" if count > 0 else "N/A"
            row_data[f"{short_name} #"] = count
        folder_results[folder] = row_data
    return folder_results

def compare_csv_transcriptions(csv_file, ref_column='cleaned_utterance', hyp_columns=None, 
                              output_file=None,
                              target_column=None, verbose=False, folder_breakdown=False,
                              analyze_phonological=False, phonological_wer=False):
    """
    Compare transcriptions from multiple columns in a CSV file against a reference column.
    
    Parameters:
        csv_file (str): Path to the CSV file
        ref_column (str): Column name for reference transcriptions
        hyp_columns (list): List of column names for hypothesis transcriptions
        output_file (str): Optional path to save results
        target_column (str): Optional column name for target phrases to check recognition
        verbose (bool): Whether to print detailed results
        folder_breakdown (bool): Whether to break down metrics by folder
        analyze_phonological (bool): Whether to analyze phonological errors specifically
        phonological_wer (bool): Whether to calculate WER for phonological errors
        
    Returns:
        dict: Dictionary of metrics
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        return {}
    
    if analyze_phonological:
        required_columns = ['has_error', 'error_type']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            analyze_phonological = False
    
    # Set default hyp columns if none provided
    if hyp_columns is None:
        hyp_columns = ['whisper_transcription']
        if 'whisper_turbo_0.5' in df.columns:
            hyp_columns.append('whisper_turbo_0.5')
    
    # Filter out columns that don't exist in the dataframe
    hyp_columns = [col for col in hyp_columns if col in df.columns]
    
    if len(hyp_columns) == 0:
        return {}
    
    # Check if required columns exist
    if ref_column not in df.columns:
        return {}
        
    # Initialize results dictionaries
    results = {col: {} for col in hyp_columns}
    results['overall'] = {}
    
    # Add folder column if not present but filename is
    if 'folder' not in df.columns and 'filename' in df.columns:
        df['folder'] = df['filename'].str.extract(r'^([^_]+)')
    
    folder_results = {}
    if folder_breakdown and 'folder' in df.columns:
        for folder in df['folder'].unique():
            folder_results[folder] = {col: {} for col in hyp_columns}
    
    # Process reference column
    df['ref_processed'] = df[ref_column].apply(process_transcript)
    
    # Calculate word and character counts for reference
    df['word_count'] = df['ref_processed'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    df['char_count'] = df['ref_processed'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    
    # Process all hypothesis columns
    for hyp_column in hyp_columns:
        df = process_hypothesis_column(df, hyp_column, ref_column, target_column)
    
    # Calculate overall WER and CER for all columns
    all_results = []
    
    for hyp_column in hyp_columns:
        overall_wer, overall_cer, valid_samples = calculate_metrics(df, hyp_column)
        results[hyp_column]['wer'] = overall_wer
        results[hyp_column]['cer'] = overall_cer
        results[hyp_column]['valid_samples'] = valid_samples
        
        all_results.append({
            'Column': hyp_column,
            'WER': f"{overall_wer:.2f}%",
            'CER': f"{overall_cer:.2f}%",
            'Valid Samples': valid_samples
        })
    
    # Print overall results in a table format
    try:
        table = tabulate(all_results, headers="keys", tablefmt="grid")
    except Exception as e:
        for result in all_results:
            line = f"Column: {result['Column']} | WER: {result['WER']} | CER: {result['CER']} | Samples: {result['Valid Samples']}"
    
    # Calculate folder-level metrics for side-by-side comparison if requested
    if folder_breakdown and 'folder' in df.columns:
        folder_results = calculate_folder_metrics(df, hyp_columns)
        try:
            table = tabulate([v for v in folder_results.values()], headers="keys", tablefmt="grid")
        except Exception as e:
            for hyp_column in hyp_columns:
                line = f"{hyp_column} WER: {folder_results[folder][hyp_column].get(f'{hyp_column} WER', 'N/A')}"
                line += f" | {hyp_column} CER: {folder_results[folder][hyp_column].get(f'{hyp_column} CER', 'N/A')}"
    
    # Analyze phonological errors specifically if requested
    if analyze_phonological and target_column:
        phonological_results = analyze_phonological_errors(df, hyp_columns, target_column)
        results['phonological_analysis'] = phonological_results
    
    # Save results to file if requested
    if output_file:
        try:
            df.to_csv(output_file, index=False)
        except Exception as e:
            pass
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Calculate WER and CER for transcriptions in CSV files.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--ref', type=str, default='cleaned_utterance', 
                        help='Column name for reference transcriptions')
    parser.add_argument('--hyp', action='append', default=None,
                        help='Column name(s) for hypothesis transcriptions (can be used multiple times)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for detailed metrics')
    parser.add_argument('--target', type=str, default=None,
                        help='Column name for target phrases to check recognition')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--folders', action='store_true',
                        help='Break down metrics by folder')
    parser.add_argument('--analyze-phonological', action='store_true',
                        help='Analyze phonological errors specifically')
    parser.add_argument('--phonological-wer', action='store_true',
                        help='Calculate WER for phonological errors')
    
    try:
        args = parser.parse_args()
        
        # Check if file exists
        if not os.path.exists(args.csv_file):
            return
        
        metrics = compare_csv_transcriptions(
            args.csv_file,
            ref_column=args.ref,
            hyp_columns=args.hyp,
            output_file=args.output,
            target_column=args.target,
            verbose=args.verbose,
            folder_breakdown=args.folders,
            analyze_phonological=args.analyze_phonological,
            phonological_wer=args.phonological_wer
        )

    except Exception as e:
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()