import whisper
import Levenshtein
import torch
import torch.nn.functional as F
import re
import pandas as pd
import argparse

def do_batch_asr(audio_tensors, model_size='small', batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    model = whisper.load_model(model_size)
    for tensor in audio_tensors:
        tensor.to(device)
        result = model.transcribe(tensor)
        results.append(result['text'])
    return results

def compute_wer(ref_sentences, hyp_sentences):
    """
    Compute the Word Error Rate (WER) over a list of reference and hypothesis sentences.

    Parameters:
    - ref_sentences (list of str): List of reference sentences.
    - hyp_sentences (list of str): List of hypothesis sentences.

    Returns:
    - float: Average WER across all sentences.
    """
    total_words = 0
    total_errors = 0

    cer_list = []
    for ref_sent, hyp_sent in zip(ref_sentences, hyp_sentences):
        ref_words = ref_sent.split()
        hyp_words = hyp_sent.split()
        num_words = len(ref_words)
        total_words += num_words

        # Compute the word-level Levenshtein distance
        word_distance = Levenshtein.distance(' '.join(ref_words), ' '.join(hyp_words))
        total_errors += word_distance

        # Compute the character-level Levenshtein distance for CER
        char_distance = Levenshtein.distance(ref_sent, hyp_sent)
        cer_value = char_distance / len(ref_sent) if len(ref_sent) > 0 else 0
        cer_list.append(cer_value)

    wer = (total_errors / total_words) * 100 if total_words > 0 else 0
    average_cer = sum(cer_list) / len(cer_list) if cer_list else 0
    return wer, average_cer

def calculate_cer(reference, hypothesis):
    """
    Calculate the Character Error Rate (CER) between a reference and hypothesis string.

    Parameters:
        reference (str): The ground-truth string.
        hypothesis (str): The string to compare.

    Returns:
        float: CER as a fraction (e.g., 0.25 for 25% error rate).
    """
    # Avoid division by zero if the reference is empty
    if len(reference) == 0:
        return 0.0

    # Compute the Levenshtein distance between the two strings
    distance = Levenshtein.distance(reference, hypothesis)

    # Compute the CER as the edit distance divided by the number of characters in the reference
    cer = distance / len(reference)
    return cer

def process_transcript(text):
    if not isinstance(text, str):
        return ""
    # Convert the text to lower-case
    text = text.lower()
    # Remove punctuation using a regular expression
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def process_transcripts_list(transcripts):
    """
    Process a list of transcripts.

    Parameters:
        transcripts (list of str): List of transcript strings.

    Returns:
        list of str: List of processed transcript strings.
    """
    return [process_transcript(t) for t in transcripts]

def calculate_batched_wer(ref_sentences, synth_audio_tensors):
    hyp_sentences = do_batch_asr(synth_audio_tensors)
    hyp_sentences_norm = process_transcripts_list(hyp_sentences)
    wer, avg_cer = compute_wer(ref_sentences, hyp_sentences_norm)
    return wer, avg_cer, hyp_sentences_norm

def compare_csv_transcriptions(csv_file, ref_column='clean_transcription', hyp_column='whisper_transcription', output_file=None):
    """
    Compare transcriptions from two columns in a CSV file.
    
    Parameters:
        csv_file (str): Path to the CSV file
        ref_column (str): Column name for reference transcriptions
        hyp_column (str): Column name for hypothesis transcriptions
        output_file (str): Optional path to save results
        
    Returns:
        tuple: (WER, CER, DataFrame with results)
    """
    # Read the CSV file
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Check if columns exist
    if ref_column not in df.columns:
        raise ValueError(f"Reference column '{ref_column}' not found in CSV")
    if hyp_column not in df.columns:
        raise ValueError(f"Hypothesis column '{hyp_column}' not found in CSV")
    
    # Process transcriptions
    print("Processing transcriptions...")
    df['ref_processed'] = df[ref_column].apply(process_transcript)
    df['hyp_processed'] = df[hyp_column].apply(process_transcript)
    
    # Add word and character counts
    df['word_count'] = df['ref_processed'].apply(lambda x: len(x.split()))
    df['char_count'] = df['ref_processed'].apply(len)
    
    # Calculate CER for each row
    print("Calculating CER for each utterance...")
    df['cer'] = df.apply(lambda row: calculate_cer(row['ref_processed'], row['hyp_processed']) * 100, axis=1)
    
    # Calculate word-level and character-level edit distances, and WER for each row
    print("Calculating edit distances and WER for each utterance...")
    df['word_edit_distance'] = df.apply(lambda row: Levenshtein.distance(row['ref_processed'].split(), row['hyp_processed'].split()), axis=1)
    df['char_edit_distance'] = df.apply(lambda row: Levenshtein.distance(row['ref_processed'], row['hyp_processed']), axis=1)
    df['wer'] = df.apply(lambda row: (row['word_edit_distance'] / len(row['ref_processed'].split())) * 100 if len(row['ref_processed'].split()) > 0 else 0, axis=1)
    
    # Calculate overall WER and CER
    print("Calculating overall WER and CER...")
    total_words = df['word_count'].sum()
    total_word_errors = df['word_edit_distance'].sum()
    total_chars = df['char_count'].sum()
    total_char_errors = df['char_edit_distance'].sum()
    
    overall_wer = (total_word_errors / total_words) * 100 if total_words > 0 else 0
    overall_cer = (total_char_errors / total_chars) * 100 if total_chars > 0 else 0
    
    # Print results
    print(f"\nResults:")
    print(f"Total utterances: {len(df)}")
    print(f"Total words: {total_words}")
    print(f"Word Error Rate (WER): {overall_wer:.2f}%")
    print(f"Average Character Error Rate (CER): {overall_cer:.2f}%")
    
    # Save results if output file is provided
    if output_file:
        print(f"Saving results to {output_file}")
        df.to_csv(output_file, index=False)
    
    return overall_wer, overall_cer, df

def main():
    parser = argparse.ArgumentParser(description="Compare transcriptions in a CSV file")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("--ref", default="clean_transcription", help="Column name for reference transcriptions")
    parser.add_argument("--hyp", default="whisper_transcription", help="Column name for hypothesis transcriptions")
    parser.add_argument("--output", help="Path to save results CSV")
    
    args = parser.parse_args()
    
    compare_csv_transcriptions(args.csv_file, args.ref, args.hyp, args.output)

if __name__ == "__main__":
    main()