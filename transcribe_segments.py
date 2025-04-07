import sys
import os
import pandas as pd
import argparse
import glob
from tqdm import tqdm
import torch

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import from your local whisper folder
from whisper import load_model, load_audio, log_mel_spectrogram, DecodingOptions, decode, pad_or_trim
from whisper.tokenizer import Tokenizer

def transcribe_audio_segments(extracted_dir, csv_file, model_name="base", batch_size=10):
    """
    Transcribe extracted audio segments using Whisper and add results to CSV
    """
    # Load the original CSV
    df = pd.read_csv(csv_file)
    
    # Add a column for Whisper transcriptions if it doesn't exist
    if 'whisper_transcription' not in df.columns:
        df['whisper_transcription'] = None
    
    # Load the Whisper model
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    # Find all extracted audio files
    all_segments = glob.glob(os.path.join(extracted_dir, "**/*.wav"), recursive=True)
    print(f"Found {len(all_segments)} audio segments")
    
    # Process each audio file
    for audio_file in tqdm(all_segments, desc="Transcribing"):
        # Extract participant ID and timestamp from filename
        filename = os.path.basename(audio_file)
        parts = filename.split('_')
        participant_id = parts[0]
        timestamp = '_'.join(parts[1:3]).replace('.wav', '')
        
        # Find the corresponding row in the DataFrame
        row_idx = df[(df['filename'] == participant_id) & 
                     (df['timestamp'] == timestamp)].index
        
        if len(row_idx) == 0:
            print(f"Warning: No matching row found for {filename}")
            continue
        
        # Transcribe the audio
        result = model.transcribe(audio_file)
        transcription = result["text"].strip()
        
        # Update the DataFrame
        df.loc[row_idx, 'whisper_transcription'] = transcription
        
        # Save periodically (e.g., every batch_size files)
        if all_segments.index(audio_file) % batch_size == 0:
            df.to_csv(csv_file, index=False)
    
    # Final save
    df.to_csv(csv_file, index=False)
    print(f"Transcription complete. Results saved to {csv_file}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio segments with Whisper")
    parser.add_argument("extracted_dir", help="Directory containing extracted audio segments")
    parser.add_argument("csv_file", help="Path to the original CSV file")
    parser.add_argument("--model", "-m", default="base", 
                        help="Whisper model to use (tiny, base, small, medium, large)")
    parser.add_argument("--batch-size", "-b", type=int, default=10,
                        help="How often to save progress to CSV")
    
    args = parser.parse_args()
    transcribe_audio_segments(args.extracted_dir, args.csv_file, 
                             args.model, args.batch_size)

if __name__ == "__main__":
    main()