import sys
import os
import pandas as pd
import argparse
import glob
from tqdm import tqdm
import torch
import whisper  # Import the whole module

from whisper.tokenizer import Tokenizer

def transcribe_audio_segments(extracted_dir, csv_file, model_name="base", use_jargon=False, biasing_list_path=None, beam_size=10, dict_coeff=0.0, batch_size=10, output_column="whisper_transcription"):
    """
    Transcribe extracted audio segments using Whisper and add results to CSV
    """ 
    # Add device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the original CSV
    df = pd.read_csv(csv_file)
    
    # Add a column for Whisper transcriptions if it doesn't exist
    if output_column not in df.columns:
        df[output_column] = None
    
    # Load the Whisper model
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name, device=device)
    
    # Find all extracted audio files
    all_segments = glob.glob(os.path.join(extracted_dir, "**/*.wav"), recursive=True)
    print(f"Found {len(all_segments)} audio segments")
    
    # Count how many need processing
    to_process = 0
    for audio_file in all_segments:
        filename = os.path.basename(audio_file)
        parts = filename.split('_')
        participant_id = parts[0]
        timestamp = '_'.join(parts[1:3]).replace('.wav', '')
        
        row_idx = df[(df['filename'] == participant_id) & 
                    (df['timestamp'] == timestamp)].index
        
        if len(row_idx) > 0 and pd.isna(df.loc[row_idx[0], output_column]):
            to_process += 1
    
    print(f"Found {to_process} audio segments that need transcription")
    
    # Process each audio file
    processed = 0
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
        
        # Skip if already transcribed
        if len(row_idx) > 0 and not pd.isna(df.loc[row_idx[0], output_column]):
            continue
        
        # Load and process audio
        audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        
        # Set decoding options
        options = whisper.DecodingOptions(
            task="transcribe",
            language="en",
            beam_size=beam_size,
            dict_path=biasing_list_path if use_jargon else None,
            dict_coeff=dict_coeff if use_jargon else 0.0,
            transcription_file=audio_file
        )
        
        # Transcribe the audio
        result = whisper.decode(model, mel, options)
        transcription = result.text.strip()
        
        # Update the DataFrame
        df.loc[row_idx, output_column] = transcription
        
        # Update progress counter
        processed += 1
        
        # Save more frequently and show progress
        if processed % batch_size == 0:
            df.to_csv(csv_file, index=False)
            print(f"Progress: {processed}/{to_process} ({processed/to_process*100:.1f}%)")
    
    # Final save
    df.to_csv(csv_file, index=False)
    print(f"Transcription complete. Processed {processed} files.")

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio segments with Whisper")
    parser.add_argument("extracted_dir", help="Directory containing extracted audio segments")
    parser.add_argument("csv_file", help="Path to the original CSV file")
    parser.add_argument("--model", "-m", default="base", 
                        help="Whisper model to use (tiny, base, small, medium, large)")
    parser.add_argument("--use-jargon", action="store_true", 
                        help="Use jargon decoding with biasing list")
    parser.add_argument("--biasing-list", "-b", 
                        help="Path to the biasing list file")
    parser.add_argument("--beam-size", type=int, default=10, 
                        help="Beam size for decoding")
    parser.add_argument("--dict-coeff", type=float, default=0.0, 
                        help="Dictionary coefficient for biasing")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="How often to save progress to CSV")
    parser.add_argument("--output-column", "-o", default="whisper_transcription",
                        help="Column name for storing transcriptions")
    
    args = parser.parse_args()
    transcribe_audio_segments(
        args.extracted_dir, 
        args.csv_file, 
        args.model, 
        args.use_jargon,
        args.biasing_list, 
        args.beam_size, 
        args.dict_coeff, 
        args.batch_size,
        args.output_column
    )

if __name__ == "__main__":
    main()