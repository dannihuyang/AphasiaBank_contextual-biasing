import os
import argparse
import pandas as pd
import subprocess
import glob
from tqdm import tqdm

def find_audio_file(participant_id, audio_dir):
    """
    Find the audio file for a given participant ID in the audio directory.
    
    Parameters:
    participant_id (str): The participant ID (e.g., 'fridriksson12a')
    audio_dir (str): Directory containing audio files
    
    Returns:
    str: Path to the audio file, or None if not found
    """
    # Try common audio extensions
    extensions = ['.wav', '.mp3', '.WAV', '.MP3', '.mp4', '.MP4']
    
    for ext in extensions:
        audio_path = os.path.join(audio_dir, f"{participant_id}{ext}")
        if os.path.exists(audio_path):
            return audio_path
    
    # If exact match not found, print available files for debugging
    print(f"Warning: Could not find exact audio file for {participant_id}")
    print(f"Looking in directory: {audio_dir}")
    print(f"Available files: {os.listdir(audio_dir)[:10]}")
    
    return None

def extract_audio_segment(audio_file, start_time, end_time, output_file):
    """
    Extract a segment from an audio file using ffmpeg.
    
    Parameters:
    audio_file (str): Path to the input audio file
    start_time (float): Start time in seconds
    end_time (float): End time in seconds
    output_file (str): Path to save the extracted segment
    
    Returns:
    bool: True if extraction was successful, False otherwise
    """
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Calculate duration
        duration = end_time - start_time
        
        # Run ffmpeg command
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-i', audio_file,  # Input file
            '-ss', str(start_time),  # Start time
            '-t', str(duration),  # Duration
            '-acodec', 'pcm_s16le',  # Output codec (standard WAV)
            '-ar', '16000',  # Sample rate (16kHz)
            '-ac', '1',  # Mono audio
            output_file  # Output file
        ]
        
        # Run the command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if the command was successful
        if result.returncode != 0:
            print(f"Error extracting segment: {result.stderr.decode()}")
            return False
        
        return True
    
    except Exception as e:
        print(f"Error extracting audio segment: {str(e)}")
        return False

def process_csv(csv_file, audio_dir, output_dir, timestamp_col='timestamp', filename_col='filename'):
    """
    Process a CSV file with timestamps and extract audio segments.
    
    Parameters:
    csv_file (str): Path to the CSV file
    audio_dir (str): Directory containing audio files
    output_dir (str): Directory to save extracted segments
    timestamp_col (str): Name of the column containing timestamps
    filename_col (str): Name of the column containing filenames
    
    Returns:
    int: Number of successfully extracted segments
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        if timestamp_col not in df.columns:
            print(f"Error: CSV file does not have a '{timestamp_col}' column")
            return 0
        
        if filename_col not in df.columns:
            print(f"Error: CSV file does not have a '{filename_col}' column")
            return 0
        
        # Create a dictionary to cache audio file paths
        audio_file_cache = {}
        
        # Count successful extractions
        success_count = 0
        
        # Process each row
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting audio segments"):
            # Get the timestamp and filename
            timestamp = row[timestamp_col]
            participant_id = row[filename_col]
            
            # Skip if timestamp is missing
            if pd.isna(timestamp) or not timestamp:
                continue
            
            # Parse the timestamp (format: start_end)
            try:
                start_ms, end_ms = map(int, timestamp.split('_'))
                start_sec = start_ms / 1000.0
                end_sec = end_ms / 1000.0
            except ValueError:
                print(f"Warning: Invalid timestamp format for row {i}: {timestamp}")
                continue
            
            # Find the audio file if not already cached
            if participant_id not in audio_file_cache:
                audio_file = find_audio_file(participant_id, audio_dir)
                if not audio_file:
                    print(f"Warning: Could not find audio file for {participant_id}")
                    continue
                audio_file_cache[participant_id] = audio_file
            else:
                audio_file = audio_file_cache[participant_id]
            
            # Create output filename
            output_subdir = os.path.join(output_dir, participant_id)
            output_file = os.path.join(output_subdir, f"{participant_id}_{timestamp}.wav")
            
            # Extract the segment
            if extract_audio_segment(audio_file, start_sec, end_sec, output_file):
                success_count += 1
        
        return success_count
    
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        return 0

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract audio segments based on timestamps in a CSV file')
    parser.add_argument('csv_file', help='Path to the CSV file containing timestamps')
    parser.add_argument('audio_dir', help='Directory containing audio files')
    parser.add_argument('--output-dir', '-o', default='extracted_audio',
                        help='Directory to save extracted audio segments (default: ./extracted_audio)')
    parser.add_argument('--timestamp-col', '-t', default='timestamp',
                        help='Name of the column containing timestamps (default: timestamp)')
    parser.add_argument('--filename-col', '-f', default='filename',
                        help='Name of the column containing filenames (default: filename)')
    
    args = parser.parse_args()
    
    # Process the CSV file
    success_count = process_csv(
        args.csv_file,
        args.audio_dir,
        args.output_dir,
        args.timestamp_col,
        args.filename_col
    )
    
    print(f"\nExtraction complete. Successfully extracted {success_count} audio segments.")
    print(f"Segments saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()