#!/usr/bin/env python3
import os
import re
import sys
import csv
import argparse
import subprocess
import whisper
import numpy as np
import pandas as pd
import torch
import contextlib
import time
from pathlib import Path

def extract_timestamp_values(timestamp_str):
    """Extract start and end milliseconds from timestamp string"""
    # Check if it's already in the format "START_END"
    if '_' in timestamp_str:
        parts = timestamp_str.split('_')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return [(int(parts[0]), int(parts[1]))]
    
    # Try other common formats
    # Format like "29443-36663"
    dash_pattern = r'(\d+)-(\d+)'
    dash_matches = re.findall(dash_pattern, timestamp_str)
    if dash_matches:
        return [(int(start), int(end)) for start, end in dash_matches]
    
    # Format with any separator between numbers
    generic_pattern = r'(\d+)\s*[^\d\w]\s*(\d+)'
    generic_matches = re.findall(generic_pattern, timestamp_str)
    if generic_matches:
        return [(int(start), int(end)) for start, end in generic_matches]
    
    # If nothing matches, return empty list
    return []

def extract_error_id(text):
    """Extract error ID from the text if available"""
    # Look for error pattern like [* p:n], [* n:k], etc.
    error_pattern = r'\[\* ([a-z]:(?:[a-z]|=))\]'
    match = re.search(error_pattern, text)
    if match:
        return match.group(1).replace(':', '_')
    return "no_error_id"

def get_segment_keyword(text):
    """Extract a meaningful keyword from the segment text"""
    # Remove AphasiaBank annotations and get clean text
    clean_text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
    clean_text = re.sub(r'&[-+]?[a-zA-Z]+', '', clean_text)  # Remove &-um, &+n, etc.
    clean_text = re.sub(r'@u', '', clean_text)  # Remove @u markers
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Normalize whitespace
    
    # Extract potential keywords
    words = clean_text.split()
    if not words:
        return "untitled"
    
    # Priority to content words that might be important
    content_words = [w for w in words if len(w) > 3 and w.lower() not in 
                    {"this", "that", "then", "they", "there", "their", "would", "could", "should", "have"}]
    
    if content_words:
        return content_words[0].lower()
    else:
        return words[0].lower()

def convert_video_to_audio(video_file, output_dir):
    """Convert MP4 video file to WAV audio file with progress reporting"""
    base_name = os.path.basename(video_file)
    base_name_no_ext = os.path.splitext(base_name)[0]
    output_file = os.path.join(output_dir, f"{base_name_no_ext}.wav")
    
    # Check if output file already exists - skip conversion if it does
    if os.path.exists(output_file):
        print(f"Audio file {output_file} already exists, skipping conversion")
        return output_file
        
    command = [
        "ffmpeg", "-y", 
        "-i", video_file, 
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM 16-bit little-endian audio codec
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",  # Mono audio
        "-progress", "pipe:1",  # Output progress information
        output_file
    ]
    
    try:
        print(f"Converting video to audio: {video_file} -> {output_file}")
        print("This may take a while for large files...")
        
        # Run with progress feedback
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Simple progress indicator
        print("Conversion in progress: ", end="", flush=True)
        while process.poll() is None:
            print(".", end="", flush=True)
            time.sleep(3)  # Update progress every 3 seconds
            
        print("\nConversion completed")
        
        # Check exit code
        if process.returncode != 0:
            print(f"Error during conversion: {process.stderr.read()}")
            return None
            
        return output_file
    except subprocess.SubprocessError as e:
        print(f"Error converting video to audio: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_segment(audio_file, start_ms, end_ms, output_file):
    """Extract segment from audio file using FFmpeg"""
    start_time = float(start_ms) / 1000
    end_time = float(end_ms) / 1000
    
    start_str = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{start_time % 60:06.3f}"
    end_str = f"{int(end_time // 3600):02d}:{int((end_time % 3600) // 60):02d}:{end_time % 60:06.3f}"
    
    command = [
        "ffmpeg", "-y", "-i", audio_file, 
        "-ss", start_str, "-to", end_str, 
        "-c", "copy", output_file
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting segment: {e}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
        
        # Try again with a different method if the first one fails
        try:
            command2 = [
                "ffmpeg", "-y", "-i", audio_file, 
                "-ss", start_str, "-to", end_str, 
                "-acodec", "pcm_s16le",  # Decode and encode instead of stream copy
                output_file
            ]
            subprocess.run(command2, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e2:
            print(f"Error on second attempt: {e2}")
            print(f"FFmpeg stderr: {e2.stderr.decode()}")
            return False

def transcribe_with_whisper_debug(audio_file, model_name="turbo", log_file=None):
    """Transcribe audio using Whisper and capture debug output to a log file"""
    try:
        # Import necessary modules for low-level access
        from whisper import load_model, load_audio, log_mel_spectrogram, DecodingOptions, decode, pad_or_trim
        from dataclasses import replace
        
        # Create logs directory for detailed beam search logs
        detailed_logs_dir = "detailed_logs"
        os.makedirs(detailed_logs_dir, exist_ok=True)
        
        # Load the model
        model = load_model(model_name)
        
        # Load and preprocess audio
        audio = load_audio(audio_file)
        audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        
        # Set decoding options with beam size
        options = DecodingOptions(beam_size=10)
        
        # Create log file names - one for summary, one for detailed beam search
        if log_file is None:
            log_file = f"{os.path.splitext(audio_file)[0]}_decoding_log.txt"
        
        # Create a detailed log file with the same base name but in detailed_logs directory
        base_name = os.path.basename(os.path.splitext(audio_file)[0])
        detailed_log_file = os.path.join(detailed_logs_dir, f"{base_name}_detailed_decoding.txt")
        
        # Call decode with the detailed log filename
        result = decode(model, mel, options, log_filename=detailed_log_file)
        
        # Create a separate summary log file with just the results
        with open(log_file, "w") as f:
            f.write(f"--------------------------------decoding started for {audio_file}\n\n")
            f.write(f"Result: | avg_logprob: {result.avg_logprob:.2f} | no_speech_prob: {result.no_speech_prob:.2f}\n")
            f.write(f"Text: {result.text}\n")
            f.write(f"Length of text: {len(result.text)}\n")
            f.write(f"--------------------------------decoding finished\n\n")
        
        return result.text.strip()
    except Exception as e:
        print(f"Error transcribing {audio_file}: {e}")
        import traceback
        traceback.print_exc()
        return ""

def process_excel_file(excel_file, audio_base_path, output_dir, model_name, sheet_name=0):
    """Process an Excel file containing error phrases with custom column names
    
    Parameters:
    -----------
    excel_file : str
        Path to the Excel file
    audio_base_path : str
        Path to the audio or video file
    output_dir : str
        Directory to save the output files
    model_name : str
        Whisper model name to use
    sheet_name : int or str, default=0
        Sheet to read: 0 for first sheet, 1 for second sheet, or specify sheet name
    """
    try:
        # Load the Excel file with specified sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        results = []
        
        # Check if audio_base_path is a video file and convert it to audio if needed
        audio_base_path_ext = os.path.splitext(audio_base_path)[1].lower()
        if audio_base_path_ext in ['.mp4', '.mov', '.avi', '.mkv']:
            # It's a video file, convert to audio
            audio_file_path = convert_video_to_audio(audio_base_path, output_dir)
            if audio_file_path is None:
                print("Failed to convert video to audio. Exiting.")
                return []
            # Use the converted audio file
            base_audio_name = os.path.basename(audio_file_path).split('.')[0]
            audio_file = audio_file_path
        else:
            # It's already an audio file
            base_audio_name = os.path.splitext(os.path.basename(audio_base_path))[0]
            audio_file = audio_base_path
        
        # Print column names for debugging
        print(f"Excel columns: {', '.join(df.columns)}")
        
        # Process each row in the Excel file
        for index, row in df.iterrows():
            # Get data from columns using your specific column names
            segment_id = str(row.get('id', index))
            file_name = row.get('file', '')
            timestamp_str = str(row.get('timestamp', ''))
            error_type = row.get('error_type', 'unknown')
            pronunciation = row.get('pronounciation', '')
            target = row.get('target', '')
            
            print(f"Processing row {index+1}: ID={segment_id}, Timestamp={timestamp_str}")
            
            # Extract timestamp values
            timestamps = extract_timestamp_values(timestamp_str)
            
            if not timestamps:
                print(f"Warning: No valid timestamp found in '{timestamp_str}', skipping row {index+1}")
                continue
            
            for ts_index, (start_ms, end_ms) in enumerate(timestamps):
                # Create a unique segment name
                unique_id = f"{segment_id}_{ts_index}" if ts_index > 0 else segment_id
                segment_name = f"{base_audio_name}_{error_type}_{start_ms}_{end_ms}_{unique_id}"
                segment_file = os.path.join(output_dir, f"{segment_name}.wav")
                
                # Log file named after the segment ID
                log_file = os.path.join(output_dir, f"{segment_name}_decoding_log.txt")
                
                print(f"Extracting segment {start_ms}-{end_ms} from {audio_file} to {segment_file}")
                success = extract_segment(audio_file, start_ms, end_ms, segment_file)
                
                if success:
                    print(f"Transcribing {segment_file}...")
                    # Use the debug version that creates a log file
                    transcription = transcribe_with_whisper_debug(segment_file, model_name, log_file)
                    
                    results.append({
                        'id': unique_id,
                        'original_file': audio_file,
                        'segment_file': segment_file,
                        'start_ms': start_ms,
                        'end_ms': end_ms,
                        'error_type': error_type,
                        'pronunciation': pronunciation,
                        'target': target,
                        'transcription': transcription,
                        'log_file': log_file
                    })
        
        # Write results to CSV
        if results:
            csv_file = os.path.join(output_dir, f"{os.path.basename(excel_file).split('.')[0]}_segments.csv")
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            
            print(f"Results saved to {csv_file}")
        else:
            print("No segments were processed. Check if timestamps are in the correct format.")
        
        return results
    except Exception as e:
        print(f"Error processing Excel file: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    parser = argparse.ArgumentParser(description="Extract and transcribe segments from Excel file")
    parser.add_argument("input_file", help="Excel file with timestamps")
    parser.add_argument("audio_path", help="Path to audio file or video file")
    parser.add_argument("--output", "-o", default="segments", help="Output directory for segments")
    parser.add_argument("--model", "-m", default="turbo", help="Whisper model name to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--sheet", "-s", default=0, type=str, 
                        help="Sheet to use: 0 for first sheet, 1 for second sheet, or sheet name")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Determine file type and process accordingly
    ext = os.path.splitext(args.input_file)[1].lower()
    
    if ext in ['.xlsx', '.xls']:
        # Convert sheet to int if it's a number
        sheet = args.sheet
        try:
            sheet = int(sheet)
        except ValueError:
            # If conversion fails, it's a sheet name, keep as string
            pass
            
        process_excel_file(args.input_file, args.audio_path, args.output, args.model, sheet_name=sheet)
    else:
        print(f"Unsupported file format: {ext}. Please provide an Excel file.")
        sys.exit(1)

if __name__ == "__main__":
    main()