import os
import sys
from whisper.tokenizer import get_tokenizer

def tokenize_biasing_phrases(file_path, multilingual=True, language="en"):
    """
    Tokenize each phrase in the biasing file and show token IDs and decoded tokens
    
    Parameters:
    -----------
    file_path : str
        Path to the biasing_phonetic.txt file
    multilingual : bool, default=True
        Whether to use the multilingual tokenizer
    language : str, default="en"
        Language code to use for the tokenizer
    """
    # Load the Whisper tokenizer
    tokenizer = get_tokenizer(multilingual=multilingual, language=language)
    
    # Read the biasing file
    with open(file_path, 'r') as f:
        phrases = [line.strip() for line in f if line.strip()]
    
    # Debugging: Print the number of phrases read
    print(f"Read {len(phrases)} phrases from the file.")
    
    # Print header
    print(f"{'Phrase':<20} {'Token IDs':<40} {'Decoded Tokens'}")
    print("-" * 80)
    
    # Initialize a list to store token IDs for all phrases
    all_token_ids = []
    
    # Tokenize each phrase
    for phrase in phrases:
        # Tokenize the phrase
        token_ids = tokenizer.encode(phrase)
        
        # Append token_ids to the list
        all_token_ids.append(token_ids)
        
        # Get decoded versions of each token for visualization
        decoded_tokens = []
        for id in token_ids:
            if id >= tokenizer.timestamp_begin:
                # Skip timestamp tokens
                continue
            decoded = tokenizer.decode_with_timestamps([id])
            decoded_tokens.append(decoded)
        
        # Print results
        print(f"{phrase:<20} {str(token_ids):<40} {' '.join(decoded_tokens)}")
    
    # Define output file path
    output_file_path = "tokenized_output.txt"
    
    # Write the results to the output file
    with open(output_file_path, 'w') as output_file:
        # Write header
        output_file.write(f"{'Phrase':<20} {'Token IDs':<40} {'Decoded Tokens'}\n")
        output_file.write("-" * 80 + "\n")
        
        # Write each phrase and its token information
        for phrase, token_ids in zip(phrases, all_token_ids):
            decoded_tokens = []
            for id in token_ids:
                if id >= tokenizer.timestamp_begin:
                    continue
                decoded = tokenizer.decode_with_timestamps([id])
                decoded_tokens.append(decoded)
            output_file.write(f"{phrase:<20} {str(token_ids):<40} {' '.join(decoded_tokens)}\n")
    
    print(f"Output written to {output_file_path}")
    
    return phrases, all_token_ids

if __name__ == "__main__":
    # Check if file path is provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "biasing_phonetic.txt"
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    # Tokenize the phrases
    tokenize_biasing_phrases(file_path)