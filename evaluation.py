# When building the biasing list
unique_phrases = set()  # Use a set to automatically deduplicate
for transcript in all_transcripts:
    unique_phrases.update(extract_target_phrases(transcript))

# Write unique phrases to biasing file
with open("biasing_list.txt", "w") as f:
    for phrase in sorted(unique_phrases):
        f.write(f"{phrase}\n")

# When evaluating - track by speaker-phrase pairs
results = {}  # Dictionary keyed by (speaker_id, phrase)
for speaker_id, phrase, was_recognized in evaluation_data:
    results[(speaker_id, phrase)] = was_recognized

# Calculate metrics
phrase_accuracy = {}  # Accuracy per phrase across speakers
for phrase in unique_phrases:
    instances = [(s, p) for (s, p) in results.keys() if p == phrase]
    correct = sum(1 for i in instances if results[i])
    phrase_accuracy[phrase] = correct / len(instances) if instances else 0