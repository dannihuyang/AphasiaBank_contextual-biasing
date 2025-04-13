# 🗣️ AphasiaBank Contextual Biasing

A pipeline for processing, transcribing, and evaluating speech data from the AphasiaBank dataset (https://aphasia.talkbank.org/) using Whisper (https://github.com/openai/whisper)with contextual biasing (using refactored code from https://github.com/Cinnamon/whisper-jargon, Nguyen et al., 2024).

## 📂 Project Structure

AphasiaBank_contextual-biasing/
├── data/                      # Data directory (mounted in Docker)
├── output/                    # Output directory (mounted in Docker)
├── parse_error.py             # Extract utterances from transcripts
├── clean_utterances.py        # Clean CLAN notation from utterances
├── extract_audio_segments.py  # Extract audio segments based on timestamps
├── create_biasing_list.py     # Create word lists for contextual biasing
├── transcribe_segments.py     # Transcribe audio segments using Whisper
├── evaluate_wer.py            # Evaluate transcription accuracy
├── Dockerfile                 # Docker container configuration
└── run_in_cluster/            # Scripts for running on compute clusters

## 🔄 Workflow

### 1️⃣ Parse Transcription Files

Extract utterances and error information from CLAN transcription files:

```bash
python parse_error.py /path/to/transcripts output/utterances_with_errors.csv
```

### 2️⃣ Clean Utterances

Remove CLAN notation and format the transcripts:

```bash
python clean_utterances.py output/utterances_with_errors.csv output/cleaned_utterances.csv
```

### 3️⃣ Extract Audio Segments

Extract the audio segments based on timestamps in the CSV:

```bash
python extract_audio_segments.py output/utterances_with_errors.csv /path/to/audio_files --output-dir data/extracted_audio
```

### 4️⃣ Create Biasing Lists (Optional)

Generate word lists for contextual biasing:

```bash
python create_biasing_list.py output/utterances_with_errors.csv --filter-stopwords
```

### 5️⃣ Transcribe Audio Segments

Transcribe the extracted audio segments using Whisper:

```bash
python transcribe_segments.py data/extracted_audio output/utterances_with_errors.csv --model base
```

For biasing-enabled transcription:

```bash
python transcribe_segments.py data/extracted_audio output/utterances_with_errors.csv --model base --use-jargon --biasing-list output/biasing_list.txt --dict-coeff 3.0
```

### 6️⃣ Evaluate Transcription Results

Calculate Word Error Rate (WER) and Character Error Rate (CER):

```bash
python evaluate_wer.py output/utterances_with_errors.csv --ref cleaned_utterance --hyp whisper_transcription
```

7. Run in Compute Cluster
Sync data and run the pipeline in a compute cluster:
./run_in_cluster/sync_back_local.sh
./run_in_cluster/run_pipeline.sh

## 🐳 Docker Usage

Build the Docker image:

```bash
docker build -t aphasia_contextual .
```

Run the container:

```bash
docker run --gpus all -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  aphasia_contextual
```

## ☁️ Cluster Usage

Use the scripts in `run_in_cluster/` to run on HPC clusters:

```bash
sbatch run_in_cluster/run_transcribe_gpu.sh
```

Sync data to/from the cluster:

```bash
# From local to cluster
./run_in_cluster/sync_to_cloud.sh

# From cluster to local
./run_in_cluster/sync_back_local.sh
```

## 📋 Requirements

See `requirements.txt` for a complete list of dependencies.

