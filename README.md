# TRIBE v2 — Brain Response from Audio (Kaggle)

Predict and visualize how the brain reacts to an audio clip using Meta's
[TRIBE v2](https://github.com/facebookresearch/tribev2) multimodal brain-encoding model,
run entirely in a Kaggle notebook.

TRIBE v2 predicts fMRI brain responses to naturalistic stimuli (video, audio, text).
This project feeds it an audio file and produces the predicted cortical response
on the **fsaverage5** mesh (~20k vertices), plus a full set of audio and brain
visualizations.

## Requirements

- A Kaggle account with **GPU enabled** (Settings → Accelerator → GPU T4 x2 or P100)
- **Internet ON** (Settings → Internet)
- A **HuggingFace account + access token**
- Accepted license terms on:
  - https://huggingface.co/facebook/tribev2
  - the LLaMA 3.2 model page (used as TRIBE's text backbone)

> The model is gated. The token alone will not work unless you have clicked
> **Agree/Accept** on the gated model pages while logged in.

## Setup

### 1. HuggingFace token

1. huggingface.co → **Settings → Access Tokens → Create new token** (Read scope is enough).
2. Copy the `hf_...` string.
3. In Kaggle: **Add-ons → Secrets → Add a new secret**
   - **Label:** `HF_TOKEN`
   - **Value:** your `hf_...` token
4. Toggle the secret **ON** for the notebook.

### 2. Upload your audio

Right panel → **Add Input → Upload** your audio file (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`).
It will mount under `/kaggle/input/`.

## How to run

Run the cells in order. Important notes baked into the code:

- Kaggle's `/kaggle/input/` is **read-only**, and TRIBE writes a transcript `.tsv`
  next to the audio file. The setup **copies the audio into `/kaggle/working/`** first
  to avoid an `OSError: Read-only file system`.
- The first run downloads model weights (~700 MB) and transcribes the audio, so the
  transcription step takes a while.

### Pipeline overview

| Step | What it does |
|------|--------------|
| Install | Clones the repo, `pip install -e ".[plotting]"`, plus `librosa` and `imageio` |
| Auth | Logs into HuggingFace with `HF_TOKEN` |
| Prep | Copies audio to a writable folder |
| Inference | `TribeModel.from_pretrained` → `get_events_dataframe(audio_path=...)` → `predict()` |
| Outputs | Generates all files listed below |

### Core inference code

```python
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
df = model.get_events_dataframe(audio_path=AUDIO_PATH)   # AUDIO_PATH must be writable
preds, segments = model.predict(events=df)               # preds: (n_timesteps, n_vertices)
```

## Outputs

All files are written to `/kaggle/working/` and can be downloaded from the output panel.

| File | Description |
|------|-------------|
| `waveform.png` | Audio waveform over time |
| `spectrogram.png` | Log-frequency spectrogram (dB) |
| `rms.png` / `rms_raw.npy` | RMS energy envelope (plot + raw array) |
| `reaction_raw.npy` | Full prediction matrix `(n_timesteps, n_vertices)` |
| `metrics_raw.csv` | Per-timestep stats: mean, std, min, max, L2 norm |
| `brain_timestamps.npy` | Timestamp for each predicted timestep |
| `top_changing_vertices.csv` / `.png` | Vertices with highest temporal variance + time-courses |
| `brain_response.gif` | Animated cortical response over time |
| `summary_plots.png` | Audio energy vs. predicted brain reaction, stacked |

## Interpreting the output

- `preds` is a **time × vertex** matrix of predicted fMRI response on the fsaverage5
  cortical surface. Rows are timepoints, columns are surface vertices.
- Predictions are for the **"average" subject** (see the TRIBE v2 paper).
- **Hemodynamic lag:** predictions are offset **5 seconds into the past** relative to
  the stimulus. When aligning brain response against audio, the response to a given
  sound sits ~5s earlier on the brain axis than on the audio axis.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `OSError: Read-only file system` | Make sure the audio is copied to `/kaggle/working/` before inference |
| `403` / gated model error | Accept license terms on the HF model pages; verify `HF_TOKEN` secret is set and ON |
| HF rate-limit warning | Harmless if downloads succeed; set `HF_TOKEN` to silence it |
| CUDA out of memory (T4) | Use a shorter audio clip to confirm the pipeline, then scale up |
| Timestamps look wrong | Run `print(pd.DataFrame(segments).columns)` to check the real timing column |

## License & attribution

TRIBE v2 is released under **CC BY-NC 4.0** (non-commercial).
If you use it, cite the paper:

```
@article{dascoli2026foundation,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, St{\'e}phane and Rapin, J{\'e}r{\'e}my and Benchetrit, Yohann and Brooks, Teon and Begany, Katelyn and Raugel, Jos{\'e}phine and Banville, Hubert and King, Jean-R{\'e}mi},
  journal={arXiv preprint arXiv:2605.04326},
  year={2026}
}
```
