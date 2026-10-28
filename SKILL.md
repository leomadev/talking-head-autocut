---
name: talking-head-autocut
description: >-
  Automated 3-step video editing skill for solo talking-head recordings.
  Step 1 removes non-user sounds (Google Translate TTS, phone notifications, other speakers)
  using the user's permanent voice fingerprint (auto-enrolled on first run). Step 2 extracts
  pure subtitles and word timestamps via Apple Silicon GPU mlx-whisper + 20ms RMS acoustic trimming.
  Step 3 removes repeated takes (always keeping the last clean take), preserves natural breath
  pauses, applies pure linear volume gain, and exports the final 1080p MP4 via FFmpeg without
  requiring CapCut Pro or manual NLE tools.
---

# Talking-Head AutoCut (3-Step Pipeline)

Use this skill whenever the user wants to edit, clip, or clean up a raw talking-head video recording into a coherent, finished MP4 video.

## Core Principles & Constraints

1. **Solo Speaker Only (Pre-Whisper Voice Gate First)**:
   - The user records solo talking-head videos where external sounds occur:
     1. Google Translate pronunciation playback on the phone.
     2. Phone/app notification chimes or system sounds.
     3. Other people speaking nearby.
   - Always run **Step 1 (`step1_voice_gate.py`)** FIRST on the raw audio before running Whisper, so foreign voices never pollute subtitle generation or get merged into the user's sentences.
   - **Automatic Voice Enrollment**: On a user's very first run, if `resources/my_voice_profile.npy` does not exist yet, `step1_voice_gate.py` automatically builds and saves their permanent 256-dim `Resemblyzer` voice embedding from their 8 longest monologue islands ($\ge 3.5\text{s}$).
2. **Keep the Last Take (Deduplication)**:
   - The user often rehearses words or repeats sentences multiple times.
   - When multiple takes of the same phrase or sentence exist, **always select the LAST complete take**.
   - Drop isolated 1–2 word vocabulary pronunciation drills (e.g., `"rind, rind"`, `"chicken coop"`, `"crouching"`).
   - Split across long mid-sentence pauses ($\ge 0.8\text{s}$) so dead air inside a sentence is removed.
3. **Natural Human Pacing & Transparent Linear Volume Boost**:
   - Never over-clip into breathless robotic jump-cuts. Keep `0.08s` lead-in and `0.25s–0.35s` natural trailing breath at sentence boundaries, with `0.015s` audio micro-fades (`afade`) to prevent clicks/pops.
   - `step3_render_mp4.py` measures peak/mean levels via `volumedetect` and applies **pure constant linear gain (`volume=+XdB`)** (never `afftdn` spectral denoising or dynamic `loudnorm` AGC pumping, which cause phase distortion and ear fatigue).
4. **Zero Manual Editing Tools**:
   - Export the final 1080p MP4 directly via FFmpeg (`step3_render_mp4.py`).

---

## Directory Structure

- `resources/my_voice_profile.npy`: The user's permanent 256-dim d-vector voice embedding (`Resemblyzer`), auto-generated on first run (git-ignored for privacy).
- `scripts/step1_voice_gate.py`: Extracts PTS-aligned 16kHz audio (`first_pts=0`), runs WebRTC VAD + `0.8s` surgical sub-window voice verification against `my_voice_profile.npy`, and mutes all non-user audio.
- `scripts/step2_transcribe.py`: Runs Apple Silicon GPU `mlx-whisper` + Local Island Recovery + 20ms RMS word boundary trimmer & internal stutter splitter to produce `step2_subtitles.json`.
- `scripts/step3_render_mp4.py`: Renders the deduplicated Edit Decision List (`step3_final_edl.json`) into the final 1080p MP4 with `0.015s` audio fades and pure linear volume boost.

---

## Model Appointment per Step

| Step | Task | Appointed Model / Engine | Why |
| :--- | :--- | :--- | :--- |
| **Step 1** | Acoustic Voice Gate & Non-User Sound Removal | `Resemblyzer` on **Apple Silicon MPS GPU** (`torch.device("mps")`) + `WebRTC VAD` | Fast GPU acoustic speaker verification using `resources/my_voice_profile.npy`. |
| **Step 2** | Subtitle & Word Timestamp Extraction | `mlx-community/whisper-small-mlx` (default) or `mlx-community/whisper-large-v3-mlx` | **Apple Silicon Metal GPU** (`mlx-whisper`) + Local Island Recovery + 20ms RMS Trimmer. |
| **Step 3a** | Semantic Deduplication & Last-Take EDL Construction | Main Agent (`pro` / `inherit` LLM) | Requires deep narrative comprehension to spot false starts, mid-sentence restarts, and pick the last clean take. |
| **Step 3b** | Video Rendering & Post-Export QA Check | `FFmpeg` **Apple Silicon Media Engine** (`h264_videotoolbox` 4x parallel) + Subagent (`Model="flash"`) | Hardware-accelerated H.264 decode & encode on MBP GPU/Media Engine. |

---

## Step-by-Step Execution Workflow

### Step 1: Remove Non-User Voices (Acoustic Voice Gate)

Run `step1_voice_gate.py` using `uv run --with "resemblyzer"`:

```bash
SKILL_DIR="$HOME/.gemini/config/skills/talking-head-autocut"
uv run --with "resemblyzer" python3 "$SKILL_DIR/scripts/step1_voice_gate.py" "<INPUT_VIDEO>" "audio_work"
```

- *Optional*: To explicitly re-enroll / overwrite your voice profile from a sample video or audio file:
  ```bash
  uv run --with "resemblyzer" python3 "$SKILL_DIR/scripts/step1_voice_gate.py" --enroll "<SAMPLE_VIDEO_OR_WAV>"
  ```
- **Outputs**:
  - `audio_work/step1_clean_voice.wav` (non-user sounds surgically muted)
  - `audio_work/step1_gate_report.json` (kept vs. muted intervals)

### Step 2: Extract Pure Subtitles & Word Timestamps

Run `step2_transcribe.py` (uses `uvx --from mlx-whisper` on Metal GPU):

```bash
SKILL_DIR="$HOME/.gemini/config/skills/talking-head-autocut"
python3 "$SKILL_DIR/scripts/step2_transcribe.py" "audio_work"
```

- **Outputs**:
  - `audio_work/step2_subtitles.json` (list of speaker-verified, RMS-trimmed subtitle segments with word-level timestamps, pre-split at pauses $\ge 0.45\text{s}$ and internal word restarts).

### Step 3: Deduplicate Subtitles (Keep Last Take) & Render MP4

1. **Inspect `audio_work/step2_subtitles.json` & Follow Strict EDL Rules**:
   - Read all segments (`start`, `end`, `text`, `words`) from `audio_work/step2_subtitles.json`.
   - Identify the coherent narrative flow from start to finish using these **5 Mandatory EDL Rules**:
     1. **Discard Vocabulary Drills & Incomplete Fragments**: Drop isolated 1–2 word pronunciation drills and incomplete sentence fragments that pause $> 1.0\text{s}$ before finishing.
     2. **Always Pick the LAST Clean Complete Take**:
        - When a clause or sentence is repeated across multiple segments (even if separated by 10–60 seconds of rehearsing the next clause), **always scan forward to the LAST complete take**.
        - If the very last take has an internal word repeat/stutter at the end (e.g., repeating the last word twice inside the same breath), prefer the cleanest complete take that has zero internal repeats, or trim the trailing repeated word using the exact word `end` timestamp.
     3. **Never Include Mid-Segment Dead Air or Stretched Words ($> 0.85\text{s}$)**:
        - Inspect the `words` array of every selected segment: if there is a gap $\ge 0.65\text{s}$ between consecutive words or a single word with duration $> 1.0\text{s}$, either split the segment into two separate EDL entries around the pause or trim to the restart after the pause.
     4. **Never Pad `"in"` or `"out"` Across Word/Island Boundaries**:
        - Use the `start` and `end` already computed in `step2_subtitles.json` (which are acoustically trimmed via 20ms RMS envelopes and clamped against Step 1 muted intervals).
        - If trimming to a specific word inside a segment, set `"in"` to `word["start"] - 0.06` (never earlier than the previous word's `end + 0.05`) and `"out"` to `word["end"] + 0.25` (never later than the next word's `start - 0.05` or the Step 1 island's `end`).
     5. **Preserve Natural Breath Pauses at Sentence Ends**:
        - Keep `0.25s–0.35s` trailing room tone at the end of completed sentences (up to the Step 1 island `end`) so the speaker's last syllable never feels abruptly cut off.
   - Write the resulting clean list of `{"in": float, "out": float, "text": str}` objects to `audio_work/step3_final_edl.json`.

2. **Render the Final Video**:

```bash
SKILL_DIR="$HOME/.gemini/config/skills/talking-head-autocut"
python3 "$SKILL_DIR/scripts/step3_render_mp4.py" "audio_work/step3_final_edl.json" "<INPUT_VIDEO>" "<OUTPUT_MP4>" "audio_work"
```

3. **Quality Assurance Check (Optional Subagent / Whisper Verification)**:
   - Verify the exported MP4 duration with `ffprobe` and review the EDL transcript to confirm zero repeated sentences remain.
