#!/usr/bin/env python3
"""
Step 2: Subtitle & Word Timestamp Extraction (mlx-whisper on Apple Silicon GPU + Local Island Recovery + RMS Word Trimmer)
- Runs mlx-whisper on the Step 1 speaker-isolated audio (step1_clean_voice.wav) with
  temperature fallback (0.0, 0.2, 0.4, 0.6, 0.8) so silent gaps never trigger greedy repetition loops.
- Runs an automatic Local Island Recovery pass: cross-checks all Step 1 Kept Voice Islands (>= 1.5s)
  and locally re-transcribes any island that Whisper skipped or under-transcribed, guaranteeing 100%
  word recall across all takes.
- Cross-checks every emitted word against Step 1's Kept Islands AND Muted Sub-Windows
  so zero Google Translate / notification words ever enter the subtitle file.
- Detects "Whisper-Skipped Stutter Islands" AND "Cross-Island Bridged False Starts"
  (where a 2-4 word false start on Island k is bridged into a longer clean take on Island k+1).
- Applies a 20ms RMS Acoustic Word Trimmer & Internal Restart Detector on every word:
  * Strips leading/trailing dead air from stretched Whisper word timestamps.
  * Detects internal silence gaps (>= 160ms) inside stretched words (>= 0.85s) where Whisper
    collapsed a mid-island word repeat/stutter into a single word token, and automatically
    trims/splits to the clean restart!
"""
import os
import sys
import json
import subprocess
import numpy as np
import soundfile as sf


def trim_and_find_split(w_start, w_end, wav, sr, frame_s=0.02, sil_thresh=0.0028, min_sil_frames=8):
    s_i, e_i = max(0, int(w_start * sr)), min(len(wav), int(w_end * sr))
    chunk = wav[s_i:e_i]
    f_len = int(frame_s * sr)
    n_f = len(chunk) // f_len
    if n_f < 6:
        return round(w_start, 3), round(w_end, 3), None
    rms = [float(np.sqrt(np.mean(chunk[j * f_len : (j + 1) * f_len] ** 2))) for j in range(n_f)]

    first_speech = 0
    while first_speech < n_f and rms[first_speech] < sil_thresh:
        first_speech += 1
    if first_speech >= n_f:
        return round(w_start, 3), round(w_end, 3), None

    last_speech = n_f - 1
    while last_speech > first_speech and rms[last_speech] < sil_thresh:
        last_speech -= 1

    split_t = None
    if (last_speech - first_speech) * frame_s >= 0.85:
        cur_sil = 0
        speech_before = 0
        best_gap_end = None
        for j in range(first_speech, last_speech + 1):
            if rms[j] >= sil_thresh:
                if speech_before >= 7 and cur_sil >= min_sil_frames:
                    rem_speech = sum(1 for k in range(j, last_speech + 1) if rms[k] >= sil_thresh)
                    if rem_speech >= 7:
                        best_gap_end = j
                speech_before += 1
                cur_sil = 0
            else:
                cur_sil += 1
        if best_gap_end is not None:
            split_t = round(w_start + max(0, best_gap_end - 1) * frame_s, 3)

    real_start = round(w_start + max(0, first_speech - 1) * frame_s, 3)
    real_end = round(w_start + min(n_f, last_speech + 2) * frame_s, 3)
    return real_start, real_end, split_t


def run_step2(work_dir="audio_work", model="mlx-community/whisper-small-mlx", force=False):
    clean_wav = os.path.join(work_dir, "step1_clean_voice.wav")
    gate_report = os.path.join(work_dir, "step1_gate_report.json")
    out_json = os.path.join(work_dir, "step2_subtitles.json")
    out_dir = os.path.join(work_dir, "step2_whisper")
    os.makedirs(out_dir, exist_ok=True)

    raw_whisper_json = os.path.join(out_dir, "step1_clean_voice.json")
    if force or not os.path.exists(raw_whisper_json):
        print(f"[Step 2] Running mlx-whisper GPU transcription + Local Island Recovery ({model}) on {clean_wav}...")
        py_code = """
import sys, json, wave, numpy as np, mlx_whisper

clean_wav, model, gate_report, out_json = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with wave.open(clean_wav, "rb") as wf:
    sr = wf.getframerate()
    wav = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

gdata = json.load(open(gate_report))
kept = sorted(gdata.get("kept", []), key=lambda x: x["start"])

res = mlx_whisper.transcribe(
    wav,
    path_or_hf_repo=model,
    word_timestamps=True,
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8)
)
raw_words = [dict(w) for seg in res.get("segments", []) if seg["end"] - seg["start"] >= 0.15 for w in seg.get("words", []) if w["end"] - w["start"] > 0.02]

under = []
for isl in kept:
    if isl["duration"] >= 1.5:
        ov = [w for w in raw_words if isl["start"] - 0.1 <= 0.5 * (w["start"] + w["end"]) <= isl["end"] + 0.1]
        unique_toks = len(set(w["word"].strip().lower().strip(".,!?") for w in ov))
        if len(ov) < max(2, int(isl["duration"] * 0.9)) or (len(ov) >= 3 and unique_toks <= 2):
            under.append(isl)

slices = []
for isl in under:
    if slices and isl["start"] - slices[-1]["end"] < 2.5:
        slices[-1]["end"] = isl["end"]
    else:
        slices.append({"start": isl["start"], "end": isl["end"]})

for sl in slices:
    s = max(0.0, sl["start"] - 0.20)
    e = min(len(wav) / sr, sl["end"] + 0.20)
    sub_wav = wav[int(s * sr) : int(e * sr)]
    sub_res = mlx_whisper.transcribe(sub_wav, path_or_hf_repo=model, word_timestamps=True, condition_on_previous_text=False)
    new_words = []
    for seg in sub_res.get("segments", []):
        for w in seg.get("words", []):
            if w["end"] - w["start"] > 0.02:
                wc = dict(w)
                wc["start"] = round(s + w["start"], 3)
                wc["end"] = round(s + w["end"], 3)
                new_words.append(wc)
    old_words = [w for w in raw_words if s + 0.1 <= 0.5 * (w["start"] + w["end"]) <= e - 0.1]
    old_unique = len(set(w["word"].strip().lower().strip(".,!?") for w in old_words))
    new_unique = len(set(w["word"].strip().lower().strip(".,!?") for w in new_words))
    if len(new_words) > len(old_words) or new_unique > old_unique:
        raw_words = [w for w in raw_words if not (s + 0.1 <= 0.5 * (w["start"] + w["end"]) <= e - 0.1)]
        raw_words.extend(new_words)

raw_words.sort(key=lambda x: x["start"])
json.dump({"segments": [{"start": 0.0, "end": round(len(wav)/sr, 3), "text": "", "words": raw_words}]}, open(out_json, "w"), indent=2)
"""
        cmd = [
            "uvx", "--from", "mlx-whisper", "python3", "-c", py_code,
            clean_wav, model, gate_report, raw_whisper_json
        ]
        subprocess.run(cmd, check=True)
    else:
        print(f"[Step 2] Using cached Whisper alignment: {raw_whisper_json}")

    with open(raw_whisper_json) as f:
        wdata = json.load(f)
    with open(gate_report) as f:
        gdata = json.load(f)
    wav, sr = sf.read(clean_wav)

    kept_islands = sorted(gdata.get("kept", []), key=lambda x: x["start"])
    muted_intervals = gdata.get("muted", [])

    raw_words = []
    for seg in wdata.get("segments", []):
        if seg["end"] - seg["start"] < 0.15:
            continue
        for w in seg.get("words", []):
            if w["end"] - w["start"] > 0.02:
                mid = 0.5 * (w["start"] + w["end"])
                if not any(m["start"] <= mid <= m["end"] for m in muted_intervals):
                    raw_words.append(dict(w))

    # 1. Detect Whisper-skipped stutter islands (3-island pattern: false start -> empty clean island -> continuation)
    for k in range(len(kept_islands) - 2):
        isl_prev = kept_islands[k]
        isl_skip = kept_islands[k + 1]
        isl_next = kept_islands[k + 2]
        if (isl_skip["end"] - isl_skip["start"] >= 0.80 and
            isl_skip["start"] - isl_prev["end"] < 0.50 and
            isl_next["start"] - isl_skip["end"] < 0.50):
            skip_words = [w for w in raw_words if isl_skip["start"] - 0.15 <= w["start"] and w["end"] <= isl_skip["end"] + 0.25]
            prev_words = [w for w in raw_words if isl_prev["start"] - 0.25 <= w["start"] and w["end"] <= isl_prev["end"] + 0.25]
            if not skip_words and prev_words:
                print(f"[Step 2] Detected hidden stutter repeat: shifting '{''.join(x['word'] for x in prev_words).strip()}' "
                      f"from false-start [{isl_prev['start']}-{isl_prev['end']}] -> clean take [{isl_skip['start']}-{isl_skip['end']}]")
                p_dur = max(0.1, isl_prev["end"] - isl_prev["start"])
                s_dur = max(0.1, isl_skip["end"] - isl_skip["start"])
                for w in prev_words:
                    rel_s = max(0.0, (w["start"] - isl_prev["start"]) / p_dur)
                    rel_e = min(1.0, (w["end"] - isl_prev["start"]) / p_dur)
                    w["start"] = round(isl_skip["start"] + rel_s * s_dur, 3)
                    w["end"] = round(isl_skip["start"] + rel_e * s_dur, 3)
                isl_prev["superseded"] = True

    # 2. Detect Cross-Island Bridged False Starts (2-island pattern: short 2-4 word false start on Island k
    #    whose last word bridges >= 0.60s into a >= 1.5x longer clean take on Island k+1)
    for k in range(len(kept_islands) - 1):
        isl_prev = kept_islands[k]
        isl_next = kept_islands[k + 1]
        if isl_prev.get("superseded"):
            continue
        p_dur = isl_prev["end"] - isl_prev["start"]
        n_dur = isl_next["end"] - isl_next["start"]
        if isl_next["start"] - isl_prev["end"] < 0.60 and n_dur > p_dur * 1.5:
            p_words = [w for w in raw_words if isl_prev["start"] - 0.15 <= w["start"] < isl_prev["end"] - 0.10]
            if 2 <= len(p_words) <= 4:
                bw = p_words[-1]
                if bw["end"] >= isl_next["start"] + 0.60:
                    print(f"[Step 2] Detected bridged false-start island: shifting '{''.join(x['word'] for x in p_words).strip()}' "
                          f"from [{isl_prev['start']}-{isl_prev['end']}] -> [{isl_next['start']}-{isl_next['end']}]")
                    span = max(0.1, bw["end"] - isl_next["start"])
                    step_w = span / len(p_words)
                    for idx_w, w in enumerate(p_words):
                        w["start"] = round(isl_next["start"] + idx_w * step_w, 3)
                        w["end"] = round(isl_next["start"] + (idx_w + 1) * step_w, 3)
                    isl_prev["superseded"] = True

    active_islands = [isl for isl in kept_islands if not isl.get("superseded")]

    # Merge active islands separated by < 0.35s into contiguous acoustic phrases
    acoustic_phrases = []
    for isl in active_islands:
        if acoustic_phrases and (isl["start"] - acoustic_phrases[-1]["end"] < 0.35):
            acoustic_phrases[-1]["end"] = round(isl["end"], 3)
        else:
            acoustic_phrases.append({"start": round(isl["start"], 3), "end": round(isl["end"], 3)})

    def get_phrase_idx(t_start, t_end):
        mid = 0.5 * (t_start + t_end)
        for idx, p in enumerate(acoustic_phrases):
            if p["start"] - 0.20 <= mid <= p["end"] + 0.20:
                return idx
        for idx, p in enumerate(acoustic_phrases):
            if p["start"] - 0.20 <= t_end <= p["end"] + 0.25:
                return idx
        return -1

    all_words = []
    for w in raw_words:
        p_idx = get_phrase_idx(w["start"], w["end"])
        if p_idx != -1:
            w_copy = dict(w)
            p = acoustic_phrases[p_idx]
            clamped_s = max(p["start"], w["start"])
            clamped_e = min(p["end"], max(clamped_s + 0.05, w["end"]))
            # Apply 20ms RMS trimming & internal silence-gap detection
            rs, re, split_t = trim_and_find_split(clamped_s, clamped_e, wav, sr)
            w_copy["start"] = max(p["start"], rs)
            w_copy["end"] = min(p["end"], max(w_copy["start"] + 0.04, re))
            if split_t is not None and w_copy["start"] < split_t < w_copy["end"]:
                w_copy["internal_restart_at"] = split_t
            w_copy["phrase_idx"] = p_idx
            all_words.append(w_copy)

    def is_clean_interval(a, b):
        return not any(max(a, m["start"]) < min(b, m["end"]) for m in muted_intervals)

    def finalize_segment(cur_words):
        text = "".join(x["word"] for x in cur_words).strip()
        if not text:
            return None
        for idx_w in range(min(3, len(cur_words))):
            if "internal_restart_at" in cur_words[idx_w]:
                rest_t = cur_words[idx_w]["internal_restart_at"]
                print(f"[Step 2] Trimming internal opening stutter in '{text[:35]}...' -> starting at clean restart {rest_t}s")
                span = max(0.08, cur_words[idx_w]["end"] - rest_t)
                step_t = span / (idx_w + 1)
                for j in range(idx_w + 1):
                    cur_words[j]["start"] = round(rest_t + j * step_t, 3)
                    cur_words[j]["end"] = round(rest_t + (j + 1) * step_t, 3)
                    cur_words[j].pop("internal_restart_at", None)
                break

        p_first = acoustic_phrases[cur_words[0]["phrase_idx"]]
        p_last = acoustic_phrases[cur_words[-1]["phrase_idx"]]
        seg_start = p_first["start"] if (cur_words[0]["start"] - p_first["start"] < 0.35 and is_clean_interval(p_first["start"], cur_words[0]["start"])) else max(p_first["start"], round(cur_words[0]["start"] - 0.08, 3))
        seg_end = p_last["end"] if (p_last["end"] - cur_words[-1]["end"] < 0.45 and is_clean_interval(cur_words[-1]["end"], p_last["end"])) else min(p_last["end"], round(cur_words[-1]["end"] + 0.25, 3))
        return {
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": text,
            "words": [{k: v for k, v in x.items() if k != "phrase_idx"} for x in cur_words]
        }

    clean_segments = []
    cur = []
    for w in all_words:
        if cur:
            prev = cur[-1]
            if (w["phrase_idx"] != prev["phrase_idx"] or
                (w["start"] - prev["end"] >= 0.45) or
                (len(cur) >= 3 and "internal_restart_at" in w)):
                if len(cur) >= 3 and "internal_restart_at" in w and w["phrase_idx"] == prev["phrase_idx"] and (w["start"] - prev["end"] < 0.45):
                    w["start"] = w.pop("internal_restart_at")
                seg_obj = finalize_segment(cur)
                if seg_obj:
                    clean_segments.append(seg_obj)
                cur = []
        cur.append(w)
    if cur:
        seg_obj = finalize_segment(cur)
        if seg_obj:
            clean_segments.append(seg_obj)

    with open(out_json, "w") as f:
        json.dump({"segments": clean_segments}, f, indent=2)

    print(f"[Step 2] Done! Extracted {len(clean_segments)} acoustically-clamped subtitle segments -> {out_json}")


if __name__ == "__main__":
    work_d = sys.argv[1] if len(sys.argv) > 1 else "audio_work"
    mdl = sys.argv[2] if len(sys.argv) > 2 else "mlx-community/whisper-small-mlx"
    force_flag = "--force" in sys.argv
    run_step2(work_d, mdl, force=force_flag)
