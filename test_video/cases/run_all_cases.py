#!/usr/bin/env python3
"""
Fast Micro-Test Runner (~12s total) for all 3 edge-case clips in test_video/cases/
"""
import os, json, subprocess, time

SKILL_DIR = "/Users/philma/.gemini/config/skills/ai-talking-head-editor"
CASES_DIR = "/Users/philma/codes/ai-talking-head-editor/test_video/cases"

# Clean deduplicated EDLs for each test case
CASE_EDLS = {
    "case1_tts_and_drills": [
        {"in": 35.80, "out": 43.38, "text": "I carried a rind over to a chicken coop and fed the chickens with dead flies."}
    ],
    "case2_repeated_sentences": [
        {"in": 0.60, "out": 2.60, "text": "After my uncle introduced us,"},
        {"in": 10.50, "out": 13.26, "text": "I fed him some apples and other treats"},
        {"in": 13.71, "out": 16.95, "text": "and he soon recognized me as a friend."},
        {"in": 25.35, "out": 29.24, "text": "I liked to play with the dog and I could tell he liked me too."},
        {"in": 30.15, "out": 34.71, "text": "Whenever I played with him, I treated him like a horse,"},
        {"in": 35.07, "out": 40.92, "text": "grabbing his fur, playing with his paw and trying to climb up to his back."},
        {"in": 45.50, "out": 52.56, "text": "But my uncle wouldn't allow me to play with him alone because the dog was far too big for me."}
    ],
    "case3_mid_sentence_pause": [
        {"in": 0.35, "out": 5.97, "text": "The TV only had five channels and it didn't play cartoons at all time."},
        {"in": 13.15, "out": 19.11, "text": "They had two fierce cats with short tempers which made them difficult to play with."}
    ]
}

t0 = time.time()
for name, edl in CASE_EDLS.items():
    work_d = os.path.join(CASES_DIR, f"work_{name}")
    os.makedirs(work_d, exist_ok=True)
    edl_path = os.path.join(work_d, "step3_edl.json")
    with open(edl_path, "w") as f:
        json.dump(edl, f, indent=2)
    in_mov = os.path.join(CASES_DIR, f"{name}.mov")
    out_mp4 = os.path.join(CASES_DIR, f"{name}_out.mp4")
    subprocess.run([
        "python3", f"{SKILL_DIR}/scripts/step3_render_mp4.py", edl_path, in_mov, out_mp4, work_d
    ], check=True)
print(f"\nAll 3 test cases rendered in {time.time() - t0:.2f}s!")
