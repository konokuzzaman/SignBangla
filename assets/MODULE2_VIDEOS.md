# SignBangla — Module 2: Text/Speech-to-Sign Language Synthesis Demos

This document showcases synthesized continuous sign language sentences produced by SignBangla's **Module 2 (Speech/Text &rarr; Sign)** pipeline. Each animation combines 67-point upper-body and hand skeletons retrieved from the high-quality BdSLW60 sign dictionary, velocity-trimmed, and stitched using smooth cosine-eased transitions.

---

## Synthesized Multi-Sign Sentence Videos

### 1. "বাবা চা বিস্কুট" (*Father Tea Biscuit*)
- **Sign Sequence**: W0 (`Father`) &rarr; W26 (`Tea`) &rarr; W27 (`Biscuit`)
- **Video File**: [`assets/videos/baba_cha_biskut.mp4`](videos/baba_cha_biskut.mp4)
- **Duration / Frames**: ~5.1 seconds (154 frames @ 30 FPS)
- **Features**: Includes rest pose entry, two 12-frame cosine-interpolated joint transitions between signs, and smooth return to rest pose. Bilingual HUD overlay displays current active sign token.

---

### 2. "ডাক্তার চিকিৎসা ব্যান্ডেজ" (*Doctor Treatment Bandage*)
- **Sign Sequence**: W49 (`Doctor`) &rarr; W50 (`Treatment`) &rarr; W52 (`Bandage`)
- **Video File**: [`assets/videos/doctor_chikitsa_bandage.mp4`](videos/doctor_chikitsa_bandage.mp4)
- **Duration / Frames**: ~9.0 seconds (270 frames @ 30 FPS)
- **Features**: Multi-phase complex medical vocabulary synthesis demonstrating fine-grained two-handed coordination and spatial positioning.

---

### 3. "মা আপেল চকোলেট" (*Mother Apple Chocolate*)
- **Sign Sequence**: W6 (`Mother`) &rarr; W15 (`Apple`) &rarr; W28 (`Chocolate`)
- **Video File**: [`assets/videos/ma_apel_chokolet.mp4`](videos/ma_apel_chokolet.mp4)
- **Duration / Frames**: ~4.7 seconds (140 frames @ 30 FPS)
- **Features**: High-speed dynamic wrist and finger articulate motions stitched cleanly with zero skeletal jitter.

---

## Synthesis Pipeline Architecture

```text
[Input Text / Speech Transcript]
              │
              ▼
   [Bengali / English Tokenizer]
              │
              ▼
    [Dictionary Matcher]  ◄── BdSLW60 Curated Sign Dictionary (3 best trials/word)
              │
              ▼
    [Idle-Frame Velocity Trimmer]
              │
              ▼
    [Cosine-Eased Joint Stitcher] (12-frame inter-sign smooth interpolation)
              │
              ▼
   [67-Point Skeleton Renderer] (Bilingual HUD + MP4 30 FPS output)
```

---

## How to Synthesize Custom Sentences

To generate a new skeleton video from any sequence of words in the BdSLW60 vocabulary:

```bash
# English words
python scripts/20_m2_demo.py --sentence "Father Tea Biscuit" --out work/custom_demo.mp4

# Or Bengali words
python scripts/20_m2_demo.py --sentence "বাবা চা বিস্কুট" --out work/custom_demo.mp4
```
