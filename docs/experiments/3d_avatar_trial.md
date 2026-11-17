# 3D avatar trial (5 Nov 2026): abandoned

**What was tried.** Ten BdSLW60 dictionary signs (মা, বাবা, ভাই, বোন, চা, আম, চশমা, টিভি, ডক্টর, জুতা) were converted from our normalized 67 keypoints into MediaPipe-style input: 33 pose points in meters around the hips, plus 21 points per hand. They then drove a VRM avatar (pixiv's VRM1_Constraint_Twist_Sample, VRM Public License 1.0) in a static web page using three.js 0.164.1, @pixiv/three-vrm 2.1.3 and Kalidokit 1.1.5 (`Pose.solve` and `Hand.solve`). A headless-Chromium test compared the avatar's bone positions with the keypoints on every frame.

**Why it failed.**
- **Arms:** Kalidokit could not bend the elbow up to the face. The signing forearm was a median 57° off the keypoints and stayed near the waist in 9 of 10 words. Kalidokit is tuned for webcam avatars: it solves each forearm as an absolute direction, clamps it, and handles depth poorly. MediaPipe's exaggerated image depth made this worse.
- **Mirrored hands and axes:** Kalidokit is built as a mirror: its avatar "Right" arm comes from the person's left shoulder. It also assumes VRM 0.x bone axes. The data had to be mirrored, and the x and z rotations flipped; without the flip the upper arms were 120–150° wrong.
- **Twisting:** aiming the bones straight at the keypoints instead put the hands in the right places, but with no joint limits the wrists twisted up to 174° against the forearm.
- **Fingers:** `Hand.solve` gives each finger one curl angle and no sideways spread, so thumbs stuck out and handshapes such as জুতা looked wrong. The left hand in one-handed signs is an averaged rest shape, with some impossible finger bends.
- **Other problems:** the rest pose crossed the hands in front of the waist. There was no face, although BdSL uses facial expression. The 3D stack (three.js, three-vrm and Kalidokit versions, VRM 0/1 conventions) needed many workarounds for an uncertain result.

**Decision.** Replace it with a stylised 2D avatar drawn directly from the same 67 keypoints, which keeps the handshapes exactly as recorded.

## 2D avatar attempts (5 Nov 2026): also dropped

- **Simple 2D figure:** a flat figure drawn from the same 67 keypoints: filled torso, round head, capsule arms with different left/right sleeve colours, and thick rounded finger segments. The arms and hands followed the keypoints exactly, and the far arm was drawn first by depth. But it looked like a diagram rather than a person, and a hand in front of the face (same skin colour) was separated only by outlines.
- **Illustrated characters:** a male (short hair, light beard, panjabi) and a female (bun, salwar kameez, orna) Bangladeshi character with two brown skin tones, drawn as vector shapes with gradients, shaded hands from all 21 points, and blinking and breathing at rest. It needed a new drawing library (pycairo), took about 19 ms per frame (under the 60 fps target), and still could not show depth well: arms pointing at the camera and fingers bent toward it flatten in 2D.
- **Decision:** both styles were removed. The skeleton is the only drawing style; the motion work made for the avatars (glitch removal, smoothing, fixed bone lengths, joint-angle transitions, holds, 60 fps, speed control, highlighted captions) was kept because it improves the skeleton too.

## Blender + MPFB attempt (5 Nov 2026): dropped

**What was tried.** Two characters (a man and a woman, brown skin, black hair) were made in Blender 5.2 with MPFB 2.0.18 (MakeHuman for Blender) and the CC0 MakeHuman system assets. Each had MPFB's standard rig of 163 bones, with 3 per finger and the upper arm and forearm split for twist. A plain long tunic and trousers were cut from the body mesh, smoothed, and kept 1–2.5 cm off the skin. The keypoints went through the skeleton demo's motion pipeline: glitch removal, smoothing, holds, and minimum-jerk transitions from and back to rest. Custom retargeting then set every bone:
- arm directions from the keypoints, with depth taken from how short each bone looks on screen
- the elbow as a hinge
- wrist twist from the palm, split along the forearm
- every finger joint aimed in turn
- joint limits, with the lower body kept still

The result was rendered with Eevee, next to the skeleton and with keypoint markers. Only the Phase 3 checkpoint was reached: one word (মা), male character.

**What went wrong.**
- **It looked bad.** The figures read as generic mannequins. The clothes were a fitted long-sleeved bodysuit rather than a loose panjabi or kameez, with a jagged neckline, no collar, flare or orna, and no beard (there is no CC0 beard asset). The female "braid" hair looked like a side-swept bob from the front, and the skin was shiny. Eevee left grainy shadows on the chest (`blender_trial_characters.png`, `blender_trial_W20_mid.png`).
- **The numbers matched, the meaning did not.** The bones pointed within 0–5° of the keypoints and the handshape followed them (`blender_trial_W20_hand.png`). Even so, the hand did not touch the chin, because contact was not enforced. The elbow limit (150°) left the forearm up to 16° short in 28 of 89 frames, since depth from one camera is a guess. The left hand in one-handed signs is the dictionary's averaged rest hand, so one finger stayed 52° off.
- **The cost was high for a trial.** It needed a 46 MB extension and a 281 MB asset pack outside the project, `.blend` files that only open where those are installed, two runtimes (Blender's Python has no scipy), and about 0.65 s per frame to render.

**Decision.** Removed. The skeleton animation, with the improved motion, stays the only output.

| Evidence | File |
|---|---|
| Both characters in the rest pose | `blender_trial_characters.png` |
| মা, middle of the sign: character with keypoint markers next to the skeleton | `blender_trial_W20_mid.png` |
| Close-up of the signing hand (plain and with markers) | `blender_trial_W20_hand.png` |
