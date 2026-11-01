# Independent video workflow

For installation/onboarding requests, follow `docs/SETUP-AGENT.md` first. Do not run paid inference during setup. Distinguish a running UI from a generation-ready configuration.

Carry authorized jobs through preparation, quality review, generation and delivery without stopping after each processing stage. Ask only for missing creative choices or submission approval not already given. Never automatically purchase an HD upgrade: the user approves that step.

1. Preserve the source file and complete original audio.
2. Isolate vocals with Demucs, pitch +3 semitones with timing preserved, and embed them in the prepared composite. Do not attach separate audio to Seedance.
3. Generate depth with fal-ai/depth-anything-video and retain its exact colors.
4. Obtain raw SAM 3 masks. If depth segmentation misses the subject or includes background, retry using original RGB detail and composite the original colored depth through those masks. Record this fallback.
5. Mandatory visual mask QA before EVERY generation: review the full clip, including all cuts, subject extremities, hair, hands, clothing, background spill, holes, alignment and flicker. Automated checks and sampled contact sheets alone do not establish full visual approval. Repair failures independently within the authorized task, then review again. Never approve a known-bad mask. Use mask_review to bind approval to current artifacts. User approval of the current mask may be recorded as such, without falsely claiming agent full-clip review.
6. Match the references and prompt to the actual subjects. A single Sirio replacement uses only Sirio's reference; never carry over the previous two-person mapping.
7. Show the exact creative prompt before submission. Existing explicit authorization to submit is sufficient; do not ask again unnecessarily. Submit through Enhancor as Seedance Edit draft by default.
8. Retrieve results and restore the entire untouched original source audio, discarding generated audio. Preserve draft artifacts. Report failures honestly; never claim submission or completion without an actual result.
9. On explicit approval, complete the draft at 1080p using draft_id and webhook_url, then restore original audio again.
10. Show the final preview and download. Keep status persisted and UI preview selection stable. Never expose credentials.

## Reference-mode routing (required)
Enhancor Edit accepts exactly one video: the prepared source. If the replacement identity reference is a video, use multi_reference (Omni), with prepared source @Video1 and identity reference @Video2. Use the identity clip for appearance only unless the user explicitly asks to use its voice or performance. Do not convert that reference to pictures without permission. For image references, default to Edit. Show the actual mode in the UI and saved generation metadata. Omni requires duration and aspect ratio; use the source framing and nearest supported duration. Preserve the full source audio in the final export and disclose any necessary short last-frame hold.


## Fast alternative: face mesh

Choose **Fast · Face mesh** in the UI, or use `workflow.py prepare VIDEO --mode face_mesh`. Install the optional worker once with `bash setup-face-mesh.sh` (Python 3.12 required; isolated `.venv-face`). The model downloads from Google's official storage on first use.

This path keeps the original video, tracks and saves facial landmarks, draws the mesh, isolates vocals and embeds the +3 pitched vocals. It makes **no depth or SAM calls**. It is currently designed for a single centered speaker; inspect all frames and stop on tracking gaps or incorrect face alignment. Do not claim mask review in this mode: review facial tracking and audio instead. Saved artifact hashes gate submission.

Use the same character references, draft-first submission, collection, original-audio restoration and explicitly approved HD upgrade. In the prompt, explain that mesh lines are motion guidance and must disappear, replace the complete original identity including hair, and preserve all dialogue and timing. Image reference uses Edit; video identity uses Omni. Keep advanced depth mode available. Do not submit automatically during setup.


### Combined depth + face mesh

Select **Advanced + Face mesh** in the UI or use `workflow.py prepare VIDEO --mode depth_mesh`. This runs the original depth/SAM pipeline, preserves its depth composite, then tracks the face from the original frames and overlays the mesh onto that composite. Landmarks are saved in `face-landmarks.json`. Review BOTH mask coverage and face tracking before approval. All artifacts, including depth, mask, mesh and embedded audio, are bound to review approval. Follow the same draft → approved HD → original-audio restoration flow.

For setup with all three preparation modes, also install Python 3.12 and run `bash setup-face-mesh.sh`. This creates an isolated mesh environment; the main app can retain its existing Python version. A Google Face Landmarker model is downloaded at first use. No SAM/depth inference is used by fast mode. Mesh modes are experimental and currently optimized for one centered speaker.
