# MMC-Dance — TMD-Bench MLLM Visual–Audio Alignment Evaluation Plan

**Paper target:** CVPR 2027  
**Primary dataset:** FineDance (complete first)  
**Secondary dataset:** Motorica (only after FineDance results are complete)  
**Evaluation judge:** Gemini **2.5 Pro** via TAMU AI Chat, subject to verified video+audio support  
**Source clip duration:** **34 seconds**  
**Evaluation protocol:** TMD-Bench **Visual Audio Alignment** prompt; one **1–5 score per video**  
**Compared methods:** Released MACE-Dance and MMC-Dance (ours)  
**Execution policy:** Mandatory user approval after every numbered step (and every bulk-evaluation batch)

**Not in this plan:** Reimplementing VBCS/ABHS/CSD/HSD; pairwise preference judging; custom musicality prompts; additional perturbation controls; 95% confidence intervals; paired-difference tests; significance testing; or an invented combined evaluation score.

## 1. Why this judge and evaluation

Use TMD-Bench's published **Visual Audio Alignment** MLLM judging prompt to compare perceived synchronization of rendered 3D dance videos with their accompanying music. Evaluate the outputs of MACE-Dance and MMC-Dance using the **same input songs** and matched rendering conditions.

**Why Gemini 2.5 Pro:** TMD-Bench selected **Gemini 3.0 Pro** for its overall benchmark, but Gemini 3.0 Pro is **not available** to us. We therefore use the accessible **Gemini 2.5 Pro** on TAMU AI Chat. Importantly, this is not an unvalidated substitute: **TMD-Bench Appendix D, Table 6** evaluated Gemini 2.5 Pro against **human ratings** (10 raters assessing 100 generated music–dance videos). On its **Visual–Audio Rhythmic Alignment (VA)** dimension, Gemini 2.5 Pro obtained **PLCC = 0.558, SRCC = 0.514, and QWK = 0.410** against the human ratings. These positive correlations provide supporting evidence that its judgments track human-rated rhythmic alignment to a meaningful, though imperfect, degree. Gemini 3.0 Pro was strongest on the paper's *overall* multi-dimension comparison; we are **not** claiming Gemini 2.5 Pro is universally better. This is an **adaptation** of the TMD-Bench judging protocol, not an exact replication of its chosen model. The cited results concern **human ratings**, not a proven ability to reproduce every human preference or guaranteed validity on our rendered 3D dance videos.

The output is a **model-judged alignment score**. Do not describe it as human preference measured in this study. Do not add BPL/IMC criteria to the TMD-Bench prompt. Keep objective BPL, intensity coupling, BAS, and motion-quality evaluations separate from the MLLM score.

**TMD-Bench source:** Yang et al., *TMD-Bench: A Multi-Level Evaluation Paradigm for Music–Dance Co-Generation*, Section 3.3, Section 5.3.4/Table 4, Appendix D and Appendix E/Figure 6. https://arxiv.org/abs/2605.01809

## 2. Data preparation: 34-second matched clips

- Use **34.0-second** dance clips when full synchronized music and motion are available (same duration as the generation setup).
- For a given music excerpt, MACE-Dance and MMC-Dance clips must have the **same song, start/end timestamps, avatar, camera, framing, resolution, background, lighting, video encoding, and audio mix**.
- Retain the source file's existing **24 FPS or 30 FPS**; do not conflate source video FPS with the FPS at which Gemini samples its frames.
- Remove overlays/watermarks or other visual labels that reveal the model identity. Maintain a private mapping of video IDs to methods.
- Do not pad clips artificially to 34 seconds. Flag shorter or invalid clips and use a consistent, declared inclusion rule.
- Render audio and video together in one file (e.g., MP4 with audible music). Confirm the clip is not silent.

## 3. Judge FPS: use high temporal sampling if TAMU supports it

| Source video | Preferred Gemini video sampling | Important note |
|---|---|---|
| 24 FPS | Request **24 FPS** | Must confirm TAMU actually accepts/uses this setting |
| 30 FPS | Request **24 FPS** | Native Gemini video sampling is documented up to 24 FPS, not 30 FPS |

**Do not assume uploading a 24/30-FPS file forces Gemini to analyze 24/30 frames per second.** Default static video understanding is commonly 1 FPS. The ability to change video sampling through TAMU Chat's UI or API is **not yet verified**.

During the pilot, record the source FPS, requested judge FPS, and any **confirmed** effective judge FPS. If TAMU does not expose this setting, mark it **unverified/default** and do not call the resulting study a 24-FPS MLLM evaluation. Resolve this limitation before scaling if high judge FPS is a requirement.

This is a necessary technical configuration check of the **two pilot videos**, not a separate experimental sanity-check suite.

## 4. Phase 1: first evaluate only TWO FineDance videos

Choose **one FineDance song** and one **34-second** excerpt. Prepare:

- One **existing or newly prepared** MACE-Dance video for that excerpt.
- One **existing or newly prepared** MMC-Dance video for the **same** excerpt.

**Reuse videos already present in the project wherever possible.** Do not generate duplicates or move files unnecessarily. The agent should inspect the existing codebase to locate the appropriate videos; it must not assume any particular paths or filenames.

Keep the method identity private from the judge. For **each video**, start a fresh TAMU Chat conversation/request, select **Gemini 2.5 Pro (not Flash)**, upload the video with synchronized audio, and send **the identical prompt in Section 5**.

**The pilot only needs to establish:**

1. Both 34-second video/audio files can be submitted and evaluated.
2. Gemini returns a valid `va_alignment` score **1, 2, 3, 4, or 5** and a reasoning field.
3. Source FPS versus accepted judge sampling FPS is documented.
4. End-to-end request time, reported token/quota usage, and actual or estimated cost are recorded for **both** videos.

No shifted-audio, wrong-song, extra preference, or repeated-run sanity checks for now. If the technical pilot succeeds, use its measurements to decide how large the FineDance run can be.

### Pilot cost/time log

| Field | Pilot A | Pilot B |
|---|---|---|
| Song ID / clip start / duration (34 s) | | |
| Video source FPS (24 or 30) | | |
| Judge requested FPS / confirmed effective FPS | | |
| File size (MB) | | |
| Upload + processing time (s), if separable | | |
| Gemini response time (s), if separable | | |
| **Total elapsed time (s)** | | |
| Input tokens (if available) | | |
| Output tokens including thinking (if available) | | |
| TAMU quota/credits before and after (if visible) | | |
| Actual TAMU charge, if any | | |
| Public Gemini API-equivalent estimated cost | | |
| `va_alignment` score (1–5) | | |
| Raw response path / request info | | |

**Cost rules:**

- Use **actual TAMU charges** if disclosed. TAMU's service may be free within daily quota; do not equate public API pricing with a bill from TAMU.
- Use **reported usage tokens**, when available. If TAMU hides usage, mark it unknown rather than inventing token counts.
- For comparison only, check the current public `gemini-2.5-pro` pricing at https://ai.google.dev/gemini-api/docs/pricing. At the previously documented standard paid rates, prompts up to 200k input tokens were priced at $1.25 per million input and $10 per million output tokens (including thinking); higher-input-tier rates were $2.50 and $15 respectively. These amounts **must be reconfirmed** if used for costing.
- At **24 sampled FPS**, a 34-second video may involve around **816 sampled frames**, which can be expensive or exceed an institutional request limit. The pilot should establish whether the intended configuration is feasible.

After the two videos, compute the **observed mean elapsed time** and, if measurable, **mean input/output tokens and mean cost** per video. Multiply by the **number of remaining videos in the declared FineDance scope** to obtain a planning estimate, clearly marked as an estimate rather than a guaranteed time/cost. Do not run Motorica yet.

## 5. TMD-Bench's original Visual Audio Alignment prompt (ready to paste)

**Source:** Appendix E, Figure 6 of the user-provided TMD-Bench paper. Transcribed from the figure, with line wrapping/spacing normalized for readability; the scoring definitions and fields are preserved. The original template says **“Video B”** even though it evaluates one generated video at a time. Keep this as written for consistency. In the JSON schema, `A` means the integer rating **1–5**, not the pilot's video letter.

Copy everything inside the following fenced block as the prompt for **each** video:

````text
You are a strict video-audio alignment evaluator for music/dance videos.You will Evaluate how well the VISUAL motion rhythm aligns with the AUDIO rhythm.

You will be given:
1. **Video B**: the generated video
--------------------------------------------------------------
## Key Definitions
- Audio beats: salient rhythmic pulses in the soundtrack (what a listener would clap/tap to).
- Motion beats: salient movement accents (peaks of motion intensity, sharp direction changes, footfalls, body hits, or clear pose transitions).
- Alignment: perceptual temporal coherence between audio beats and motion beats.
--------------------------------------------------------------
## Reasoning Steps
### 1. Alignment Assessment
Evaluate temporal relationship between audio beats and motion accents:
- Timing proximity: do motion hits land near audio beats?
- Coverage: how many audio beats are answered by motion accents?
- Stability: is alignment consistent, or only present briefly?
- Errors: systematic early/late offsets, drift, or random mismatches?
--------------------------------------------------------------
### 2. Decision
- **5 – Excellent**: Extremely tight synchronization: motion accents reliably land on or very near beats; drift and timing errors are minimal; sustained alignment throughout the clip.
- **4 – Strong**: Good synchronization but not perfect; mostly aligned with occasional jitter, misses, or partial drift; still clearly beat-aware.
- **3 – Mixed/Moderate**: Partial alignment; some segments synced and others off; intermittent or unstable alignment; common in generic dance videos.
- **2 – Weak**: Sparse or unreliable alignment; motion rarely responds to audible beats; may show only vague rhythmic awareness.
- **1 – Poor/None**: Mostly unaligned; motion does not correspond to beat structure; OR video is silent/no audio; OR no meaningful rhythmic correspondence is detected.
--------------------------------------------------------------
## Output Format
Return ONLY one JSON object with the following fields:
{
  "va_alignment": A,
  "reasoning": "..."
}
````

For automation, the same prompt is additionally saved as `tmd_va_original_prompt.txt` alongside this Markdown document (and included in the ZIP file). **Do not rewrite its rubric** to favor BPL or intensity-matching.

## 6. Scoring: follow TMD-Bench's simple model comparison

TMD-Bench's main **Table 4** reports an aggregate MLLM `Align.` score per generation model and **does not display confidence intervals, paired statistical tests, or significance tests** for that table. Follow that simple comparison style.

1. Submit every video **individually** with the unchanged prompt.
2. Parse `va_alignment` from the response and preserve the original **integer 1–5** score for that video.
3. For each model in each dataset, calculate the **arithmetic mean across the evaluated videos** (same declared clip set for both models). Compare the two model averages directly.
4. Preserve all raw responses and count missing/invalid responses. Report the **number of clips evaluated**.
5. Report **FineDance and Motorica separately**. Do not combine objective and MLLM scores or invent a multi-metric aggregate.

**Important replication distinction:** The original prompt specifies **1–5**, but TMD-Bench's Table 4 displays `Align.` on a **0–1 scale**. The exact transformation to that table scale is not established from the available methodology. Therefore, report the transparent **mean raw 1–5 score**, labeled `Mean VA Alignment (1–5)`, rather than asserting it is numerically identical to TMD-Bench's published `Align.` values.

**Do not add:** 95% CIs, paired-difference statistics, sign-flip/permutation tests, p-values, or win/tie/loss ratios. These are not required for the requested TMD-Bench-style main comparison.

## 7. Phase 2: finish FineDance completely FIRST

- Inventory existing FineDance videos first. Freeze the FineDance evaluation set before interpreting results. Existing working materials refer to **19 songs and 84 slices of 34 seconds**; confirm the usable counts and select the intended coverage before scaling.
- **Preferred full run:** one video per method per slice, so if 84 slices are valid there will be **168 total videos**, including any qualifying pilot pair.
- If the two-video pilot shows that the time/quota/cost is too high, declare a smaller, song-balanced FineDance subset **before** scoring it. For example, one 34-second excerpt from each of 19 songs = **38 videos total**. Report that this is a subset rather than full FineDance coverage.
- Keep the judge model, prompt, rendering, music, FPS configuration, duration, and clip selection rules identical for both methods.
- Calculate and save **one mean VA Alignment (1–5)** per model across the selected FineDance clips; report the clip and song counts.
- **Complete FineDance and finalize its table before beginning Motorica.**

## 8. Phase 3: Motorica SECOND

Only after FineDance is complete:

- Apply the **same fixed prompt** and Gemini 2.5 Pro model to the Motorica 34-second video excerpts.
- Use matched songs and identical video preparation and scoring for MACE-Dance versus MMC-Dance.
- Estimate Motorica time/quota using the measured FineDance/pilot requests before running the secondary set.
- Report Motorica separately as secondary/out-of-domain evidence.

## 9. Final result format

| Model | FineDance mean VA Alignment (1–5) ↑ | Motorica mean VA Alignment (1–5) ↑ |
|---|---:|---:|
| Released MACE-Dance | TBD | TBD |
| MMC-Dance (ours) | TBD | TBD |

Caption: *TMD-Bench Visual Audio Alignment rubric (Appendix E, Figure 6), evaluated with Gemini 2.5 Pro via TAMU AI Chat because Gemini 3.0 Pro was unavailable. Scores are averages of the model's original 1–5 ratings for identical 34-second source excerpts. Video/frame sampling details and sample counts reported in implementation details.*

## 10. Integrate into the existing codebase (no imposed folder structure)

The agent must **inspect the existing repository and reuse its current assets and conventions** before proposing any new code or files. Some generated videos already exist. The agent is responsible for choosing appropriate implementation locations, filenames, manifests, and result storage based on the project as found.

- **Do not create a prescribed `mmc_mllm_eval/` tree** or reorganize the repository just to match this plan.
- Reuse existing videos, dataset IDs, metadata, scripts, and results infrastructure wherever they are suitable.
- Keep the original TMD-Bench prompt accessible to the evaluation code. A copy is embedded in **Section 5** of this document and supplied separately as `tmd_va_original_prompt.txt` for convenience.
- Store raw Gemini outputs, the exact prompt/model identifier, media configuration, clip identity, scores, and pilot usage information in the conventions chosen for the existing repository.
- Any new files, paths, or changes that could affect existing code should be **proposed to the user at the relevant approval checkpoint before implementation**.

**Suggested data fields, not a required file or folder layout:**

```csv
dataset,song_id,clip_id,method,video_path,duration_s,source_fps,requested_judge_fps,confirmed_effective_judge_fps,gemini_model_id,va_alignment,raw_response_location,elapsed_seconds,input_tokens,output_tokens,actual_tamu_cost_usd,estimated_public_api_cost_usd,status
```

## 11. Strict execution order: mandatory human approval after EVERY step

**Human-in-the-loop rule:** Execute **one numbered step at a time**. After completing the step, report (a) what was done, (b) evidence/results or paths, (c) uncertainties, (d) expected effect on time/cost if relevant, and (e) the proposed next step. Then **STOP and explicitly ask the user to approve proceeding**. Never interpret silence as approval, never launch the next step automatically, and never run FineDance or Motorica in the background. If a step is blocked or changes scope, present options and wait for user direction. The user may request changes at any checkpoint.

1. **Inspect existing codebase and media.** Locate already-generated dance videos and related metadata; identify how evaluation utilities fit the repository. Propose a minimal implementation approach with no arbitrary directory structure. **Stop for user approval.**
2. **Select the two-video FineDance pilot.** Identify one song/excerpt and matched MACE-Dance and MMC-Dance videos, preferably existing 34-second outputs. Verify video/audio synchronization, duration, source FPS, and comparable rendering. **Stop for user approval.**
3. **Verify TAMU Chat feasibility and configuration.** Confirm that the available judge is **Gemini 2.5 Pro**, synchronized video+audio can be submitted, and whether **24 FPS judge sampling** can actually be requested. Record unsupported or unverifiable settings without silently changing requirements. **Stop for user approval.**
4. **Run only the two pilot judgments.** Submit each video independently using the **unmodified Section 5 prompt**. Save the raw response and `va_alignment` values. Do not submit any other clips. **Stop for user approval.**
5. **Measure pilot resources and plan the FineDance run.** Report elapsed time per request, visible input/output token or quota usage, actual TAMU charges (if any), and clearly labeled public-API-equivalent estimates where defensible. Estimate total resource needs from the observed two-video pilot, stating unknowns. **Stop for user approval.**
6. **Confirm the complete FineDance scope.** Inventory eligible 34-second clips and agree on the final song/clip count for both methods based on feasibility and available media. Do not begin bulk evaluation until the user approves the set. **Stop for user approval.**
7. **Evaluate FineDance, using approved batches if needed.** Run the unchanged TMD-Bench prompt on the agreed matched videos. **After every batch, report scores, errors, consumed quota, and costs; stop for explicit approval before the next batch.** When FineDance is finished, do not automatically start Motorica.
8. **Finalize FineDance results.** Calculate mean raw VA alignment score (1–5) per method; report the number of videos, invalid responses, FPS configuration, and a simple comparison as in TMD-Bench's main table. **Stop for user approval.**
9. **Plan Motorica only after FineDance completion.** Locate matched existing Motorica videos, verify the 34-second coverage, and estimate time/cost using the pilot/FineDance observations. **Stop for user approval.**
10. **Evaluate Motorica, using approved batches.** Apply the same prompt and evaluation setup; **stop for approval after each batch**. Report Motorica's mean raw 1–5 VA alignment scores separately. **Stop for user approval before any further analyses or publication exports.**

At each checkpoint, the agent should ask a concrete question such as: **“Step N is complete. Here are the results and the proposed next action. Do you approve Step N+1?”**

## References

- TMD-Bench manuscript, including exact Figure 6 prompt: https://arxiv.org/abs/2605.01809
- TAMU AI Chat: https://www.it.tamu.edu/services/services-by-category/communication-and-collaboration/tamu-ai-chat.html
- Gemini video understanding and FPS parameters: https://ai.google.dev/gemini-api/docs/video-understanding
- Gemini API public pricing (comparison only; not TAMU billing): https://ai.google.dev/gemini-api/docs/pricing
