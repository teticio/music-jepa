# Music JEPA — Architecture Walkthrough

A ground-up explanation of how the model works, from raw audio to track embeddings.
Code references point to the live source files; update them if the files move.

---

## Architecture overview

```
TRAINING (one step)

 Playlist: [... track_N,      track_N+1 ...]
                   │                 │
           context spec        target spec
           (1, 96, 216)       (1, 96, 216)
                   │                 │
        ┌──────────┴──────┐   ┌──────┴──────────┐
        │   Patch Embed   │   │   Patch Embed   │
        │  8×8 → 324×384d │   │  8×8 → 324×384d │
        └──────────┬──────┘   └──────┬──────────┘
                   │                 │
        ┌──────────┴──────┐   ┌──────┴──────────┐
        │  Context ViT-S  │   │  Target ViT-S   │ ← EMA copy, no grad
        │  12L · 6H · 384d│   │  12L · 6H · 384d│
        └──────────┬──────┘   └──────┬──────────┘
             (B,324,384)             │ gather M=240
                   │          (B,240,384) ──────────┐
        ┌──────────┴──────┐         │               │
        │    Predictor    │ ← positional queries    │
        │  4L · 6H · 192d │   for 240 positions     │
        └──────────┬──────┘                         │
             (B,240,384)                            │
                   └──────── MSE loss ──────────────┘
                                 +
                           VICReg loss
                      (on mean-pooled targets)


INFERENCE (embedding extraction)

 spectrogram (1, 96, 216)
                │
        ┌───────┴─────────┐
        │  Context ViT-S  │
        └───────┬─────────┘
           (B, 324, 384)
                │
        mean pool over 324 patches
                │
        track embedding (B, 384)  →  embeddings/embeddings.npy
```

---

## 1. The input: mel spectrograms

A raw MP3 is a pressure wave over time. Rather than feeding that directly to the
model, each 30-second Spotify preview is converted to a **mel spectrogram** — a
2D image where the x-axis is time and the y-axis is frequency, with pixel
intensity encoding energy.

The "mel" scale compresses the frequency axis to match human hearing: we are
sensitive to small differences at low frequencies and much less so at high
frequencies, so the scale is logarithmic. The result encodes musical content
perceptually — a bass drum hit looks different from a hi-hat, a chord different
from a single note.

In this project only the **first 5 seconds** of the preview is used, producing
a **96 × 216** image: 96 mel frequency bins tall, 216 time frames wide (~23ms
per frame). Images are stored as PNGs in `data/spectrograms/`. The playlist
CSVs provide the only training signal: which tracks appear consecutively.

---

## 2. Patch embedding — from image to sequence

Transformers don't process images pixel by pixel (96 × 216 = 20,736 inputs is
too many). Instead the spectrogram is chopped into a grid of non-overlapping
**8 × 8 pixel patches** and each patch is treated as one element in a sequence.

With an 8 × 8 patch size the grid is **12 rows × 27 columns = 324 patches**.
Each patch covers roughly 184ms of audio across 8 frequency bands.

The conversion is a learned linear projection implemented as a strided
convolution ([jepa/vit.py:12](../jepa/vit.py#L12)):

```python
self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
```

A convolution with `kernel_size = stride = 8` is exactly "chop into
non-overlapping 8 × 8 tiles and apply a learned linear map to each one." Each
patch's 64 raw pixel values are projected to a **384-dimensional vector**. The
full pipeline:

```
(1, 96, 216) image
  → conv with kernel=8, stride=8
  → (384, 12, 27) feature map
  → flatten spatial dims
  → (324, 384) patch vectors
  → + positional embeddings
  → ready for transformer
```

**Are these "tokens"?** Not in the NLP sense. In an LLM, a word is first mapped
to a discrete integer from a fixed vocabulary, then looked up in an embedding
table. Here there is no discrete step — 64 pixel values go *directly* to a
384-d vector via a learned matrix multiply. The patch vectors are equivalent to
the *already-embedded* tokens at the input of an LLM transformer: continuous
384-d vectors, not integers. Two nearly-identical patches produce nearly-identical
vectors, which is appropriate for audio where small spectrogram differences
should mean small representation differences.

**Positional embeddings** ([jepa/vit.py:65](../jepa/vit.py#L65)) are added to
each patch vector to tell the transformer where in the 12 × 27 grid each patch
came from. Without them the model has no notion of spatial layout.

---

## 3. The Vision Transformer — contextualising the patches

After patch embedding there are 324 vectors of 384 dimensions. The transformer's
job is to let every patch gather information from every other patch, so each
vector reflects not just "what is in my 8 × 8 tile" but "what is in my tile
*given the whole spectrogram*."

This happens through **self-attention**, repeated 12 times (12 transformer
blocks). In one round, each patch simultaneously computes:

- **Query** — what am I looking for?
- **Key** — what do I advertise about myself?
- **Value** — what will I share if someone attends to me?

Every patch takes the dot product of its query against every other patch's key.
High dot product → strong attention. The scores are softmaxed into weights
summing to 1, then used to take a weighted average of all values. Each patch
updates itself by blending in information from whichever other patches it found
relevant ([jepa/vit.py:36-40](../jepa/vit.py#L36-L40)):

```python
h = self.norm1(x)
h, _ = self.attn(h, h, h)   # Q = K = V = h: self-attention
x = x + h                    # residual: refine, don't replace
x = x + self.mlp(self.norm2(x))
```

After 12 rounds, a patch in the bass-heavy lower rows can "know" about the
rhythmic pattern in distant time columns — information has flowed freely across
the whole sequence.

There is also a **CLS token** — a learned vector prepended to the 324 patches
that can aggregate a global summary. It participates in all the attention but is
stripped from the output ([jepa/vit.py:96](../jepa/vit.py#L96)):

```python
return x[:, 1:]  # patch tokens only: (B, N, D)
```

Why it is discarded is covered in Section 9.

**Output of the encoder:** 324 vectors of 384 dimensions, each contextualised
by the whole spectrogram.

---

## 4. The JEPA training objective

The core idea in one sentence: **given the encoding of track N, predict the
patch-level representations of track N+1.**

Consecutive tracks in a playlist provide the supervision. The assumption —
borrowed from how V-JEPA treats video frames — is that consecutive playlist
tracks are coherently related: same DJ set, same mood, same genre. The model
must figure out *what* makes them related purely from audio.

### Predicting in latent space

We do not predict track N+1's raw pixels. We predict its **latent patch
representations** — the 384-d vectors that come out of the target encoder when
it processes track N+1. From the I-JEPA paper:

> *"The key idea is to predict representations of the target in latent space,
> without having to predict every detail of the target image in pixel space."*

Predicting pixels forces the model to reconstruct exact textures, recording
artefacts, irrelevant noise. Predicting in latent space means the target encoder
has already abstracted those details away. The predictor must learn something
semantically meaningful to match it.

### The masking — what M is

Rather than predicting all 324 target patches, a **contiguous block of time
columns** is selected as the prediction target, covering about 75% of the grid.
With 27 columns and mask ratio 0.75: `round(27 × 0.75) = 20` columns masked,
each spanning all 12 frequency rows.

**M = 12 × 20 = 240 patches** to predict; 84 patches remain as visible context
([jepa/dataset.py:84-91](../jepa/dataset.py#L84-L91)).

The block is always time-aligned (whole columns, all frequency rows), mimicking
V-JEPA's temporal tube masking. This forces the predictor to reason about what
happens *over a period of time* in the target track — much harder than filling
in a few scattered patches.

### The loss

Simple MSE between predicted and actual target patch vectors
([jepa/loss.py:48](../jepa/loss.py#L48)):

```python
mse = F.mse_loss(predicted, target.detach())
```

---

## 5. The EMA target encoder — preventing collapse via asymmetry

### What collapse is

Mode collapse does not require the model to "know" the target. It can happen
purely through gradient descent discovering a shortcut: if the context encoder
learns to output the **same constant vector for every input**, and the target
encoder does the same, the predictor just has to learn one constant output and
MSE goes to zero. Training "succeeds" but the embeddings are completely useless.

The model never needs to cheat on any individual example — it just needs to
discover that "ignore the input, output a constant" is a valid global minimum of
the MSE loss. Gradient descent will find it if nothing prevents it.

### How EMA breaks the conspiracy

The target encoder is initialised as an exact copy of the context encoder, then
updated only as an **exponential moving average** of the context encoder's
weights — never directly by gradients
([jepa/model.py:97](../jepa/model.py#L97)):

```python
p_tgt.data.mul_(m).add_(p_enc.data, alpha=1 - m)
```

With momentum m ≈ 0.996–0.999 the target encoder is a slow, smoothed copy —
always slightly behind the context encoder. For collapse to occur, both encoders
would need to simultaneously agree on the same constant output. They cannot:
if the context encoder drifts toward a constant, the target encoder is still
producing the previous, more varied representations, and the MSE loss penalises
the context encoder for not matching them. The two networks can never conspire
because one is always chasing the other.

This asymmetry was introduced by BYOL (Bootstrap Your Own Latent) and is the
mechanism JEPA inherits.

### Momentum schedule

Momentum ramps from 0.996 to 0.999 over training
([jepa/model.py:95](../jepa/model.py#L95)):

```python
m = 1 - (1 - self.ema_momentum) * (math.cos(math.pi * step / total_steps) + 1) / 2
```

Lower momentum early → target encoder updates faster, helping bootstrap learning
when both encoders are random. Higher momentum later → stable, slowly-evolving
targets that give the context encoder something solid to learn from.

---

## 6. VICReg — making collapse mathematically impossible

EMA asymmetry makes collapse hard to reach in practice but does not rule it out
in principle — a very slow drift toward constant outputs could still slip
through. VICReg adds two explicit penalties that close the loophole
([jepa/loss.py:5-30](../jepa/loss.py#L5-L30)).

By default VICReg is applied to the **predictor output** (mean-pooled to one
384-d vector per batch item), so gradients flow back into the predictor and
context encoder. The same penalty can also be evaluated on the EMA target
encoder's representations (still mean-pooled to `(B, D)`) as a diagnostic-only
mode that contributes no gradient — set `training.vicreg_target: target` in the
encoder config to switch. The original I-JEPA recipe uses MSE only and relies
on EMA asymmetry alone; gradient-flowing VICReg is a safety net inspired by
C-JEPA / V-JEPA's VCR variant.

### Variance term — no dimension may go constant

```python
std = torch.sqrt(z.var(dim=0) + eps)
var_loss = F.relu(1.0 - std).mean()
```

Look at each of the 384 dimensions *across the batch*. If any dimension has
standard deviation below 1, it is penalised. This directly prevents any
dimension from collapsing to a constant: if all B tracks output the same value
on dimension 47, variance is zero and the penalty is maximum.

### Covariance term — no two dimensions may say the same thing

```python
cov = (z.T @ z) / (B - 1)
cov_loss = cov.pow(2).fill_diagonal_(0).sum() / D
```

Penalises off-diagonal entries of the covariance matrix. If dimensions 12 and
47 always move together they are redundant and get penalised. Every dimension
must encode something distinct from every other. From the VICReg paper:

> *"The variance term prevents collapse of the representations to a point, the
> covariance term prevents collapse to a lower-dimensional subspace by
> decorrelating the variables."*

The covariance term closes a loophole the variance term alone leaves open: you
could satisfy variance with just 2 active dimensions and 382 near-zero ones —
technically non-constant, but still a severe collapse.

### Batch size and gradient accumulation

VICReg's statistics are computed *across the batch*. A very small batch (B=8)
might happen to contain 8 similar tracks, giving falsely low variance estimates
and firing a counterproductive penalty. B=32+ is generally sufficient; VICReg
was specifically designed to be less batch-size-sensitive than contrastive
methods (which need hundreds of negatives).

The full training config uses `accumulate_grad_batches: 4`
([configs/encoder.yaml](../configs/encoder.yaml)) with batch_size=64 on 2 GPUs,
giving an effective batch of 64 × 2 × 4 = 512 for the MSE loss. However,
gradient accumulation does **not** help VICReg's statistics — variance and
covariance are computed within each forward pass on the local 64-sample batch.
Accumulating gradients over 4 steps sums four independent 64-sample estimates
rather than pooling into one 256-sample estimate. VICReg sees 64 samples per
computation regardless of the accumulation setting. 64 is comfortably above the
threshold where this becomes a problem.

---

## 7. The predictor — cross-track prediction

The predictor takes the context encoder's 324 tokens and produces predicted
representations for the M=240 masked positions in the target track — without
ever seeing track N+1 ([jepa/model.py:43-67](../jepa/model.py#L43-L67)).

### Step 1 — project context down

```python
ctx = self.ctx_proj(ctx_tokens)  # (B, 324, 384) → (B, 324, 192)
```

The predictor works in 192 dimensions, half the encoder's 384. This is
deliberate: a weaker predictor cannot shortcut the task by re-encoding the
target itself.

### Step 2 — positional query tokens

```python
query_tokens = torch.gather(pos, 1, idx)  # (B, 240, 192)
```

For each of the 240 target positions a **learned positional embedding** is
looked up — a 192-d vector encoding "I am asking about position X in the target
spectrogram." These carry no information about what *is* at that position (the
predictor has not seen track N+1), only *where* the prediction is needed.

### Step 3 — self-attend over context + queries

```python
x = torch.cat([ctx, query_tokens], dim=1)  # (B, 324+240, 192)
for block in self.blocks:
    x = block(x)
preds = x[:, ctx.shape[1]:]  # extract query outputs only
return self.out_proj(preds)   # (B, 240, 384)
```

Context and query tokens are concatenated and run through 4 transformer blocks.
Query tokens attend to context tokens — pulling in whatever from track N is
relevant to predicting track N+1 at that position. Only the query outputs are
extracted; context token outputs are discarded.

The intuition: each query token asks *"given everything I know about track N,
what should the energy look like at this frequency-time position in track N+1?"*
The encoder only gets better representations by making this prediction possible.

---

## 8. One training step — end to end

Batch size B=64, so 64 consecutive playlist pairs.

1. **Sample:** for each item, pick a random playlist and a random consecutive
   pair. Load both spectrograms. Sample a random masked column block (240
   positions). Each item may have a different block position.

2. **Context encoder (gradients on):**
   `ctx_spec → patch embed → 324 tokens → 12 blocks → (64, 324, 384)`

3. **Target encoder (no gradients):**
   `tgt_spec → patch embed → 324 tokens → 12 blocks → gather 240 patches → (64, 240, 384)`

4. **Predictor:** `(64, 324, 384) + 240 query tokens → (64, 240, 384) predictions`

5. **Loss:**
   ```python
   mse = F.mse_loss(predicted, target.detach())
   # Default: VICReg on predictor output (gradient flows).
   reg = vicreg_loss(predicted.mean(dim=1))
   loss = mse + reg
   ```

6. **Backward + EMA update:** gradients flow through predictor and context
   encoder only. Target encoder gets its slow EMA nudge.

7. **Repeat** for 4 accumulation steps, then the optimiser steps.

At no point are any labels, genre tags or human annotations used. The only
signal is "these two tracks appeared consecutively in someone's playlist."

---

## 9. Inference — extracting track embeddings

Once training is complete, the predictor is discarded. Each track is passed
through the context encoder and its 324 patch tokens are **mean-pooled** to one
384-d vector ([jepa/vit.py:98-100](../jepa/vit.py#L98-L100)):

```python
def get_embedding(self, x):
    return self.forward(x).mean(dim=1)  # (B, D)
```

Results are saved to `EMBEDDINGS_DIR/embeddings.npy` (default
`embeddings/embeddings.npy`) as a `{track_id: 384-d vector}` dict.

### Why CLS is discarded rather than used for the embedding

The CLS token participates fully in all 12 attention blocks and acts as a useful
global information aggregator. It is stripped from the encoder output because the
JEPA predictor queries **specific spatial positions** using `target_pos_embed` —
a table of one learned 192-d vector per patch position (324 entries). The CLS
token has no position; there is no meaningful target to compare against if you
try to predict it. Keeping CLS in the output would require the predictor to
handle a position-less slot with no counterpart in the target spectrogram.

Mean pooling of the 324 patch tokens is the natural substitute and works as well
as CLS in practice when CLS has not been explicitly trained as a classification
head (as in BERT).

> **Note:** the target encoder typically produces slightly better embeddings
> because EMA smoothing acts as ensembling over many past model states. Switching
> `eval/embed_tracks.py` to use `model.target_encoder` instead of `model.encoder` is
> a one-line change worth trying.

---

## 10. The playlist head

Once the JEPA encoder is frozen and embeddings are extracted to
`EMBEDDINGS_DIR/embeddings.npy`,
a lightweight **playlist head** can be trained on top to generate playlists
directly from the embedding space. Code lives in `jepa/playlist_head.py` and
`train_head.py`.

### Two tasks

**Continuation** — given a seed history of tracks, predict the next track's
embedding. **Infill** — given a left anchor and a right anchor, predict a
missing track that sits between them. The task is set in
`configs/head_continuation.yaml` or `configs/head_infil.yaml` and controls how
training examples are constructed and how the context vector is built.

### Context representation

Both tasks encode context as a **1152-d vector** (3 × 384d):

```
Continuation:  [last embedding | mean of history | drift (last − first)]
Infill:        [left embedding | right embedding | linear interpolation at alpha]
```

`mean` encodes the current musical zone; `drift` encodes the direction the playlist is moving through it.
`alpha = (target_idx − left_idx) / (right_idx − left_idx)` encodes where in
the gap the missing track sits.

```
CONTINUATION HEAD

  history: [track_1, ..., track_k]   (k ≤ max_history)
                      │
            look up JEPA embeddings
                      │
          ┌───────────┴───────────────┐
          │  [last | mean | drift]    │  (1152d = 3 × 384d)
          └───────────┬───────────────┘
                      │
          ┌───────────┴───────────────┐
          │    PlaylistHead MLP       │
          │  LayerNorm                │
          │  1152 → 1024 → GELU       │
          │  1024 → 1024 → GELU       │
          │  1024 → 384               │
          └───────────┬───────────────┘
                      │  + residual (last embedding)
                 L2 normalise
                      │
          predicted next-track embedding (384d)
                      │
          cosine search → top-k → sample


INFILL HEAD

  left track              right track
       │                       │
    embed                   embed      (alpha = position fraction)
       │                       │
       └──────────┬────────────┘
                  │
          ┌───────┴───────────────────┐
          │  [left | right | interp]  │  (1152d = 3 × 384d)
          └───────┬───────────────────┘
                  │
          ┌───────┴───────────────────┐
          │    PlaylistHead MLP       │  (same architecture, separate weights)
          └───────┬───────────────────┘
                  │  + residual (interpolation vector)
             L2 normalise
                  │
          predicted missing-track embedding (384d)
```

### Residual connection

The MLP predicts a **delta**, not an absolute target. The residual is the last
track embedding (continuation) or the interpolation vector (infill), set by
`residual_source` in the config. For infill, the head learns "how far and in
what direction to deviate from naive linear interpolation between the anchors" —
a much easier task than predicting the target from scratch.

### Training

Loss is **InfoNCE**: the predicted embedding must be closer to the true
next/missing track than to all other tracks in the batch. Temperature 0.07.
Optimiser: AdamW with cosine LR decay. Head configs currently allow up to 100
epochs and use validation-loss early stopping (`early_stopping_patience` and
`early_stopping_min_delta`) so longer runs can stop once the head starts
overfitting.

Only playlists with **10–30 tracks** are used for head training
(`min_playlist_len` / `max_playlist_len` in the head configs). The active head
configs are selected by the Makefile knobs documented in the README
configuration table. Make sure `make embed` has covered the same catalogue the
head config points at; otherwise the loader drops missing track IDs and trains
on chopped-up playlist fragments.

Short playlists lack context; very long playlists tend to be grab-bags with weak
track-to-track coherence — noisier training signal than the JEPA encoder sees,
because InfoNCE false positives (a "next track" that doesn't really follow
musically) actively train the head in the wrong direction.

Make targets:

```
make train-head-cont
make train-head-infil
```

Make targets read/write checkpoints through the configured checkpoint dir.

`train_head.py` is plain PyTorch. If multiple CUDA GPUs are visible it wraps the
head with `torch.nn.DataParallel`; the saved checkpoints are unwrapped so
`load_head` can read them normally.

### Playlist generation (`eval/generate_playlist.py`)

There are two head-backed generation modes:

```
make playlist SEEDS="TRACK_ID [TRACK_ID ...]"
make journey JOURNEY="START_ID END_ID [WAYPOINT_ID ...]"
```

`make playlist` uses the configured continuation head and `--seeds` to keep
adding next tracks. `make journey` uses the configured infill head and
`--journey` to fill between waypoint IDs. Both targets write HTML under the
configured output dir with each track ID, artist/title, per-track audio
controls, and a top-level "Play all previews" button that chains available
30-second previews.

`HEAD_WEIGHT` (0–1, default 1) controls how much the head's prediction is
trusted. At 0, playlist continuation averages recent embeddings and journey
fill linearly interpolates — equivalent to a head-free baseline. The MP3ToVec
audio baseline is available by passing `--mp3tovec_model_dir` directly to
`eval/generate_playlist.py`; it uses Deej-AI's `spotifytovec.p` vectors and
linearly interpolates or averages embeddings without any head.

Track IDs can be found locally with:

```
make search QUERY="artist or title"
```

Example galleries can be regenerated with:

```
make examples
```

This writes head-based playlist and journey pages under the configured output
dir, plus matching raw-JEPA baselines with an `_embeddings.html` suffix and
Deej-AI MP3ToVec baselines with a `_mp3tovec.html` suffix. It also writes an
index page linking to all generated pages.

### Journey generation (`eval/generate_playlist.py`)

Given two or more waypoint track IDs, the infill head fills the gaps using
**binary-search gap-filling**:

1. Place waypoints at the ends of a slot array.
2. Repeatedly find the largest unfilled gap.
3. Call the infill head at the midpoint with the appropriate alpha.
4. Snap to the nearest catalogue track (excluding already-used tracks).
5. Repeat until all slots are filled.

This distributes tracks evenly across the journey rather than front-loading
them, and builds each local decision from its actual neighbours rather than a
fixed global direction.

A **noise** parameter controls stochasticity: `noise=0` is greedy; higher
values sample softly from top-k candidates via a temperature-scaled softmax,
introducing variety without losing coherence.

---

## 11. Future directions

### Using the full 30 seconds

Currently only the first 5s of each 30s preview is used, giving one embedding
per track. All options below first require extending `data/make_spectrograms.py`
to write six 96×216 PNGs per track (one per non-overlapping 5s window) rather
than a single first-5s PNG.

**Option A — multi-window average inference (no architecture change).**
At inference, extract embeddings for all 6 windows and mean-pool them. Captures
more of the track at moderate extra cost; does not weight distinctive moments
over generic ones.

**Option B — TF-IDF weighted aggregation.** Extension of A. Apply TF-IDF
weighting across the 6 window embeddings per track:

- **TF (term frequency):** within a track, count how many of its 6 windows are
  within epsilon cosine distance of each other. Recurring windows dominate the
  track's sound and are weighted up.
- **IDF (inverse document frequency):** across all tracks, count how many tracks
  contain a similar window vector. Generic vectors (common chord patterns,
  silence) appear in many tracks and are weighted down.

The final embedding is the TF-IDF weighted sum of the 6 window vectors. The
`epsilon` threshold becomes a knob: large epsilon → more averaging → global/genre
character; small epsilon → finer discrimination → local texture and feel. A
reference implementation is in `train/calc_tfidf.py` in the Deej-AI repo.

**Option C — hierarchical encoder.** Encode each 5s window to a 384-d vector,
then run a small second transformer over the 6 resulting vectors to integrate
temporal structure across the full 30s. More expressive than A/B; requires a
new training objective.

**Option D — full 30s as one sequence.** All 6 windows as one 1944-token
sequence. Maximally expressive but self-attention is O(n²): 36× more expensive
than 324 tokens, likely prohibitive without efficient attention variants.

**Recommendation:** A is the cheapest entry point. B adds meaningful signal with
no architecture change. C is the right long-term answer if temporal track
structure matters.

### Creativity knob (collaborative vs audio signal)

Deej-AI's creativity knob interpolates between a **collaborative embedding**
(Word2Vec on playlist track IDs — "fans of X also listen to Y") and an **audio
embedding** (CNN on spectrograms). High collaborative weight → genre-accurate
global recommendations; high audio weight → texture/feel local recommendations.

The JEPA embedding is audio-based with playlist co-occurrence baked into the
training signal, but there is no separate collaborative embedding. To replicate
the knob: train a lightweight Track2Vec (Word2Vec on playlist sequences — fast
with gensim, data already available in `data/playlists_dedup.csv`), then blend
rankings from Track2Vec similarity and JEPA similarity with a weight α at
retrieval time. α=0 → pure audio feel; α=1 → pure collaborative genre signal.
