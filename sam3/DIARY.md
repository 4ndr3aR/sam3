# Debugging Diary: "monument" Prompt Not Working

## Date: 2026-04-17

---

## Problem Statement

Fine-tuned SAM3 checkpoints segment a famous Indonesian fortress perfectly with prompt **"Rocky coastal temple"** but completely fail with prompt **"monument"** - zero pixels segmented.

---

## Investigation Timeline

### Session 1: Initial Diagnosis

**Hypothesis 1:** Checkpoint loading issue causing wrong weights
- **Investigation:** Examined `sam3/model/sam3_image.py` `_load_checkpoint` method
- **Finding:** Checkpoint filtering skips keys containing "detector" - training checkpoints have keys like `backbone.vision_backbone.*` without "detector." prefix
- **Result:** ~1134 missing keys when loading
- **Fix:** Modified `_load_checkpoint` to detect checkpoint format (HF vs training) and apply appropriate key filtering
- **Verification:** User confirmed 0 missing keys after fix

**Hypothesis 2:** Mask decoder needs to be unfrozen
- **Investigation:** User tried unfreezing decoder - no improvement
- **Result:** ❌ Not the issue

**Hypothesis 3:** Language backbone embeddings not adapting to new prompt
- **Investigation:** Examined model architecture and training setup
- **Finding:** Config had `freeze_backbone: true` which freezes BOTH vision AND language backbones
- **Fix:** Added selective backbone freezing parameters to `model_builder.py`:
  ```python
  freeze_vision_backbone: bool = True
  freeze_language_backbone: bool = True
  unfreeze_last_n_text_layers: int = 0
  ```
- **Added function:** `_freeze_layers()` for layer-wise freezing of text transformer
- **Result:** ✅ User can now unfreeze last N text layers efficiently (~58M params vs 347M)

---

### Session 2: Segmentation Meter Implementation

**Problem:** User reported validation errors and zero metrics

**Investigation:**
1. Wrong attribute access: `model.backbone.text` → `model.backbone.language_backbone`
2. Wrong attribute: `language_backbone.transformer` → `language_backbone.encoder.transformer`
3. Batch access: `batch.get("find_masks")` failed - `batch` is `BatchedDatapoint`, not dict
   - **Correct:** `getattr(batch, "find_targets")[0].segments`

**Created:** `eval/segmentation_meter.py`
- Computes IoU, Dice, precision, recall, F1, pixel accuracy
- Uses bipartite matching to match predictions to ground truth
- Handles both `SegmentationMeter` (instance-level) and `SimpleSegmentationMeter` (semantic)

---

### Session 3: ROOT CAUSE DISCOVERY 🔍

**Critical Finding in `train/data/coco_json_loaders.py`:**

```python
# Line 245-248
query["query_text"] = (
    self._cat_idx_to_text[cat_id]  # Uses category.name from COCO categories list
    if self.prompts is None
    else self.prompts[cat_id]
)
```

**The Problem:**
- Training uses `category.name` from COCO categories list as prompts
- Dataset categories: "Rocky coastal temple", "Coastal temple structure", "Temple on rocks", etc. (8 total)
- **ALL 5892 annotations have `noun_phrase: "monument"`**
- Model was NEVER trained on "monument" prompt!

**Why "rock" works but "monument" fails:**
| Prompt | Why It Works/Fails |
|--------|-------------------|
| "Rocky coastal temple" | ✅ Exact category name - model trained on this |
| "rock" | ✅ Generic word from pretrained language embeddings |
| "monument" | ❌ Never used as training prompt - model has no concept |

---

## Implementation: noun_phrase Support

### Files Modified

**1. `train/data/coco_json_loaders.py`**

Added to `COCO_FROM_JSON.__init__`:
```python
def __init__(
    self,
    annotation_file,
    prompts=None,
    include_negatives=True,
    category_chunk_size=None,
    use_noun_phrase_as_prompt=False,  # NEW
    noun_phrase_mode="unique",  # NEW
):
```

Added method `_build_noun_phrase_mapping()`:
- Mode "unique": First unique noun_phrase per category
- Mode "all_monument": Force all queries to "monument"
- Mode "per_annotation": Different query per annotation

Updated `loadQueriesAndAnnotationsFromDatapoint()`:
```python
# Priority: custom prompts > noun_phrase > category name
if self.prompts is not None:
    query["query_text"] = self.prompts[cat_id]
elif self.use_noun_phrase_as_prompt and self._cat_id_to_noun_phrase:
    query["query_text"] = self._cat_id_to_noun_phrase.get(cat_id, self._cat_idx_to_text[cat_id])
else:
    query["query_text"] = self._cat_idx_to_text[cat_id]
```

**2. `train/data/sam3_image_dataset.py`**

Added `coco_json_loader_kwargs` parameter to:
- `CustomCocoDetectionAPI.__init__()`
- `Sam3ImageDataset.__init__()`
- Passed through to `coco_json_loader()` instantiation

**3. `train/configs/roboflow_v100/statuario-v2-tiny-1GPU-seg.yaml`**

Added under `coco_train`:
```yaml
coco_json_loader_kwargs:
  use_noun_phrase_as_prompt: true  # Use noun_phrase from annotations
  noun_phrase_mode: "all_monument"  # Force all to "monument"
```

Passed to both train and val datasets:
```yaml
dataset:
  coco_json_loader_kwargs: ${coco_train.coco_json_loader_kwargs}
```

---

## Summary of All Changes

| File | Change | Purpose |
|------|--------|---------|
| `model/sam3_image.py` | Checkpoint format detection | Fix training checkpoint loading |
| `model_builder.py` | Selective backbone freezing | Efficient language adaptation |
| `eval/segmentation_meter.py` | NEW FILE | IoU/Dice metrics for segmentation |
| `train/data/coco_json_loaders.py` | noun_phrase support | **SOLVE "monument" problem** |
| `train/data/sam3_image_dataset.py` | Pass kwargs to loader | Enable noun_phrase config |
| Config YAML | noun_phrase settings | Enable training with "monument" |

---

## How to Test Tomorrow

### Step 1: Verify noun_phrase is being used
```bash
# Add debug print in coco_json_loaders.py line ~250
print(f"DEBUG: cat_id={cat_id}, query_text={query['query_text']}")
```

Run training and check logs - should see "monument" as query_text.

### Step 2: Train with noun_phrase enabled
```yaml
# In config:
coco_json_loader_kwargs:
  use_noun_phrase_as_prompt: true
  noun_phrase_mode: "all_monument"
```

### Step 3: Test inference
```bash
python sam3-predict-image-or-video.py \
  --model_path checkpoint_N.pt \
  --image damaged-statues/damaged-statues-Image_5_00001_.png \
  --text 'monument'
```

Expected: Should now segment the Indonesian fortress!

---

## Alternative Configurations to Test

### Option A: Train with "monument" only (RECOMMENDED)
```yaml
coco_json_loader_kwargs:
  use_noun_phrase_as_prompt: true
  noun_phrase_mode: "all_monument"
```

### Option B: Train with category names (current behavior)
```yaml
coco_json_loader_kwargs:
  use_noun_phrase_as_prompt: false
```

### Option C: First unique noun_phrase per category
```yaml
coco_json_loader_kwargs:
  use_noun_phrase_as_prompt: true
  noun_phrase_mode: "unique"
```

---

## Key Insight

> **The model learns the VISUAL concept during training, but associates it with the TEXTUAL prompt used.** If you train with "Rocky coastal temple" as the prompt, the model learns to segment that visual pattern when it sees the text embedding for "Rocky coastal temple". It does NOT automatically generalize to "monument" because that's a completely different text embedding it was never trained on.

**Solution:** Train with "monument" as the prompt so the model learns the association between the "monument" text embedding and the visual pattern of the fortress.

---

# Session 4: noun_phrase Fix Didn't Work - More Debug Needed

## Date: 2026-04-20

### User Report

The noun_phrase fix **did not work**:

1. Trained **18 epochs** with `use_noun_phrase_as_prompt: true` (`noun_phrase_mode: "all_monument"`)
2. Trained a few more epochs with same set to **false** after **renaming all categories** in annotations to "monument"
3. **Neither approach worked**

### Results

| Prompt | Result |
|-- ---- |-- --- -- ----|
| "monument" | ❌ **Zero masks** produced by ANY checkpoint |
| "Rocky coastal temple" | ✅ Works until checkpoint 5, then ❌ zero masks |
| "rock" | ✅ Works but with **decreasing masks** over training |

### Analysis

This is concerning:
1. Training with "monument" as prompt produced **zero masks** at inference - suggests the model isn't learning anything
2. Performance **degrades over time** with "rocky coastal temple" and "rock" prompts
3. Even renaming categories didn't help

### Next Steps: Comprehensive Debug Output

Added debug output to trace the entire data pipeline:

**1. `train/data/coco_json_loaders.py`**
- Debug print in `_build_noun_phrase_mapping()` showing what noun_phrases are found in annotations
- Debug print when assigning `query_text` showing source (CUSTOM/NOUN_PHRASE/CATEGORY)

**2. `train/data/collator.py`**
- New function `debug_print_batch_sample()` prints:
  - Query text
  - Number of objects per query
  - Bounding boxes
  - Mask areas (not full masks)
- Batch summary showing unique query texts and total objects
- Limited to first 3 samples per batch

### What to Look For in Debug Output

1. **Are noun_phrases actually being read?** Check if annotations contain `noun_phrase` field
2. **Is query_text set to "monument"?** Verify the correct prompt is being used
3. **Are segmentation masks being loaded?** Check if masks have non-zero area
4. **Are bounding boxes valid?** Check if boxes have positive area

### Possible Issues

1. **noun_phrase field missing:** Annotations might not actually have `noun_phrase` field
2. **Masks not loading:** Segmentation might be disabled or broken
3. **Learning rate too high/low:** Model might be diverging or not learning
4. **Loss not being computed:** Mask loss might not be active
5. **Gradient vanishing:** Language backbone might not be receiving gradients

### Verification Commands

After running training, check logs for:
```
DEBUG COCO_LOADER: Building noun_phrase mapping
DEBUG QUERY[idx=0]: cat_id=X, source=..., query_text='...'
DEBUG COLLECT FN: batch_size=X, unique_queries=X, unique_query_texts: [...]
DEBUG BATCH SAMPLE [0]: X queries, X images
  Query 0: query_text='...'
```
