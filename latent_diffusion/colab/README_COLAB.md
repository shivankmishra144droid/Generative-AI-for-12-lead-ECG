# Training the bigger model (C-M) on Google Colab

This folder has two files:

- **`ecg_colab_bundle.zip`** (481 MB): the code plus the training data. It contains the train split only, with no
  validation and no test data.
- **`ECG_colab_CM.ipynb`**: the notebook you run on Colab.

The run trains C-M, a 16 × 128 latent diffusion model with 5.9 M parameters (the current model has 0.8 M). Everything
else is the same as the current model. It was pre-registered as amendment 4 in `audits/v2/amendment1.md`.

## One-time setup (about 10 min)

1. **Upload the bundle to Google Drive.**
   - Open https://drive.google.com.
   - Drag `ecg_colab_bundle.zip` into **My Drive**, at the top level, not into a folder.
   - Wait for the upload to finish.
2. **Open the notebook in Colab.**
   - Go to https://colab.research.google.com.
   - Choose File → Upload notebook → `ECG_colab_CM.ipynb`.
3. **Pick a GPU.**
   - Choose Runtime → Change runtime type → Hardware accelerator: **T4 GPU**. The free tier offers this.
   - On Colab Pro, L4 or A100 is faster.
   - Click Save.

## Running it

Run the cells in order. Click a cell, then press Shift+Enter.

| cell | what it does | what you should see |
|---|---|---|
| 1 | Shows the GPU | A table naming the GPU, for example "Tesla T4". |
| 2 | Connects Google Drive | A pop-up: allow access to your own Drive. |
| 3 | Unpacks the bundle into `My Drive/ECG_colab` | Takes about 2 min the first time and is skipped after that. It ends with a folder list. |
| 4 | Installs libraries | `pykeops OK` (fast) or `pykeops not usable` (slower but fine). **Do not import torch or pykeops in the notebook itself**: the GPU lock would then wait on the notebook forever. |
| 5 | **Speed test** (1,000 steps) | A final line like `cm_profile 1000 0.87 [...] 3.1 min`. **Send me that minute number.** |
| 6 | **Main training** (100k steps) | One line every 1,000 steps. This runs for hours. |
| 7 | Packs the results | Creates `My Drive/cm_results.zip`. |

**How long cell 6 takes:** the minutes from cell 5 × 100 = total minutes. For example, 3 min → 300 min ≈ 5 h. If that
comes to more than 12 h, lower `STEPS` in cell 6 so that it fits in 12 h; for example, 6 min per 1,000 steps →
`STEPS = 120000 // 6 = 20000`. Change it before the first run of cell 6 and don't change it afterwards.

## If Colab disconnects (this is normal on the free tier)

A checkpoint is saved to Drive every 1,000 steps, so you lose at most about 1,000 steps of work. After a disconnect:

1. Runtime → Connect, and make sure the runtime is still a T4 GPU.
2. Run cells **1, 2, 3 and 4** again. Cell 3 skips the unpacking.
3. Run cell **6** again with the **same** `STEPS`. It prints the next step number and carries on.

Free-tier tips:
- Keep the browser tab open and the laptop awake.
- Sessions usually end after a few hours, or after about 90 min with no activity on the page.

## When it finishes

1. Run cell **7**.
2. Download `cm_results.zip` and `cm_gpu.txt` from My Drive.
3. Put them in `latent_diffusion/colab/` on the laptop and tell me. I'll unpack them and run the dev comparison against
   the current model, as pre-registered.

## Notes

- **Two runs:** don't run cell 6 in two Colab tabs at once. Both would write the same checkpoint.
- **What goes to Google:** the bundle uploads PTB-XL training ECGs. PTB-XL is a public dataset (PhysioNet, CC-BY 4.0),
  and the validation and test splits are left out on purpose.
- **To stop early:** use Runtime → Interrupt execution. The last checkpoint is kept, and running cell 7 still packs
  what exists.
