## Evaluation on InterHand2.6M (IH)

Here we show how we personalize on IH and render with seen IH environment and novel poses for evaluation. 

First, we need to prepare IH sequences in PALM-Net format. Follow the instructions at [`generator/README.md`](../generator/README.md). Then put those IH sequences under `./load/personalize/*`. 

To personalize on a single image from an IH sequence (`ih_c0_ROM03_RT_No_Occlusion_cam400262` for example) using our prior `0301-1442-124.zip`, you can run:

```bash
./run_personalize.sh ih_c0_ROM03_RT_No_Occlusion_cam400262 exp/0301-1442-124/ckpt/0/epoch=0-step=24000.ckpt
```

After personalizing on all images, you use `./run_render_interhand_all.sh` to render all sequences. Update the `shape_code_basis_path`, `appearance_code_basis_path`, and the new checkpoint paths accordingly. 

```bash
./run_render_interhand_all.sh
```


After rendering the sequences, you will find the sequence name and experiement ID at `./exp_ids.txt`. 

To evaluate, run:

```bash
python eval_seen_env.py
```

Note that, the evaluation script takes in the entries at `./exp_ids.txt`, so you need to remove any entry in `./exp_ids.txt` that are not related to evaluation (for example, failed rendering checkpoints). 


