
## Anything related to preprocessing for PALM-Net format
### Preprocess in-the-wile images for PALM-Net personalization

Install `hamer` as per their instructions (or see installation guide [here](https://github.com/zc-alexfan/hold/blob/master/docs/setup.md)), then put images under `hamer/images`. 

```bash
conda activate hamer
python demo.py --seq_name blah --batch_size=2  --full_frame --body_detector regnety # this will dump itw_hamer.npy
```

Unpack Hamer results: 

```bash
python unpack_hamer.py --hpe_path ./itw_hamer.npy
```

MANO registration:

```bash
python register_mano.py --save_mesh --use_beta_loss
```

Install `lang-segment-anything` following their instructions. Extract hand masks:

```bash
conda activate lang-sam
cd lang-segment-anything
python demo.py
cd ..
```

Build datasets:

```bash
python build_dataset.py --rebuild --build_all
```


### Convert InterHand2.6M data to PALM-Net format

Download InterHand2.6M annotations and put them in this structure:

```bash
./interhand/annotations/ # `annotations` is the downloaded folder from IH
```

Create InterHand2.6M folders: 

```bash
python interhand2PALM-Net.py
```

Extract hand masks:

```bash
conda activate lang-sam
cd lang-segment-anything
python demo.py ## Change the glob path inside and use the keyword "foreground."
cd ..
```
