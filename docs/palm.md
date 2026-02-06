
# Documentation

## Data Download

Register an account [here](https://forms.gle/zbNPFtJDaZP4cQmz7) and then you will recieve a link to download the PALM dataset. After downloading, you will have this folder structure: 

```bash
./PALM/XR20A/0000.zip
./PALM/XR20A/0001.zip
./PALM/XR20A/0002.zip
...
./PALM/XR20A/meshes/0000.zip
./PALM/XR20A/meshes/0001.zip
...
./PALM/XR20A/gt_checksums.json
```

Set environment variable, for example: `export PALM_DIR=/media/zfan/ssd/palm_download/PALM/XR20A`. 

Create symbolic link:

```bash
cd code
ln -s $PALM_DIR PALMDB_highres
```

## PALM Usage

To use PALM, you can use this simple script which gives an example to load the annotations and overlay the results on an image:

```bash
python visualize_PALM.py -c <cam_id> -id <subject_id> -frame <frame_num>
```

- `cam_id` is the id of the camera. All scans have 7 cameras with ids [MCU_01, MCU_02, MCU_03, MCU_04, MCU_05, MCU_06, MCU_07]
- `subject_id` is the id of the subject. There are a total of 262 subjects with ids [0000, 0001,...., 0262]
- `frame_num` Each scan has a varying number of frames in the range [0, 60]

## Joint ordering

We use MANO joint ordering. See [`zc-alexfan/arctic`](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md) for the joint names and ordering. 
