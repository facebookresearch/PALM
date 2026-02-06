from PIL import Image
import json
from smplx import MANO
import torch
import numpy as np
import json
import os
import os.path as op

mano_layer = MANO("../code/data/body_models",
                  is_rhand=True, flat_hand_mean=False, use_pca=False)


def select_uniformly(data, num_elements):
    # Calculate the step size
    step = len(data) // num_elements
    
    # Select every k-th element
    selected_elements = data[::step]
    
    # If the number of selected elements is more than required, trim the list
    return selected_elements[:num_elements]
    
def create_dataset(is_train, seq_name, capture_id, cam_id, image_data, capture_data, cameras_all):
    if is_train:
        sid = '_train'
    else:
        sid = '_val'

    K_list = []
    global_orient_list = []
    hand_pose_list = []
    transl_list = []
    betas_list = []
    world2cam_list = []
    basenames = []
    num_images = 1
    image_data_selected = [idata for idata in image_data 
                        if seq_name in idata['file_name'] and cam_id in idata['file_name'] and str(idata['frame_idx']) in capture_data
                        and f'Capture{capture_id}/' in idata['file_name']]

    assert len(image_data_selected) > 0
    if is_train:
        image_data_selected = image_data_selected[:1]
    else:
        # image_data_selected = image_data_selected[1:]
        image_data_selected = select_uniformly(image_data_selected, 20)
    for tidx, mydata in enumerate(image_data_selected):
        # print(idx)
        basename = mydata['file_name']
        camera_id = mydata['camera']
        width = mydata['width']
        height = mydata['height']
        frame_idx = mydata['frame_idx']
        print(sid, basename)
            
        im = Image.open("./interhand/InterHand2.6M_5fps_batch1/images/test/" + basename)
        im_p = f'./interhand_seqs/ih_c{capture_id}_{seq_name}_{cam_id}/folders/{sid}/images/MCU_02/{num_images:04}.jpg'
        im_p = im_p.replace('.jpg', '.png')
        os.makedirs(op.dirname(im_p), exist_ok=True)
        im.save(im_p)
        basenames.append(basename)


        campos = np.array(cameras_all[capture_id]['campos'][camera_id]).reshape(-1)/1000
        camrot = np.array(cameras_all[capture_id]['camrot'][camera_id]).reshape(-1, 3, 3)
        focal = np.array(cameras_all[capture_id]['focal'][camera_id]).reshape(-1)
        princpt = np.array(cameras_all[capture_id]['princpt'][camera_id]).reshape(-1)
        # Assuming camrot is already a 3x3 rotation matrix
        rotation_matrix = camrot
        
        # Create the 4x4 world2cam matrix
        world2cam = np.eye(4)
        world2cam[:3, :3] = rotation_matrix
        world2cam[:3, 3] = -np.dot(rotation_matrix, campos)  # Translation part
        # Assuming focal is a scalar or has two components for fx and fy
        # fx = fy = focal[0]  # If focal is a single value
        fx, fy = focal  # If focal has two components
        
        cx, cy = princpt
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        pose_dict = capture_data[str(frame_idx)]['right']
        for key, val in pose_dict.items():
            pose_dict[key] = np.array(val).reshape(1, -1)
        K_list.append(K)
        global_orient_list.append(pose_dict['pose'][:, :3])
        hand_pose_list.append(pose_dict['pose'][:, 3:])
        transl_list.append(pose_dict['trans'])
        betas_list.append(pose_dict['shape'])
        world2cam_list.append(world2cam)
        num_images += 1
        
    K = np.stack(K_list, axis=0)[0]
    global_orient_list = np.concatenate(global_orient_list, axis=0)
    hand_pose_list = np.concatenate(hand_pose_list, axis=0)
    transl_list = np.concatenate(transl_list, axis=0)
    betas_list = np.concatenate(betas_list, axis=0)
    world2cam_list = np.stack(world2cam_list, axis=0)

    out_pose_p = f'./interhand_seqs/ih_c{capture_id}_{seq_name}_{cam_id}/folders/{sid}/poses.npy'
    os.makedirs(op.dirname(out_pose_p), exist_ok=True)
    global_orient_list, transl_list = mano_layer.transform_mano_params(
        torch.FloatTensor(betas_list),
        torch.FloatTensor(global_orient_list),
        torch.FloatTensor(transl_list), 
        torch.FloatTensor(world2cam_list[0])
    )

    out = {}
    out['betas'] = betas_list
    out['global_orient'] = global_orient_list.numpy()
    out['hand_pose'] = hand_pose_list
    out['transl'] = transl_list.numpy()
    out['betas'] = betas_list

    np.save(out_pose_p, out)
    cam = {}
    cam['K'] = K
    cam['Rt'] = np.eye(4)
    cam['width'] = width
    cam['height'] = height

    cam_p = f'./interhand_seqs/ih_c{capture_id}_{seq_name}_{cam_id}/folders/{sid}/cameras.npy'
    np.save(cam_p, {
        'cameras': {'MCU_02': cam}
    })
    print('Created dataset at: ', f'./interhand_seqs/ih_c{capture_id}_{seq_name}_{cam_id}/folders/{sid}')

def main():
    with open('./interhand/annotations/test/InterHand2.6M_test_MANO_NeuralAnnot.json', 'r') as f:
        data = json.load(f)

    with open('./interhand/annotations/test/InterHand2.6M_test_camera.json', 'r') as f:
        cameras_all = json.load(f)

    with open('./interhand/annotations/test/InterHand2.6M_test_data.json', 'r') as f:
        image_data = json.load(f)['images']


    capture_id = '0'
    seq_name = 'ROM03_RT_No_Occlusion'
    cam_id = 'cam400262'
    capture_data = data[capture_id]
    for is_train in [True, False]:
        create_dataset(is_train, seq_name, capture_id, cam_id, image_data, capture_data, cameras_all)


    capture_id = '0'
    seq_name = 'ROM04_RT_Occlusion'
    cam_id = 'cam400275'
    capture_data = data[capture_id]
    for is_train in [True, False]:
        create_dataset(is_train, seq_name, capture_id, cam_id, image_data, capture_data, cameras_all)

    capture_id = '0'
    seq_name = 'ROM05_RT_Wrist_ROM'
    cam_id = 'cam400270'
    capture_data = data[capture_id]
    for is_train in [True, False]:
        create_dataset(is_train, seq_name, capture_id, cam_id, image_data, capture_data, cameras_all)

    capture_id = '1'
    seq_name = 'ROM03_RT_No_Occlusion'
    cam_id = 'cam400456'
    capture_data = data[capture_id]
    for is_train in [True, False]:
        create_dataset(is_train, seq_name, capture_id, cam_id, image_data, capture_data, cameras_all)

    capture_id = '1'
    seq_name = 'ROM04_RT_Occlusion'
    cam_id = 'cam400266'
    capture_data = data[capture_id]
    for is_train in [True, False]:
        create_dataset(is_train, seq_name, capture_id, cam_id, image_data, capture_data, cameras_all)

    capture_id = '1'
    seq_name = 'ROM05_RT_Wrist_ROM'
    cam_id = 'cam400314'
    capture_data = data[capture_id]
    for is_train in [True, False]:
        create_dataset(is_train, seq_name, capture_id, cam_id, image_data, capture_data, cameras_all)


    ######

    capture_id = '0'
    seq_name = 'ROM03_RT_No_Occlusion'
    cam_id = 'cam400451'
    capture_data = data[capture_id]
    for is_train in [True, False]:
        create_dataset(is_train, seq_name, capture_id, cam_id, image_data, capture_data, cameras_all)


    capture_id = '0'
    seq_name = 'ROM04_RT_Occlusion'
    cam_id = 'cam400418'
    capture_data = data[capture_id]
    for is_train in [True, False]:
        create_dataset(is_train, seq_name, capture_id, cam_id, image_data, capture_data, cameras_all)

    capture_id = '0'
    seq_name = 'ROM05_RT_Wrist_ROM'
    cam_id = 'cam400488'
    capture_data = data[capture_id]
    for is_train in [True, False]:
        create_dataset(is_train, seq_name, capture_id, cam_id, image_data, capture_data, cameras_all)

    capture_id = '1'
    seq_name = 'ROM03_RT_No_Occlusion'
    cam_id = 'cam400486'
    capture_data = data[capture_id]
    for is_train in [True, False]:
        create_dataset(is_train, seq_name, capture_id, cam_id, image_data, capture_data, cameras_all)

    capture_id = '1'
    seq_name = 'ROM04_RT_Occlusion'
    cam_id = 'cam400439'
    capture_data = data[capture_id]
    for is_train in [True, False]:
        create_dataset(is_train, seq_name, capture_id, cam_id, image_data, capture_data, cameras_all)

    capture_id = '1'
    seq_name = 'ROM05_RT_Wrist_ROM'
    cam_id = 'cam400469'
    capture_data = data[capture_id]
    for is_train in [True, False]:
        create_dataset(is_train, seq_name, capture_id, cam_id, image_data, capture_data, cameras_all)




if __name__ == "__main__":
    main()
