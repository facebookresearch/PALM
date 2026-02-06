seq_name=$1
ckpt_name=$2
val_check_interval=300
batch_size=8
loader_size=8
num_pixels=3
resume_weights_only=True
pose_lr=0.00001
python launch.py dataset=personalize.yaml \
    gpu=\"0\" model.batch_size=$batch_size \
    model.loader_size=$loader_size resume_weights_only=$resume_weights_only \
    personalize=True trainer.val_check_interval=$val_check_interval \
    no_mesh=True checkpoint.save_top_k=0 resume=\"$ckpt_name\" \
    sampler.num_sample=$num_pixels \
    system.optimizer.params.pose_correction.lr=$pose_lr seq_name=$seq_name \
    trainer.max_steps=5200