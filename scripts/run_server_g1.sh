export CUDA_VISIBLE_DEVICES=1
export WORKDIR="/home/taing/workspace/Segmentation-Uncertainty"
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
# python src/train_frozen_model.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 1 -g 1 --exp-name conv
# python src/train_reconstruction.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 1 -g 1 --exp-name conv
# python src/test_reconstruction.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 1 --exp-name conv

# python src/train_frozen_model.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 3 -g 1 --exp-name conv
# python src/train_reconstruction.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 3 -g 1 --exp-name conv
# python src/test_reconstruction.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 3 --exp-name conv

# python src/train_frozen_model.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 4 -g 1 --exp-name conv
# python src/train_reconstruction.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 4 -g 1 --exp-name conv
# python src/test_reconstruction.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 4 --exp-name conv

# python src/train_frozen_model.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 1 -g 1 --exp-name channel_att
# python src/train_reconstruction.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 1 -g 1 --exp-name channel_att
# python src/test_reconstruction.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 1 --exp-name channel_att

# python src/train_frozen_model.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 1 -g 1 --exp-name spatial_att
# python src/train_reconstruction.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 1 -g 1 --exp-name spatial_att
# python src/test_reconstruction.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 1 --exp-name spatial_att

# python src/train_frozen_model.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 1 -g 1 --exp-name conv
# python src/train_reconstruction.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 1 -g 1 --exp-name conv
# python src/test_reconstruction.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 1 --exp-name conv

# python src/train_reconstruction.py -t configs/tmpl.yaml -e configs/server_teacher_student.yaml -s 1 -g 1 --exp-name tch_std
# python src/test_reconstruction.py -t configs/tmpl.yaml -e configs/server_teacher_student.yaml -s 1 --exp-name tch_std

# python src/train_reconstruction.py -t configs/tmpl.yaml -e configs/server_tch_std_context.yaml -s 1 -g 1 --exp-name entrance
# python src/test_reconstruction.py -t configs/tmpl.yaml -e configs/server_tch_std_context.yaml -s 1 --exp-name entrance

# python src/train_reconstruction.py -t configs/tmpl.yaml -e configs/server_tch_std_context.yaml -s 1 -g 1 --exp-name channel_att
# python src/test_reconstruction.py -t configs/tmpl.yaml -e configs/server_tch_std_context.yaml -s 1 --exp-name channel_att

python src/train_reconstruction.py -t configs/tmpl.yaml -r test -e configs/server_gausscapcontext.yaml -s 2 -g 1 --exp-name conv
# python src/test_reconstruction.py -t configs/tmpl.yaml -e configs/server_tch_std_context.yaml -s 1 --exp-name spatial_att