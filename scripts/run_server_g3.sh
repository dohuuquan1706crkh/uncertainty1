export CUDA_VISIBLE_DEVICES=2
python src/train_frozen_model.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 2 -g 1 --exp-name spatial_att
python src/train_reconstruction_net.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 2 -g 1 --exp-name spatial_att
python src/test_reconstruction_net.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 2 --exp-name spatial_att

python src/train_frozen_model.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 3 -g 1 --exp-name spatial_att
python src/train_reconstruction_net.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 3 -g 1 --exp-name spatial_att
python src/test_reconstruction_net.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 3 --exp-name spatial_att

python src/train_frozen_model.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 4 -g 1 --exp-name spatial_att
python src/train_reconstruction_net.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 4 -g 1 --exp-name spatial_att
python src/test_reconstruction_net.py -t configs/tmpl.yaml -e configs/server_gausscapcontext.yaml -s 4 --exp-name spatial_att