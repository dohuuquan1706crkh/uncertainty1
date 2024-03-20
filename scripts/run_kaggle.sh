# python src/train_frozen_model.py -t configs/tmpl.yaml -e configs/gausscap.yaml -s 1 --exp-name gausscap_dice_kaggle
# python src/train_reconstruction_net.py -t configs/tmpl.yaml -e configs/gausscap.yaml -s 1 --exp-name gausscap_dice_kaggle
# python src/test_reconstruction_net.py -t configs/tmpl.yaml -e configs/gausscap.yaml -s 1 --exp-name gausscap_dice_kaggle

# python src/train_frozen_model.py -t configs/tmpl.yaml -e configs/gausscap_enet.yaml -s 2
# # python src/train_reconstruction_net.py -t configs/tmpl.yaml -e configs/gausscap_enet.yaml -s 2
# python src/test_reconstruction_net.py -t configs/tmpl.yaml -e configs/gausscap_enet.yaml -s 2

# python src/train_frozen_model.py -t configs/tmpl.yaml -e configs/kaggle.yaml -s 1 --exp-name concat
python src/train_reconstruction_net.py -t configs/tmpl.yaml -e configs/kaggle.yaml -s 1 --exp-name gausscap
python src/test_reconstruction_net.py -t configs/tmpl.yaml -e configs/kaggle.yaml -s 1 --exp-name gausscap