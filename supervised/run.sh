#Huffpost 1-shot MAML
python main.py --datasource=huffpost --meta_lr=2e-5 --update_lr=0.01 --meta_batch_size=1 --update_batch_size=1 --update_batch_size_eval=5 --num_classes=5 --logdir=xxx --num_updates=5 --num_updates_test=5 --use_kg=1 --metatrain_iterations=10000
python main.py --datasource=huffpost --meta_lr=2e-5 --update_lr=0.01 --meta_batch_size=1 --update_batch_size=1 --update_batch_size_eval=5 --num_classes=5 --logdir=xxx --num_updates=5 --num_updates_test=5 --use_kg=1 --metatrain_iterations=10000 --train=0

#Huffpost 5-shot MAML
python main.py --datasource=huffpost --meta_lr=2e-5 --update_lr=0.01 --meta_batch_size=1 --update_batch_size=5 --update_batch_size_eval=5 --num_classes=5 --logdir=xxx --num_updates=5 --num_updates_test=5 --use_kg=1 --metatrain_iterations=10000
python main.py --datasource=huffpost --meta_lr=2e-5 --update_lr=0.01 --meta_batch_size=1 --update_batch_size=5 --update_batch_size_eval=5 --num_classes=5 --logdir=xxx --num_updates=5 --num_updates_test=5 --use_kg=1 --metatrain_iterations=10000 --train=0

#Huffpost 1-shot Protonet
python main.py --datasource=huffpost --meta_lr=2e-5 --meta_batch_size=1 --update_batch_size=1 --update_batch_size_eval=5 --num_classes=5 --logdir=xxx --use_kg=1 --metatrain_iterations=5000
python main.py --datasource=huffpost --meta_lr=2e-5 --meta_batch_size=1 --update_batch_size=1 --update_batch_size_eval=5 --num_classes=5 --logdir=xxx --use_kg=1 --metatrain_iterations=5000 --train=0

#Huffpost 5-shot Protonet
python main.py --datasource=huffpost --meta_lr=2e-5 --meta_batch_size=1 --update_batch_size=5 --update_batch_size_eval=5 --num_classes=5 --logdir=xxx --use_kg=1 --metatrain_iterations=5000
python main.py --datasource=huffpost --meta_lr=2e-5 --meta_batch_size=1 --update_batch_size=5 --update_batch_size_eval=5 --num_classes=5 --logdir=xxx --use_kg=1 --metatrain_iterations=5000 --train=0

# for amazon review, change huffpost to amazonreview