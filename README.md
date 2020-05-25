# GAN
### Установка пакетов: 
pip requirements.txt
### Препроцессинг данных: 

python preporocessing.py \
    --dataset_name=spine_segmentation \
    --dataset_dir=./datasets/spine_segmentation/spine_segmentation_1/test/ \
    --output_name=spine_segmentation_train_1_fold \
    --output_dir=./datasets/tfrecords_spine_segmentation_with_superpixels
    
### Тренировка:
python train_gan.py

Можно также задать директорию для выода результатов --train_dir, директорию с датасетом--dataset_dir,
--batch_size, --optimizer, --num_epochs
