from __future__ import division
import tensorflow.compat.v1 as tf

import numpy as np
import os 
import pydicom as dicom
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from skimage.util import img_as_float
from skimage.segmentation import slic
import os
import sys
import random
import scipy.ndimage

SPINE_LABELS = {
    'none':(0,'background'),
    'vn':(1, 'Normal Vertebral'),
    'vd':(2, 'Vertebral Deformity'),
    'dn':(3, 'Normal Disc'),
    'dm':(4, 'Mild Gegeneration Disc'),
    'ds':(4, 'Severe Degeneration Disc'),
    'fn':(5, 'Neuro Foraminal Normal'),
    'fs':(6, 'Neuro Foraminal Stenosis'),
    'sv':(0, 'Caudal Vertebra')
}

DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'Dicoms/'

RANDOM_SEED = 4242
SAMPLES_PER_FILES = 300

def get_groundtruth_from_xml(xml, dm):
    dm = dicom.dcmread(dm)
    labels = []
    labels_text = []
    instance = []
    coordinates_class = [] 
    coordinates_instance = {} 
    tree = ET.parse(xml)
    root = tree.getroot()
    rows = dm.Rows
    columns = dm.Columns
    shape = [int(rows),int(columns),int(1)]
    masks= np.array([rows,columns])
    for object in root.findall('object'):
        coordinate = []
        if object.find('deleted').text != 1:
            label = object.find('name').text 
            label_int = int(SPINE_LABELS[label][0]) 
            labels.append(label_int)
            labels_text.append(label.encode('ascii'))
            
            instance_label_int = int(object.find('id').text) 
            instance.append(instance_label_int)            
            polygon = object.find('polygon')
            for pt in polygon.findall('pt'):
                x = int(pt.find('x').text)
                y = int(pt.find('y').text)
                coordinate.append((x,y))
            coordinates_class.append(coordinate)
            coordinates_instance[instance_label_int] = coordinate
    return labels, labels_text, instance, shape, coordinates_class, coordinates_instance
                
def groundtruth_to_mask(xml, dm):
    
    labels, labels_text, instance, shape, coordinates_class, coordinates_instance = get_groundtruth_from_xml(xml, dm)
    #print shape
    #draw image first and then using numpy to matrix.
    img_instance = Image.new('L', (shape[0], shape[1]), 0)
    img_class = Image.new('L', (shape[0], shape[1]), 0)
    for i in coordinates_instance:###instance mask
        polygon_instance = coordinates_instance[i]
        ImageDraw.Draw(img_instance).polygon(polygon_instance, outline=0, fill=i)    
    for j,k in enumerate(coordinates_class):
        polygon_class = k
        ImageDraw.Draw(img_class).polygon(polygon_class, outline=0, fill=labels[j])
    mask_instance = np.array(img_instance)
    mask_class = np.array(img_class)
    return mask_instance, mask_class, shape, labels, labels_text, instance

def get_image_superpixels_data_from_dicom(dm):
    dm = dicom.dcmread(dm)
    x = 512
    y = 512
    xscale = x/dm.Rows
    yscale = y/dm.Columns
    image_data = np.array(dm.pixel_array[10])
    #image_data = np.float32(image_data)
    image_data = scipy.ndimage.interpolation.zoom(image_data, [xscale,yscale])
    print(image_data.shape)
    #image = img_as_float(image_data)
    superpixels = slic(image_data, n_segments = 2000, compactness=0.01, max_iter=10)
    return image_data, superpixels
def get_image_data_from_dicom(dm):
    dm = dicom.read_file(dm)
    x = 512
    y = 512
    xscale = x/dm.Rows
    yscale = y/dm.Columns
    image_data = np.array(dm.pixel_array[10])
    #image_data = np.float32(image_data)
    image_data = scipy.ndimage.interpolation.zoom(image_data, [xscale,yscale])
    print(image_data.shape)
    #image = img_as_float(image_data)
    superpixels = slic(image_data, n_segments = 2000, compactness=0.01, max_iter=10)
    return image_data, superpixels
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _convert_to_example(image_data, superpixels, mask_instance, mask_class, shape, class_labels, class_labels_text, instance_labels):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(shape[0]),
        'image/width': _int64_feature(shape[1]),
        'image/channels': _int64_feature(shape[2]),
        'image/image_data':_bytes_feature(image_data.tostring()),
        'image/superpixels':_bytes_feature(superpixels.tostring()),
        'image/mask_instance':_bytes_feature(mask_instance.tostring()),
        'image/mask_class':_bytes_feature(mask_class.tostring()),
    }))
    return example

def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    dm = dataset_dir + DIRECTORY_IMAGES + name +'.dcm'
    xml = dataset_dir + DIRECTORY_ANNOTATIONS + name + '.xml'
    image_data, superpixels = get_image_data_from_dicom(dm)
    mask_instance, mask_class, shape, class_labels, class_labels_text, instance_labels = groundtruth_to_mask(xml, dm)
    example = _convert_to_example(image_data,superpixels, mask_instance,
                                  mask_class, shape, class_labels,
                                  class_labels_text, instance_labels)
    tfrecord_writer.write(example.SerializeToString())

def _get_output_filename(output_dir, name, idx):
    return '%s/%s.tfrecord' % (output_dir, name)

def run(dataset_dir, output_dir, name='spine_segmentation_train', shuffling=False):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)
    i = 0
    fidx = 0
    while i < len(filenames):
        tf_filename  = _get_output_filename(output_dir, name, fidx)   
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()
                filename = filenames[i]
                img_name = filename[:-4]
                sys.stdout.write('\r>> Converting image %s' % (filename))
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1
    print('\nFinished converting the spine segmentation dataset!')



        
        
