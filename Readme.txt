Aum Sri Sai Ram
Our model is implemented in Pytorch
Contents:
    1. models folder has base resnet model and attentionnet contains our SCAN and CCI branch.
    2. dataset folder has dataset class for Aff-wild2 and Aff-wild2+Expw+Affectnet separately.
    3. util.py for loading pretrained vggfacenet model.
    4. pretrainedmodels for storing pretrained weights from vggface2 model
              (https://github.com/ox-vgg/vgg_face2)
              
Usage:
   Training:
   set the path to args.root_path as 'Affwild2/' containing cropped_aligned folder of images provided by organizers
   set args.metafile as 'Annotations/annotations.pkl'. It has annotations for both train and valdiation videos frames.
   python train_affwild2.py 
                OR 
   python train_affwild2_expw_affectnet.py
   
   Testing: 
   set args.metafile 'Annotations/test_set.pkl'. 
   python test_affwild2.py
