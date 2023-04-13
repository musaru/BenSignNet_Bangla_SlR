# BenSignNet_Bangla_SlR
# BenSignNet: Bengali Sign Language Alphabet Recognition Using Concatenated Segmentation and Convolutional 
Neural Network

## Abstract

Sign language recognition is one of the most challenging applications in machine learning and human-computer interaction. Many researchers have developed classification models for
different sign languages such as English, Arabic, Japanese, and Bengali; however, no significant
research has been done on the general-shape performance for different datasets. Most research work
has achieved satisfactory performance with a small dataset. These models may fail to replicate the
same performance for evaluating different and larger datasets. In this context, this paper proposes
a novel method for recognizing Bengali sign language(BSL) alphabets to overcome the issue of
generalization. The proposed method has been evaluated with three benchmark datasets such as ‘38
BdSL’, ‘KU-BdSL’, and ‘Ishara-Lipi’. Here, three steps are followed to achieve the goal: segmentation,
augmentation, and Convolutional neural network (CNN) based classification. Firstly, a concatenated
segmentation approach with YCbCr, HSV and watershed algorithm was designed to accurately
identify gesture signs. Secondly, seven image augmentation techniques are selected to increase the
training data size without changing the semantic meaning. Finally, the CNN-based model called
BenSignNet was applied to extract the features and classify purposes. The performance accuracy of
the model achieved 94% , 99.60% , and 99.60% for the BdSL Alphabet, KU-BdSL, and Ishara-Lipi
datasets, respectively. Experimental findings confirmed that our proposed method achieved a higher
recognition rate than the conventional ones and accomplished a generalization property in all datasets
for the BSL domain.

Keywords: Bengali sign language (BSL); Convolutional neural network (CNN); 38-BdSL; Ishara-Lipi;
KU-BdSL; concatenated segmentation; Luminance blue red (YCbCr); Hue saturation value (HSV)

# Training
```
# Proposed Dataset
# BdSL38 Dataset
# Lab Dataset
```
We uploaded here the Pickle file of the three dataset which contained the landmark information and label for each trial.

# Please Cite the following paper if this information is useful:

Journal
```
@article{miah2023dynamic,
  title={Dynamic Hand Gesture Recognition using Multi-Branch Attention Based Graph and General Deep Learning Model},
  author={Miah, Abu Saleh Musa and Hasan, Md Al Mehedi and Shin, Jungpil},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}

}
@article{shin2023korean,
  title={Korean Sign Language Recognition Using Transformer-Based Deep Neural Network},
  author={Shin, Jungpil and Musa Miah, Abu Saleh and Hasan, Md Al Mehedi and Hirooka, Koki and Suzuki, Kota and Lee, Hyoun-Sup and Jang, Si-Woong},
  journal={Applied Sciences},
  volume={13},
  number={5},
  pages={3029},
  year={2023},
  publisher={MDPI}
}

@article{miah2023multistage,
  title={Multistage Spatial Attention-Based Neural Network for Hand Gesture Recognition},
  author={Miah, Abu Saleh Musa and Hasan, Md Al Mehedi and Shin, Jungpil and Okuyama, Yuichi and Tomioka, Yoichi},
  journal={Computers},
  volume={12},
  number={1},
  pages={13},
  year={2023},
  publisher={MDPI}
}
@article{miah2022bensignnet,
  title={BenSignNet: Bengali Sign Language Alphabet Recognition Using Concatenated Segmentation and Convolutional Neural Network},
  author={Miah, Abu Saleh Musa and Shin, Jungpil and Hasan, Md Al Mehedi and Rahim, Md Abdur},
  journal={Applied Sciences},
  volume={12},
  number={8},
  pages={3933},
  year={2022},
  publisher={MDPI}
}

@article{miahrotation,
  title={Rotation, Translation and Scale Invariant Sign Word Recognition Using Deep Learning},
  author={Miah, Abu Saleh Musa and Shin, Jungpil and Hasan, Md Al Mehedi and Rahim, Md Abdur and Okuyama, Yuichi},
journal={ Computer Systems Science and Engineering },
  volume={44},
  number={3},
  pages={2521–2536},
  year={2023},
  publisher={TechSchince}

}
```
