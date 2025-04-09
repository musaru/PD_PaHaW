# GmTC
# Parkinson Disease Detection Based on In-air Dynamics Feature Extraction and Selection UsingMachine Learning

If you used this resource please cited the following paper  "Parkinson Disease Detection Based on In-air
Dynamics Feature Extraction and Selection Using
Machine Learning, https://www.researchgate.net/publication/366980156_Dynamic_Hand_Gesture_Recognition_using_Multi-Branch_Attention_Based_Graph_and_General_Deep_Learning_Model" and "https://arxiv.org/abs/1907.08871"


***Abstract***
Hand gesture-based Sign Language Recognition (SLR) serves as a crucial communication bridge between deaf and non-deaf individuals. The absence of a universal sign language (SL) leads to diverse nationalities having various cultural SLs, such as Korean, American, and Japanese sign language. Existing SLR systems perform well for their cultural SL but may struggle with other or multi-cultural sign languages (McSL). To address these challenges, this paper introduces a novel end-to-end SLR system called GmTC, designed to translate McSL into equivalent text for enhanced understanding. Here, we employed a Graph and General deep-learning network as two stream modules to extract effective features. In the first stream, produce a graph-based feature by taking advantage of the superpixel values and the graph convolutional network (GCN), aiming to extract distance-based complex relationship features among the superpixel. In the second stream, we extracted long-range and short-range dependency features using attention-based contextual information that passes through multi-stage, multi-head self-attention (MHSA), and CNN modules. Combining these features generates final features that feed into the classification module. Extensive experiments with four culture SL datasets with high-performance accuracy compared to existing state-of-the-art models in individual domains affirming superiority and generalizability.


***Reason***
GmTC achieved good performance because of the effective feature extraction using the new combination. The combination of the super pixel-based GCN: distance-based superpixel complex relation, CNN: Local pattern, MHSA: global relationships and MLP-Conv features to fine-grained details and long-range dependencies captured seamlessly. This fine-grained and hierarchical feature plays a crucial role in improving performance accuracy. Moreover, the details about effectiveness features are given below: 
Two-Stream Model:
GmTC employs a two-stream deep-learning network, which is a key component of its architecture. These two streams are designed to work in parallel, each focusing on specific aspects of feature extraction.
Superpixel-based GCN Feature Extraction (First Stream):
Employs a Superpixel Based Graph Convolutional Network (GCN). It produces a graph-based feature by leveraging superpixel values and GCN. This process aims to extract complex relationship features among the superpixels, mainly focusing on distance-based relationships. 
GCNs aggregate information from neighbouring nodes in a graph and calculate super-pixel relationships to improve feature effectiveness. 
Here superpixels technique used to solve the pixel related large graph challenges. Superpixel are compact, perceptually meaningful regions of the image obtained by grouping similar characteristics pixels together in an image. Collect the compact information of the hand gesture image by aggregating the homogeneous region to reduce the computational complexity. Make higher-level representation of an image by grouping pixels that share common visual properties: color, texture, or intensity.
Attention-Based Contextual Information (Second Stream):
The second stream is responsible for extracting long-range and short-range dependency features. This is achieved through the use of attention-based contextual information that passes through multi-stage, multi-head self-attention (MHSA) and Convolutional Neural Network (CNN) modules. This attention mechanism allows the model to focus on relevant information for feature extraction.
Extracts attention-based features from the input data hierarchical relationship to improve feature effectiveness
Features processed through Multi-Head Self-Attention (MHSA) and CNN modules. Multiple stages of MHSA and CNN generate the hierarchical feature.
MHSA is used to capture dependencies and relationships between different elements and the output of the MHSA generated single head feature output in one iteration.
Intrinsic Feature Combination Features and Classification:
The features extracted from both streams, each capturing different aspects of the sign language gestures, are combined to generate the final features. This fusion process ensures that the model has a holistic representation of the input data.
Combined effective features enhance model effectiveness and increase performance accuracy.

## Dataset:

We evaluated the model with Three Dataset
PaHaW PD Handwriting Dataset:



## Preprocessing

## Main Diagram and code
coming soon....
## Citation
```
If the resource is useful please cite the following paper:
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

@inproceedings{chenBMVC19dynamic,
  author    = {Chen, Yuxiao and Zhao, Long and Peng, Xi and Yuan, Jianbo and Metaxas, Dimitris N.},
  title     = {Construct Dynamic Graphs for Hand Gesture Recognition via Spatial-Temporal Attention},
  booktitle = {BMVC},
  year      = {2019}
}

## Acknowledgement
We develop this project by taking theme andd base from "CMT: Convolutional Neural Networks Meet
Vision Transformers ", thanks them to open their project code and everything in public."[(https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/cmt_pytorch)]" https://github.com/huawei-noah/Efficient-AI-Backbones
Part of our code is borrowed from the ([https://arxiv.org/abs/1907.08871](https://gitee.com/mindspore/models/tree/master/research/cv/CMT)https://gitee.com/mindspore/models/tree/master/research/cv/CMT). We thank to the authors for releasing their codes.
