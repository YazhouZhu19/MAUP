<div align="center">

<h1>MAUP: Training-free Multi-center Adaptive Uncertainty-aware Prompting for Cross-domain Few-shot Medical Image Segmentation </h1>



<p align="center"><img width="95%" src="./overview.png" />


</div>

## Abstaract

Cross-domain Few-shot Medical Image Segmentation (CD-FSMIS) is a potential solution for segmenting medical images with limited annotation using knowledge from other domains. The significant performance of current CD-FSMIS models relies on the heavily training procedure over other source medical domains, which degrades the universality and ease of model deployment. With the development of large visual models of natural images, we propose a training-free CD-FSMIS model that introduces the Multi-center Adaptive Uncertainty-aware Prompting (MAUP) strategy for adapting the foundation model Segment Anything Model (SAM), which is trained with natural images, into the CD-FSMIS task. To be specific, MAUP consists of three key innovations: (1) K-means clustering based multi-center prompts generation for comprehensive spatial coverage, (2) uncertainty-aware prompts selection that focuses on the challenging regions, and (3) adaptive prompt optimization that can dynamically adjust according to the target region complexity. With the pre-trained DINOv2 feature encoder, MAUP achieves precise segmentation results across three medical datasets without any additional training compared with several conventional CD-FSMIS models and training-free FSMIS model.
