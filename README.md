Human Activity Recognition System
> A real-time computer vision system that automatically detects and classifies human activities using pose estimation and machine learning.

Activities Recognized
Walking | Person moving forward on foot |
| Sitting | Person seated in resting position |
| Standing | Person upright and stationary |
| Bending | Person bending forward at waist |

Technologies Used
| Python 3.13 | Core programming language |
| MediaPipe | Body pose estimation (33 keypoints) |
| OpenCV | Video processing & webcam feed |
| Scikit-learn | Random Forest classification |
| Roboflow | Dataset annotation & management |
| Pandas & NumPy | Data processing |
| Matplotlib & Seaborn | Result visualization |

Dataset & Annotation

- 24 videos collected manually across 4 activities
- Annotated using Roboflow annotation tool
- 52 labeled frames across 4 activity classes
- Images resized to 224×224 pixels
- Dataset split:
- Split | Percentage | Images |
|-------|-----------|--------|
| Train | 70% | 36 |
| Valid | 21% | 11 |
| Test  | 10% | 5  |

Roboflow Project:[har_project]
