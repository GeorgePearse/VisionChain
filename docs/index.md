## Home Page 

vision-research is home to model training and binit's Data Engine. 

```mermaid
graph TD
    A[Cloud Frames] -- run sql query --> B
    B[Object Cropper] --> C[Dataset]
    C[Dataset]-- Train Classifier --> F[Classifier Model Store] -- Clean from Embeddings --> C[Dataset]
    C -- Train Detector --> D[Detector Model Store]
    D --> B
    D --> E[Deploy Model]
```
