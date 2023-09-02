from dataclasses import dataclass

@dataclass
class Classifier:
    collection_name: str
    client: QdrantClient
    embedder: Embedder

    
    def __post_init__(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )

    
    def train(self, train_dataset: fo.Dataset):
        """
        Just needs to run inference over all of the objects. 
        Make it very easy to plug and play different models 
        into this. 
        """
        embeddings = []
        for filepath in tqdm(train_dataset.values('filepath')):
            embedding = embedder.embed(filepath)
            #print(len(embedding))
            embeddings.append(embedding)
        
        
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "file_path": file_path,
                        'label': label,
                    }
                )
                for idx, (vector, file_path, label) in enumerate(zip(
                    embeddings, 
                    object_dataset.values('filepath'),
                    object_dataset.values('ground_truth'),
                ))
            ]
        )

    def predict(self, query_image_path: str, num_nearest_neighbours: int = 5) -> ClassificationPrediction:
        hits = self.client.search(
            collection_name=collection_name,
            query_vector=get_embeddings(query_image_path).tolist()[0][0],
            limit=num_nearest_neighbours, 
        )

        predictions = []
        for hit in hits:
            prediction = Prediction(
                name=hit.payload['label']['label'],
                score=hit.score,
            )
            predictions.append(prediction)
            
        return self.aggregate(predictions)

    
    @staticmethod
    def aggregate(predictions: List[Prediction], method = 'majority') -> ClassificationPrediction:

        if method == 'majority':
            neighbour_labels = [prediction.name for prediction in predictions]
            return max(set(neighbour_labels), key=neighbour_labels.count)

        if method == 'weighted':                 
            raise Exception('Weighted aggregate not yet implemented')
        