from dataclasses import dataclass


@dataclass
class Embedder:

    def embed(self, file_path: str): 
        pass

@dataclass
class HFEmbedder(Embedder):
    preprocessor_name: str 
    model_name: str 

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ViTImageProcessor.from_pretrained(self.preprocessor_name)
        self.model = ViTModel.from_pretrained(self.model_name).to(device)

    def embed(self, file_path: str) -> List[float]: 
        image = Image.open(filepath)
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state[0][0]
    
        return embeddings


@dataclass
class EnsembleEmbedder(Embedder):
    embedders: List[Embedder]

    def embed(self, file_path) -> List[float]: 
        embeddings = []
        for embedder in self.embedders: 
            embedding = embedder.embed(file_path)
            embeddings.append(embedding)

        return embeddings