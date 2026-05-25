import threading

class ModelSingleton:
    _instance = None
    _lock = threading.Lock()  # Ensures thread-safe singleton

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, model_path=None):
        """Load the model here (this method is called only once)."""
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = "/home/abmonsur/SCR/SCR-1Bllama-lora-shared/pretrain_model/Llama-3.2-1B"
        self.model = self._load_model(self.model_path)

    def _load_model(self, path):
        """Simulate model loading (replace with your actual model loading logic)."""
        print(f"Loading model from {path}")
        return f"Model at {path}"  # Replace with the actual model object

# Usage: Create an instance
singleton_instance = ModelSingleton(model_path="path_to_model")
