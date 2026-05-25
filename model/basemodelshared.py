from transformers import AutoModelForCausalLM
import threading

class LlamaSingleton:
    _instance = None
    _lock = threading.Lock()  # To ensure thread safety for singleton

    def __new__(cls, config):
        if not cls._instance:
            with cls._lock:  # Ensure only one instance is created in multithreaded environments
                if not cls._instance:
                    print("Initializing the LLaMA model...")
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize(config)
        print("returned successfully")
        return cls._instance

    def _initialize(self, config):
        # Load the LLaMA model
        self.llama = AutoModelForCausalLM.from_pretrained(config.llama_path, output_attentions = True)
        print("LLaMA model loaded successfully!")
