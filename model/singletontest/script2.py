import sys
sys.path.append('/home/abmonsur/SCR/SCR-1Bllama-lora-shared/model/singletontest')

from ston import LlamaModelSingleton

def analyze_text():
    llama_instance = LlamaModelSingleton(llama_path="path_to_llama_model")
    print(f"Analyzing text with Llama model: {llama_instance.llama}")

if __name__ == "__main__":
    analyze_text()
