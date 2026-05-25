from ston import ModelSingleton

def use_model():
    llama_instance = ModelSingleton(llama_path="/home/abmonsur/SCR/SCR-1Bllama-lora-shared/pretrain_model/Llama-3.2-1B")
    print(f"Using Llama model: {llama_instance.llama}")

if __name__ == "__main__":
    use_model()
