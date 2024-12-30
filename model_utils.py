from tensorflow.keras.models import model_from_json

def store_keras_model(model, model_name):
    # Menyimpan struktur model ke file JSON
    model_json = model.to_json()
    with open(f"./{model_name}.json", "w") as json_file:
        json_file.write(model_json)
    
    # Menyimpan bobot model ke file .h5
    model.save_weights(f"./{model_name}.weights.h5")
    print("Model berhasil disimpan ke disk.")

def load_keras_model(model_name):
    # Memuat struktur model dari file JSON
    with open(f'./{model_name}.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    
    model = model_from_json(loaded_model_json)
    
    # Memuat bobot model dari file .h5
    model.load_weights(f"./{model_name}.weights.h5")
    return model
