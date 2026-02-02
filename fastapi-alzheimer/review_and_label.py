import pickle
import numpy as np
from pathlib import Path
from retrain import save_to_buffer, retrain_incremental, MIN_PATIENTS

PENDING_PATH = Path("buffer/predictions_pending.pkl")
BUFFER_NEW = Path("buffer/new_patients.pkl")

if BUFFER_NEW.exists():
    print("Limpiando buffer de entrenamiento anterior...")
    BUFFER_NEW.unlink()

if not PENDING_PATH.exists():
    print("No hay predicciones pendientes")
    exit()

with open(PENDING_PATH, "rb") as f:
    pending = pickle.load(f)

print(f"\nCasos pendientes: {len(pending)}\n")

total_buffer = None

for i, item in enumerate(pending):
    print("=" * 50)
    print(f"CASO #{i+1}")
    label_text = "Alzheimer" if item['predicted_label'] == 1 else "No Alzheimer"
    print(f"Predicción modelo: {label_text} (code: {item['predicted_label']})")
    print(f"Confianza: {item['confidence']*100:.2f}%")

    resp = input("¿Es correcto el diagnóstico? (s/n): ").lower()

    if resp == "s":
        final_label = item["predicted_label"]
    else:
        final_label = 1 - item["predicted_label"]


    total_buffer = save_to_buffer(
        X_prep=np.array(item["X"]),
        y=[final_label]
    )

    print(f"✔ Guardado. Total en buffer: {total_buffer}")

# AHORA SÍ, AL FINAL
if total_buffer is not None and total_buffer >= MIN_PATIENTS:
    print("\nUmbral alcanzado. Reentrenando automáticamente...\n")
    msg = retrain_incremental()
    print("✅", msg)
else:
    print("\n⏳ No se alcanzó el mínimo para reentrenar")

# limpiar pendientes
PENDING_PATH.unlink()
