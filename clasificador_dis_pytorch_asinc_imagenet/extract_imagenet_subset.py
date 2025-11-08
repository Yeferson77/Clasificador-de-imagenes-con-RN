import os
import tarfile
import random
import shutil
from pathlib import Path

def extract_subset_imagenet(tar_path, output_dir, subset_percentage=0.05):
    """
    Extrae un subconjunto del dataset ImageNet directamente del archivo .tar
    
    Args:
        tar_path: Ruta al archivo ILSVRC2012_img_train.tar
        output_dir: Directorio donde extraer el subconjunto
        subset_percentage: Porcentaje de imágenes a extraer por clase (0.05 = 5%)
    """
    print(f"Extrayendo {subset_percentage*100}% de cada clase de ImageNet...")
    
    # Crear directorio de salida
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Directorio temporal para procesar los .tar de cada clase
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        with tarfile.open(tar_path, 'r:') as train_tar:
            # Iterar sobre los .tar de cada clase
            for class_tar_info in train_tar:
                if not class_tar_info.name.endswith('.tar'):
                    continue
                    
                class_name = class_tar_info.name.split('.')[0]
                print(f"Procesando clase: {class_name}")
                
                # Extraer el .tar de la clase al directorio temporal
                class_tar = train_tar.extractfile(class_tar_info)
                temp_tar_path = temp_dir / class_tar_info.name
                with open(temp_tar_path, 'wb') as f:
                    f.write(class_tar.read())
                
                # Crear directorio para esta clase
                class_dir = output_dir / 'train' / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Extraer subset de imágenes de esta clase
                with tarfile.open(temp_tar_path, 'r:') as class_tar:
                    images = [f for f in class_tar.getnames() if f.endswith('.JPEG')]
                    # Seleccionar aleatoriamente subset_percentage del total
                    num_to_extract = max(1, int(len(images) * subset_percentage))
                    selected_images = random.sample(images, num_to_extract)
                    
                    # Extraer solo las imágenes seleccionadas
                    for img_name in selected_images:
                        class_tar.extract(img_name, class_dir)
                
                # Limpiar archivo temporal de la clase
                temp_tar_path.unlink()
                
    finally:
        # Limpiar directorio temporal
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("Extracción completada!")

if __name__ == "__main__":
    # Ajusta estas rutas según tu configuración
    TAR_PATH = "../clasificador_dis_pytorch_asinc_v3_imagenet/data/imagenet/ILSVRC2012_img_train.tar"  # archivo original
    OUTPUT_DIR = "../clasificador_dis_pytorch_asinc_v3_imagenet/data/imagenet"  # donde se extraerá el subset
    
    extract_subset_imagenet(TAR_PATH, OUTPUT_DIR, subset_percentage=0.05)