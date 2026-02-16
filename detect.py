import os
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection

def detect_objects(model_path, image_path, threshold=0.5):
    # Carrega Modelo
    try:
        processor = RTDetrImageProcessor.from_pretrained(model_path)
        model = RTDetrForObjectDetection.from_pretrained(model_path)
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return

    # Carrega Imagem
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Predição
    with torch.no_grad():
        # RT-DETR does not use pixel_mask
        outputs = model(pixel_values=inputs['pixel_values'])

    # Pós-processamento
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    # Desenha
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    print(f"\n🔍 Analisando: {os.path.basename(image_path)}")
    count = 0
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        count += 1
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        confidence = round(score.item(), 2)
        
        print(f"   -> Encontrado: {label_name} ({confidence*100}%)")
        
        # Desenha caixa
        draw.rectangle(box, outline="red", width=3)
        
        # Desenha etiqueta
        text = f"{label_name}: {int(confidence*100)}%"
        text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
        draw.rectangle((text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]), fill="red")
        draw.text((box[0], box[1]), text, fill="white", font=font)

    if count == 0:
        print("   -> Nada detectado.")

    # Salva resultado
    output_filename = f"pred_{os.path.basename(image_path)}"
    image.save(output_filename)
    print(f"✅ Resultado salvo como: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./output/model", help="Pasta do modelo treinado")
    parser.add_argument("--image", type=str, help="Caminho de uma imagem única")
    parser.add_argument("--folder", type=str, help="Caminho de uma pasta de imagens")
    parser.add_argument("--conf", type=float, default=0.5, help="Confiança mínima (0 a 1)")
    args = parser.parse_args()

    if args.image:
        detect_objects(args.model, args.image, args.conf)
    elif args.folder:
        if not os.path.exists(args.folder):
            print("Pasta não encontrada.")
            exit()
        valid_images = [os.path.join(args.folder, f) for f in os.listdir(args.folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for img_path in valid_images:
            detect_objects(args.model, img_path, args.conf)
    else:
        print("❌ Use --image ou --folder para especificar o que detectar.")