import os
import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrImageProcessor
from pytorch_lightning import Trainer

# ==============================================================================
# CONFIGURAÇÃO DO DATASET
# ==============================================================================
class CocoDetection(Dataset):
    def __init__(self, img_folder, feature_extractor):
        self.img_folder = img_folder
        self.feature_extractor = feature_extractor
        
        # Busca automática pelo arquivo JSON
        self.annotation_file = os.path.join(img_folder, '_annotations.coco.json')
        if not os.path.exists(self.annotation_file):
            raise FileNotFoundError(f"❌ Erro: '_annotations.coco.json' não encontrado em {img_folder}")

        import pycocotools.coco as coco
        self.coco = coco.COCO(self.annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotations = coco.loadAnns(ann_ids)
        
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_path = os.path.join(self.img_folder, file_name)
        
        # Fallback caso a imagem esteja na raiz sem pastas
        if not os.path.exists(img_path):
             img_path = os.path.join(self.img_folder, os.path.basename(file_name))
        
        img = Image.open(img_path).convert("RGB")

        new_annotations = []
        for ann in coco_annotations:
            new_annotations.append({
                "id": ann["id"], "image_id": ann["image_id"], "category_id": ann["category_id"],
                "bbox": ann["bbox"], "area": ann["area"], "iscrowd": ann.get("iscrowd", 0)
            })

        encoding = self.feature_extractor(images=img, annotations={'image_id': img_id, 'annotations': new_annotations}, return_tensors="pt")
        return encoding["pixel_values"].squeeze(), encoding["labels"][0]

    def __len__(self):
        return len(self.ids)

class CollateFn:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        return {'pixel_values': encoding['pixel_values'], 'pixel_mask': encoding['pixel_mask'], 'labels': labels}

# ==============================================================================
# O CÉREBRO (LIGHTNING MODULE)
# ==============================================================================
class DetrModel(pl.LightningModule):
    def __init__(self, num_labels, id2label, label2id, learning_rate=1e-4, lr_backbone=1e-5):
        super().__init__()
        self.save_hyperparameters()
        # Carrega modelo pré-treinado, mas ajusta a cabeça final para nossas classes
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        self.lr = learning_rate
        self.lr_backbone = lr_backbone

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad], "lr": self.lr_backbone},
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=1e-4)

# ==============================================================================
# EXECUÇÃO
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinador do World Model (DETR)")
    parser.add_argument("--data", type=str, default="./dataset/train", help="Caminho para a pasta de treino (contendo JSON)")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=4, help="Tamanho do batch")
    parser.add_argument("--output_dir", type=str, default="./output/model", help="Onde salvar o modelo")
    args = parser.parse_args()

    print(f"🚀 Iniciando treinamento na pasta: {args.data}")

    # 1. Analisa as classes automaticamente lendo o JSON
    try:
        with open(os.path.join(args.data, '_annotations.coco.json')) as f:
            data = json.load(f)
        categories = data['categories']
        
        # Mapeamento de ID para Nome (Ex: 0 -> 'Gado')
        id2label = {cat['id']: cat['name'] for cat in categories}
        label2id = {cat['name']: cat['id'] for cat in categories}
        num_labels = len(categories)
        print(f"✅ Classes detectadas ({num_labels}): {list(label2id.keys())}")

    except Exception as e:
        print(f"❌ Erro ao ler JSON: {e}")
        exit()

    # 2. Prepara Processador e Dados
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    train_dataset = CocoDetection(img_folder=args.data, feature_extractor=processor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=CollateFn(processor), shuffle=True, num_workers=2, persistent_workers=True)

    # 3. Inicializa e Treina
    model = DetrModel(num_labels=num_labels, id2label=id2label, label2id=label2id)
    trainer = Trainer(max_epochs=args.epochs, accelerator="auto", devices=1, logger=False)
    trainer.fit(model, train_loader)

    # 4. Salva
    print(f"💾 Salvando modelo em {args.output_dir}...")
    model.model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("✅ Concluído com sucesso!")