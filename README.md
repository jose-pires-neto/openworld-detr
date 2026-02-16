# **🧠 OpenWorld-DETR: Next-Gen Object Detection**

Este projeto implementa um detector de objetos baseado na arquitetura **DETR (DEtection TRansformer)**, alinhado com a visão de **World Models** (Modelos de Mundo) da Meta Research (FAIR).

Diferente de arquiteturas baseadas em CNNs clássicas (como YOLO), este modelo utiliza **Transformers** e **Mecanismos de Atenção Global** para "raciocinar" sobre a imagem inteira de uma vez, oferecendo robustez superior em cenários de oclusão e poucos dados.

## **🚀 Por que usar este projeto?**

* **Raciocínio Global:** Entende contexto e oclusão melhor que CNNs.  
* **Data-Efficient:** Aprende com menos épocas e imagens (Few-Shot Learning).  
* **Plug & Play:** Estrutura simplificada para treinar com datasets customizados (formato COCO).  
* **Sem Âncoras:** Elimina a necessidade de "anchor boxes" manuais e NMS (Non-Maximum Suppression).

## **📦 Instalação**

1. Clone este repositório:
```bash
git clone https://github.com/jose-pires-neto/openworld-detr.git
cd openworld-detr
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## **📂 Estrutura de Pastas**

Organize seu dataset da seguinte forma (formato padrão de exportação COCO do Roboflow):

```
openworld-detr/
├── dataset/
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   └── imagem1.jpg...
│   ├── valid/
│   │   ├── _annotations.coco.json
│   │   └── imagem2.jpg...
├── output/           # Onde o modelo treinado será salvo
├── train_world_model.py
└── detect.py
```

## **🏋️‍♂️ Como Treinar**

Basta rodar o comando abaixo. O script detecta automaticamente o número de classes no seu JSON.

```bash
python train_world_model.py --epochs 15 --batch_size 4
```

Isso salvará o modelo treinado na pasta `output/model`.

## **👁️ Como Testar (Inferência)**

Para testar o modelo em novas imagens e ver os resultados:

```bash
python detect.py --image "caminho/para/uma/imagem.jpg"
# OU para testar uma pasta inteira:
python detect.py --folder "dataset/valid"
```

## **📚 Referências**

* **DETR:** [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)  
* **I-JEPA:** [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)