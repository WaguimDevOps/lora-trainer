import gradio as gr
import zipfile
import os
import shutil
import torch
import datetime
from pathlib import Path
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel
from torch.optim import AdamW
from diffusers.optimization import get_scheduler

UPLOAD_DIR = "uploaded_dataset"
MODELS_DIR = "models"
OUTPUT_DIR = "trained_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ... [Funções clear_upload_dir e handle_zip_upload permanecem iguais] ...

def create_lora_network(unet, text_encoder, network_dim=64, network_alpha=32):
    # Adiciona camadas LoRA ao UNet e Text Encoder
    def add_lora(layer, dim, alpha):
        layer_lora = torch.nn.Linear(layer.in_features, layer.out_features, bias=False)
        layer_lora.weight = torch.nn.Parameter(torch.randn(dim, dim) * (alpha / dim))
        return layer_lora

    # Aplica LoRA às camadas do UNet
    for module in unet.modules():
        if isinstance(module, torch.nn.Linear):
            module.lora_layer = add_lora(module, network_dim, network_alpha)

    # Aplica LoRA ao Text Encoder
    for module in text_encoder.modules():
        if isinstance(module, torch.nn.Linear):
            module.lora_layer = add_lora(module, network_dim, network_alpha)

    return unet, text_encoder

def start_training(model_base, resolution, batch_size, learning_rate, epochs,
                   train_text_encoder, lr_scheduler_type, precision, use_vae,
                   gradient_checkpointing, max_train_steps, save_every_n_steps,
                   repeats, clip_skip, lr_text, lr_unet, lr_scheduler_cycles,
                   warmup_steps, optimizer_type, network_dim, network_alpha):
    try:
        # Configuração inicial
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if precision == "fp16" else torch.float32
        
        # Carrega o modelo base
        model_path = os.path.join(MODELS_DIR, model_base)
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
        
        # Configura LoRA
        unet, text_encoder = create_lora_network(
            pipe.unet, 
            pipe.text_encoder,
            network_dim=network_dim,
            network_alpha=network_alpha
        )

        # Otimizador
        optimizer = AdamW(
            [
                {"params": unet.parameters(), "lr": float(lr_unet)},
                {"params": text_encoder.parameters(), "lr": float(lr_text)}
            ],
            lr=float(learning_rate)

        )
        
        # Agendador de learning rate
        lr_scheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
            num_cycles=lr_scheduler_cycles
        )

        # Prepara dados de treinamento
        image_files = list(Path(UPLOAD_DIR).rglob("*.jpg")) + \
                     list(Path(UPLOAD_DIR).rglob("*.jpeg")) + \
                     list(Path(UPLOAD_DIR).rglob("*.png"))
        
        # Configuração do treinamento
        batch_size = int(batch_size)
        num_epochs = int(epochs)
        max_train_steps = int(max_train_steps) if max_train_steps else len(image_files) * num_epochs // batch_size

        # Loop de treinamento
        global_step = 0
        for epoch in range(num_epochs):
            for i in range(0, len(image_files), batch_size):
                # Simulação do batch (substituir por carregamento real de imagens e textos)
                batch = image_files[i:i+batch_size]
                
                # Forward pass e backward pass
                optimizer.zero_grad()
                
                # ... [Implementar lógica real de treinamento aqui] ...
                
                loss = torch.rand(1)  # Simulação de perda
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                # Progresso
                progress = global_step / max_train_steps
                yield f"Step {global_step}/{max_train_steps} | Loss: {loss.item():.4f} | LR: {lr_scheduler.get_last_lr()[0]:.6f}"

                # Salva checkpoint
                if save_every_n_steps and (global_step % int(save_every_n_steps) == 0):
                    save_path = os.path.join(OUTPUT_DIR, f"lora_step_{global_step}")
                    pipe.save_pretrained(save_path)
                    
                global_step += 1

        # Salva o modelo final
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(OUTPUT_DIR, f"lora_final_{timestamp}")
        pipe.save_pretrained(save_path)
        
        yield f"✅ Treinamento completo! Modelo salvo em: {save_path}"

    except Exception as e:
        yield f"Erro no treinamento: {str(e)}"

# ... [A interface Gradio permanece similar, com ajustes nos tipos de parâmetros] ...

if __name__ == "__main__":
    demo.queue().launch(share=True)
