import gradio as gr
import zipfile
import os
import shutil
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    AutoencoderKL
)
from peft import LoraConfig, get_peft_model

UPLOAD_DIR = "uploaded_dataset"
MODELS_DIR = "models"
OUTPUT_DIR = "output-loras"

# Verificar e criar diretórios necessários
for dir_path in [UPLOAD_DIR, MODELS_DIR, OUTPUT_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Dataset personalizado para treinamento LoRA
class LoRADataset(Dataset):
    def __init__(self, image_paths, captions=None, tokenizer=None, size=512):
        self.image_paths = image_paths
        self.size = size
        self.tokenizer = tokenizer
        
        if captions is None:
            self.captions = [os.path.splitext(os.path.basename(path))[0].replace('_', ' ') for path in image_paths]
        else:
            self.captions = captions
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]
        
        image = Image.open(img_path).convert("RGB")
        image = self._resize_and_center_crop(image, self.size)
        image = np.array(image) / 255.0
        image = image.transpose(2, 0, 1)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image).float()
        
        if self.tokenizer:
            tokens = self.tokenizer(
                caption,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            return {
                "pixel_values": image,
                "input_ids": tokens.input_ids[0],
                "attention_mask": tokens.attention_mask[0]
            }
        else:
            return {
                "pixel_values": image,
                "caption": caption
            }
    
    def _resize_and_center_crop(self, image, size):
        width, height = image.size
        if width > height:
            ratio = width / height
            new_width = int(size * ratio)
            image = image.resize((new_width, size), Image.LANCZOS)
            left = (new_width - size) // 2
            image = image.crop((left, 0, left + size, size))
        else:
            ratio = height / width
            new_height = int(size * ratio)
            image = image.resize((size, new_height), Image.LANCZOS)
            top = (new_height - size) // 2
            image = image.crop((0, top, size, top + size))
        return image

# Limpa o diretório de upload anterior, se existir
def clear_upload_dir():
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR)

# Extrair o zip para a pasta de upload
def handle_zip_upload(zip_file):
    if zip_file is None:
        return "Nenhum arquivo enviado", ""

    if not str(zip_file.name).endswith(".zip"):
        return "Erro: o arquivo precisa ser um .zip", ""

    clear_upload_dir()

    try:
        with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
            zip_ref.extractall(UPLOAD_DIR)

        extracted_files = list(Path(UPLOAD_DIR).rglob("*"))
        images = [str(f) for f in extracted_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        texts = [str(f) for f in extracted_files if f.suffix.lower() == '.txt']

        image_names = [os.path.basename(img) for img in images]
        text_names = [os.path.basename(txt) for txt in texts]

        return ", ".join(image_names) if images else "Nenhuma imagem encontrada", ", ".join(text_names) if texts else "Nenhum arquivo de texto encontrado"
    except Exception as e:
        return f"Erro ao extrair o arquivo: {str(e)}", ""

# Carrega modelo base local
def carregar_modelo_local(model_filename):
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo {model_path} não encontrado.")

    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        use_safetensors=True
    )
    pipe.to("cuda")
    return pipe

# Função principal de treinamento real
def start_training(model_base, resolution, batch_size, learning_rate, epochs,
                   train_text_encoder, lr_scheduler, precision, use_vae,
                   gradient_checkpoint, max_train_steps, save_every_n_steps,
                   repeats, clip_skip, lr_text, lr_unet, lr_scheduler_cycles,
                   warmup_steps, optimizer, network_dim, network_alpha):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"lora_{timestamp}"
        
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(Path(UPLOAD_DIR).rglob(f"*{ext}")))
        image_paths = [str(f) for f in image_files]
        image_count = len(image_paths)

        if image_count == 0:
            return "Erro: Nenhuma imagem encontrada no dataset. Faça o upload do arquivo zip primeiro."

        captions = []
        for img_path in image_paths:
            base_name = os.path.splitext(img_path)[0]
            txt_path = f"{base_name}.txt"
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    captions.append(f.read().strip())
            else:
                captions.append(os.path.basename(base_name).replace('_', ' '))

        try:
            width, height = resolution.split('x')
            size = int(width)
        except:
            size = 512

        batch_size_int = int(batch_size)
        epochs_int = int(epochs)
        repeats_int = int(repeats) if repeats.isdigit() else 10
        lr = float(learning_rate)
        lr_text_float = float(lr_text) if lr_text else 1e-5
        lr_unet_float = float(lr_unet) if lr_unet else 1e-4
        network_dim_int = int(network_dim)
        network_alpha_int = int(network_alpha)

        if max_train_steps and max_train_steps.strip():
            total_steps = int(max_train_steps)
        else:
            steps_per_epoch = (image_count * repeats_int) // batch_size_int
            if steps_per_epoch <= 0:
                steps_per_epoch = 1
            total_steps = steps_per_epoch * epochs_int
        save_interval = int(save_every_n_steps) if save_every_n_steps and save_every_n_steps.isdigit() else 0

        if precision == "fp16":
            weight_dtype = torch.float16
        elif precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32

        pipe = carregar_modelo_local(model_base)
        vae = pipe.vae
        unet = pipe.unet
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        noise_scheduler = pipe.scheduler

        vae.to("cuda", dtype=weight_dtype)
        unet.to("cuda", dtype=weight_dtype)
        text_encoder.to("cuda", dtype=weight_dtype)

        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        unet_lora_config = LoraConfig(
            r=network_dim_int,
            lora_alpha=network_alpha_int,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
        )
        unet_lora = get_peft_model(unet, unet_lora_config)

        if train_text_encoder:
            text_encoder_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
            text_encoder_lora_config = LoraConfig(
                r=network_dim_int,
                lora_alpha=network_alpha_int,
                target_modules=text_encoder_target_modules,
                lora_dropout=0.0,
                bias="none",
            )
            text_encoder_lora = get_peft_model(text_encoder, text_encoder_lora_config)

        train_dataset = LoRADataset(
            image_paths=image_paths * repeats_int,
            captions=captions * repeats_int,
            tokenizer=tokenizer,
            size=size
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size_int,
            shuffle=True,
            drop_last=True
        )

        if optimizer == "AdamW":
            params = unet_lora.parameters()
            if train_text_encoder:
                params = [
                    {"params": unet_lora.parameters(), "lr": lr_unet_float},
                    {"params": text_encoder_lora.parameters(), "lr": lr_text_float}
                ]
            optimizer = torch.optim.AdamW(params, weight_decay=0.01)
        else:
            params = unet_lora.parameters()
            if train_text_encoder:
                params = [
                    {"params": unet_lora.parameters(), "lr": lr_unet_float},
                    {"params": text_encoder_lora.parameters(), "lr": lr_text_float}
                ]
            optimizer = torch.optim.AdamW(params, weight_decay=0.01)

        if lr_scheduler == "constant":
            lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        elif lr_scheduler == "linear":
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps
            )
        elif lr_scheduler == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=1e-6
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer,
                total_iters=total_steps,
                power=1.0
            )

        global_step = 0
        yield f"Iniciando treinamento com {total_steps} steps..."

        unet_lora.train()
        if train_text_encoder:
            text_encoder_lora.train()
        else:
            text_encoder.eval()
        vae.eval()

        for epoch in range(epochs_int):
            for batch in train_dataloader:
                global_step += 1
                if global_step > total_steps:
                    break

                pixel_values = batch["pixel_values"].to("cuda", dtype=weight_dtype)
                input_ids = batch["input_ids"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

                if train_text_encoder:
                    text_embeddings = text_encoder_lora(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )[0]
                else:
                    with torch.no_grad():
                        text_embeddings = text_encoder(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )[0]
                # Ajuste de shape para SDXL
                text_embeddings = text_embeddings.mean(dim=1)

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Config adicional genérico
                has_add = hasattr(unet.config, "addition_embed_type")
                addition = unet.config.addition_embed_type if has_add else None
                if has_add:
                    added_cond_kwargs = {}
                    text_dim = getattr(unet.config, "addition_text_embed_dim",
                                        getattr(unet.config, "addition_embed_dim", None))
                    if text_dim:
                        added_cond_kwargs["text_embeds"] = torch.zeros((batch_size, text_dim), device=latents.device, dtype=weight_dtype)
                    time_dim = getattr(unet.config, "addition_time_embed_dim", None)
                    if time_dim:
                        added_cond_kwargs["time_ids"] = torch.zeros((batch_size, time_dim), device=latents.device, dtype=weight_dtype)
                    img_dim = getattr(unet.config, "addition_image_embed_dim", None)
                    if addition == "text_image_time" and img_dim:
                        added_cond_kwargs["image_embeds"] = torch.zeros((batch_size, img_dim), device=latents.device, dtype=weight_dtype)

                    noise_pred = unet_lora(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeddings,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                else:
                    noise_pred = unet_lora(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeddings
                    ).sample

                loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                yield f"Treinando... {int(global_step/total_steps*100)}% | Loss: {loss.item():.4f}"
                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break

        final_model_path = os.path.join(OUTPUT_DIR, f"{model_name}.safetensors")
        from safetensors.torch import save_file
        tensors_dict = {k: v for k, v in unet_lora.state_dict().items() if isinstance(v, torch.Tensor)}
        save_file(tensors_dict, final_model_path)
        json.dump({"base_model": model_base}, open(os.path.join(OUTPUT_DIR, f"{model_name}_config.json"), 'w'), indent=2)
        yield f"✅ Treinamento finalizado. Modelo salvo em: {final_model_path}"

    except Exception as e:
        import traceback
        yield f"Erro ao iniciar o treinamento: {str(e)}\n{traceback.format_exc()}"

# Interface Gradio
def build_ui():
    with gr.Blocks(title="LoRA Trainer SDXL") as demo:
        gr.Markdown("# Upload do Dataset para Treinamento LoRA")
        with gr.Row():
            zip_input = gr.File(label=".zip do Dataset", type="file")
            upload_btn = gr.Button("Enviar e Extrair")
        with gr.Row():
            images_gallery = gr.Textbox(label="Imagens extraídas", lines=5)
            txt_gallery = gr.Textbox(label="Legendas extraídas", lines=5)
        upload_btn.click(fn=handle_zip_upload, inputs=zip_input, outputs=[images_gallery, txt_gallery])

        gr.Markdown("# Configurações de Treinamento")
        model_files = [f.name for f in Path(MODELS_DIR).glob("*.safetensors")]
        with gr.Row():
            model_base = gr.Dropdown(label="Modelo Base", choices=model_files, value=model_files[0] if model_files else None)
            resolution = gr.Textbox(label="Resolução (ex: 512x512)", value="512x512")
            batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=64, step=1, value=4)
            learning_rate = gr.Textbox(label="Learning Rate", value="1e-4")
        with gr.Row():
            epochs = gr.Slider(label="Épocas", minimum=1, maximum=100, step=1, value=10)
            repeats = gr.Textbox(label="Repetições por imagem", value="10")
            train_text_encoder = gr.Checkbox(label="Treinar Text Encoder", value=True)
            lr_scheduler = gr.Dropdown(label="Scheduler", choices=["constant","linear","cosine","polynomial"], value="cosine")
            precision = gr.Dropdown(label="Precisão", choices=["fp16","bf16","fp32"], value="fp16")
        with gr.Row():
            max_train_steps = gr.Textbox(label="Max Train Steps (opcional)", placeholder="Ex: 10000")
            save_every_n_steps = gr.Textbox(label="Salvar a cada N Steps", placeholder="Ex: 500")
        start_btn = gr.Button("Iniciar Treinamento")
        status_output = gr.Textbox(label="Status do Treinamento")
        start_btn.click(
            fn=start_training,
            inputs=[model_base, resolution, batch_size, learning_rate, epochs,
                    train_text_encoder, lr_scheduler, precision, False,
                    True, max_train_steps, save_every_n_steps,
                    repeats, None, None, None,
                    None, None, None, None],
            outputs=status_output
        )
    return demo

if __name__ == "__main__":
    build_ui().queue().launch(share=True)
