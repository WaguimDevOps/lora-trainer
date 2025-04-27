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
import logging  # Adicione esta importação
import traceback  # Também útil para logging de erros
import torch.nn.functional as F  # Necessário para a função de perda
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    StableDiffusionPipeline, 
    UNet2DConditionModel, 
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
        
        # Se não tiver legendas, use o nome do arquivo
        if captions is None:
            self.captions = [os.path.splitext(os.path.basename(path))[0].replace('_', ' ') for path in image_paths]
        else:
            self.captions = captions
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]
        
        # Carregar e pré-processar a imagem
        image = Image.open(img_path).convert("RGB")
        
        # Redimensionar e centralizar a imagem
        image = self._resize_and_center_crop(image, self.size)
        
        # Converter para tensor e normalizar
        image = np.array(image) / 255.0
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = 2.0 * image - 1.0  # Normalizar para [-1, 1]
        image = torch.from_numpy(image).float()
        
        # Tokenizar a legenda se o tokenizer estiver disponível
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
            # Centralizar crop
            left = (new_width - size) // 2
            image = image.crop((left, 0, left + size, size))
        else:
            ratio = height / width
            new_height = int(size * ratio)
            image = image.resize((size, new_height), Image.LANCZOS)
            # Centralizar crop
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

# Configurar logging no início do arquivo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lora_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lora_trainer")

# Função principal de treinamento real
def start_training(model_base, resolution, batch_size, learning_rate, epochs,
                   train_text_encoder, lr_scheduler, precision, use_vae,
                   gradient_checkpoint, max_train_steps, save_every_n_steps,
                   repeats, clip_skip, lr_text, lr_unet, lr_scheduler_cycles,
                   warmup_steps, optimizer, network_dim, network_alpha):
    try:
        # Criar timestamp para o nome do modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"lora_{timestamp}"
        
        # Encontrar todas as imagens no dataset
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(Path(UPLOAD_DIR).rglob(f"*{ext}")))
        
        image_paths = [str(f) for f in image_files]
        image_count = len(image_paths)

        if image_count == 0:
            return "Erro: Nenhuma imagem encontrada no dataset. Faça o upload do arquivo zip primeiro."

        # Procurar por arquivos de legenda correspondentes
        captions = []
        for img_path in image_paths:
            base_name = os.path.splitext(img_path)[0]
            txt_path = f"{base_name}.txt"
            
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    captions.append(f.read().strip())
            else:
                # Usar o nome do arquivo como legenda
                captions.append(os.path.basename(base_name).replace('_', ' '))

        # Converter parâmetros para tipos adequados
        try:
            width, height = resolution.split('x')
            width, height = int(width), int(height)
            size = width  # Assumindo imagens quadradas para simplificar
        except:
            size = 512  # Valor padrão
            
        batch_size_int = int(batch_size)
        epochs_int = int(epochs)
        repeats_int = int(repeats) if repeats.isdigit() else 10
        lr = float(learning_rate)
        lr_text_float = float(lr_text) if lr_text else 1e-5
        lr_unet_float = float(lr_unet) if lr_unet else 1e-4
        network_dim_int = int(network_dim)
        network_alpha_int = int(network_alpha)
        
        # Determinar o número de passos
        if max_train_steps and max_train_steps.strip():
            total_steps = int(max_train_steps)
        else:
            steps_per_epoch = (image_count * repeats_int) // batch_size_int
            if steps_per_epoch <= 0:
                steps_per_epoch = 1
            total_steps = steps_per_epoch * epochs_int
        
        save_interval = int(save_every_n_steps) if save_every_n_steps and save_every_n_steps.isdigit() else 0
        
        # Configuração para precisão
        if precision == "fp16":
            weight_dtype = torch.float16
        elif precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32
        
        # Carregar o modelo base
        pipe = carregar_modelo_local(model_base)
        
        # Extrair componentes do pipeline
        vae = pipe.vae
        unet = pipe.unet
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        noise_scheduler = pipe.scheduler
        
        # Mover para GPU e configurar para o tipo de precisão
        vae.to("cuda", dtype=weight_dtype)
        unet.to("cuda", dtype=weight_dtype)
        text_encoder.to("cuda", dtype=weight_dtype)
        
        # Configurar LoRA para UNet
        unet_lora_config = LoraConfig(
            r=network_dim_int,
            lora_alpha=network_alpha_int,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )
        unet_lora = get_peft_model(unet, unet_lora_config)
        
        # Configurar LoRA para Text Encoder se necessário
        if train_text_encoder:
            text_encoder_lora_config = LoraConfig(
                r=network_dim_int,
                lora_alpha=network_alpha_int,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                lora_dropout=0.0,
                bias="none",
            )
            text_encoder_lora = get_peft_model(text_encoder, text_encoder_lora_config)
        
        # Configurar o Dataset
        train_dataset = LoRADataset(
            image_paths=image_paths * repeats_int,  # Repetir o dataset conforme configurado
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
        
        # Configurar otimizador
        if optimizer == "AdamW":
            if train_text_encoder:
                optimizer = torch.optim.AdamW(
                    [
                        {"params": unet_lora.parameters(), "lr": lr_unet_float},
                        {"params": text_encoder_lora.parameters(), "lr": lr_text_float}
                    ],
                    weight_decay=0.01
                )
            else:
                optimizer = torch.optim.AdamW(
                    unet_lora.parameters(),
                    lr=lr_unet_float,
                    weight_decay=0.01
                )
        else:
            # Para outros otimizadores, usar AdamW por padrão
            if train_text_encoder:
                optimizer = torch.optim.AdamW(
                    [
                        {"params": unet_lora.parameters(), "lr": lr_unet_float},
                        {"params": text_encoder_lora.parameters(), "lr": lr_text_float}
                    ],
                    weight_decay=0.01
                )
            else:
                optimizer = torch.optim.AdamW(
                    unet_lora.parameters(),
                    lr=lr_unet_float,
                    weight_decay=0.01
                )
        
        # Configurar scheduler de taxa de aprendizado
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
        else:  # polynomial or default
            lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer,
                total_iters=total_steps,
                power=1.0
            )
        
        # Loop de treinamento
        global_step = 0
        progress_text = f"Iniciando treinamento com {total_steps} steps...\n"
        yield progress_text
        
        # Definir o modelo para o modo de treinamento
        unet_lora.train()
        if train_text_encoder:
            text_encoder_lora.train()
        else:
            text_encoder.eval()
        
        vae.eval()  # VAE sempre em modo de avaliação
        
        # Verificar se o modelo é SDXL de forma mais robusta
        is_sdxl = (
            hasattr(pipe, "text_encoder_2") or 
            ("xl" in model_base.lower()) or
            (hasattr(unet.config, "cross_attention_dim") and unet.config.cross_attention_dim == 2048)
        )
        
        # Verificar se o modelo precisa de text_embeds (SD 2.x, SDXL, etc.)
        needs_text_embeds = (
            hasattr(unet.config, "addition_embed_type") and 
            unet.config.addition_embed_type in ["text_time", "text_image_time"]
        )
        for epoch in range(epochs_int):
            for batch in train_dataloader:
                global_step += 1
                if global_step > total_steps:
                    break
                
                # Mover o batch para GPU
                pixel_values = batch["pixel_values"].to("cuda", dtype=weight_dtype)
                input_ids = batch["input_ids"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")
                
                # Codificar a imagem com o VAE
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                
                # Obter texto embeddings
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

                # Ruído para o DDPM e timesteps aleatórios
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps, 
                    (batch_size,), 
                    device=latents.device
                )
                
                # Adicionar ruído às latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Para SDXL, criar condicionamentos adequados
                if is_sdxl:
                    # Preparar embeddings adicionais para SDXL
                    # Criar time_embeddings (exemplo, pode precisar ajuste para seu modelo específico)
                    original_size = (size, size)
                    target_size = (size, size)
                    
                    # Criar add_text_embeds com dimensão correta para batch
                    add_text_embeds = torch.zeros(
                        (batch_size, 1280),  # Dimensão típica para SDXL
                        device=latents.device,
                        dtype=weight_dtype
                    )
                    
                    # Preparar time_ids para SDXL (ajustar conforme necessário)
                    add_time_ids = torch.zeros(
                        (batch_size, 6),  # SDXL usa 6 valores para time_ids
                        device=latents.device,
                        dtype=weight_dtype
                    )
                    
                    # Preencher com valores corretos
                    for i in range(batch_size):
                        add_time_ids[i] = torch.tensor([
                            original_size[0],
                            original_size[1],
                            target_size[0],
                            target_size[1],
                            0,  # crop_top_x
                            0,  # crop_top_y
                        ], device=latents.device, dtype=weight_dtype)
                    
                    # Predição de ruído com condicionamentos adicionais
                    added_cond_kwargs = {
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids
                    }
                else:
                    # Para modelos não-SDXL, mas que ainda precisam de text_embeds e/ou time_ids
                    if hasattr(unet.config, "addition_embed_type") and unet.config.addition_embed_type == "text_time":
                        # Detectar a dimensão correta para text_embeds
                        if hasattr(unet, "add_embedding") and hasattr(unet.add_embedding, "linear_1"):
                            embed_dim = unet.add_embedding.linear_1.in_features
                        else:
                            embed_dim = 2816  # Valor comum para SD 2.x
                        
                        # Criar text_embeds com a dimensão correta
                        add_text_embeds = torch.zeros(
                            (batch_size, embed_dim),
                            device=latents.device,
                            dtype=weight_dtype
                        )
                        
                        # Criar time_ids para modelos não-SDXL
                        add_time_ids = torch.zeros(
                            (batch_size, 2),  # Formato típico para modelos não-SDXL
                            device=latents.device,
                            dtype=weight_dtype
                        )
                        
                        added_cond_kwargs = {
                            "text_embeds": add_text_embeds,
                            "time_ids": add_time_ids
                        }
                    else:
                        # Para outros modelos que não precisam de condicionamentos adicionais
                        added_cond_kwargs = {}
                
                # Predição de ruído
                try:
                    noise_pred = unet_lora(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeddings,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                except RuntimeError as e:
                    # Se o erro for sobre dimensões incompatíveis
                    if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                        # Extrair as dimensões do erro, se possível
                        import re
                        match = re.search(r'mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)', str(e))
                        
                        if match:
                            # Tentar usar a dimensão correta com base no erro
                            target_dim = int(match.group(3))
                            
                            # Criar text_embeds com a dimensão correta
                            new_text_embeds = torch.zeros(
                                (batch_size, target_dim),
                                device=latents.device,
                                dtype=weight_dtype
                            )
                            
                            # Atualizar o dicionário
                            new_cond_kwargs = added_cond_kwargs.copy()
                            new_cond_kwargs["text_embeds"] = new_text_embeds
                            
                            # Tentar novamente com a nova dimensão
                            try:
                                noise_pred = unet_lora(
                                    noisy_latents,
                                    timesteps,
                                    encoder_hidden_states=text_embeddings,
                                    added_cond_kwargs=new_cond_kwargs
                                ).sample
                            except Exception as nested_e:
                                # Se ainda falhar, tentar uma abordagem diferente
                                logger.error(f"Erro ao tentar com dimensão {target_dim}: {str(nested_e)}")
                                
                                # Verificar se o erro menciona dimensões específicas
                                if "3584" in str(e) and "2816" in str(e):
                                    # Este é um caso específico para SD 2.x
                                    logger.info("Detectado possível modelo SD 2.x, ajustando dimensões...")
                                    
                                    # Redimensionar os embeddings de texto para a dimensão esperada
                                    # SD 2.x espera 2816 em vez de 3584
                                    if text_embeddings.shape[-1] == 3584:
                                        # Usar uma projeção linear simples para reduzir a dimensão
                                        projection = torch.nn.Linear(3584, 2816).to(latents.device, dtype=weight_dtype)
                                        with torch.no_grad():
                                            text_embeddings_resized = projection(text_embeddings)
                                        
                                        # Tentar com os embeddings redimensionados
                                        noise_pred = unet_lora(
                                            noisy_latents,
                                            timesteps,
                                            encoder_hidden_states=text_embeddings_resized,
                                            added_cond_kwargs=added_cond_kwargs
                                        ).sample
                                    else:
                                        raise
                                else:
                                    raise
                        else:
                            # Se não conseguir extrair as dimensões, tentar uma última abordagem
                            logger.warning("Não foi possível extrair dimensões do erro. Tentando abordagem alternativa...")
                            
                            # Tentar sem added_cond_kwargs
                            try:
                                noise_pred = unet_lora(
                                    noisy_latents,
                                    timesteps,
                                    encoder_hidden_states=text_embeddings
                                ).sample
                            except Exception as final_e:
                                logger.error(f"Falha na última tentativa: {str(final_e)}")
                                raise e
                    else:
                        # Se não for um erro de dimensão, repassar o erro original
                        raise
                
                # Calcular a perda
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                # Backpropagation
                loss.backward()
                
                # Atualizar pesos
                optimizer.step()
                optimizer.zero_grad()
                
                # Atualizar learning rate
                lr_scheduler.step()
                
                # Atualizar progresso
                if global_step % 10 == 0:
                    progress_text += f"Step {global_step}/{total_steps}, Loss: {loss.item():.4f}\n"
                    yield progress_text
                
                # Salvar checkpoint intermediário se configurado
                if save_interval > 0 and global_step % save_interval == 0:
                    # Criar diretório para o checkpoint
                    checkpoint_dir = os.path.join(OUTPUT_DIR, f"{model_name}_step{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Salvar estado do modelo LoRA
                    unet_lora.save_pretrained(os.path.join(checkpoint_dir, "unet_lora"))
                    if train_text_encoder:
                        text_encoder_lora.save_pretrained(os.path.join(checkpoint_dir, "text_encoder_lora"))
                    
                    # Salvar configuração
                    config = {
                        "model_base": model_base,
                        "resolution": f"{size}x{size}",
                        "train_text_encoder": train_text_encoder,
                        "network_dim": network_dim_int,
                        "network_alpha": network_alpha_int,
                        "step": global_step,
                        "epoch": epoch,
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                    
                    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                        json.dump(config, f, indent=4)
                    
                    progress_text += f"Checkpoint salvo em {checkpoint_dir}\n"
                    yield progress_text
        
        # Salvar o modelo final
        final_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(final_dir, exist_ok=True)
        
        # Salvar estado do modelo LoRA
        unet_lora.save_pretrained(os.path.join(final_dir, "unet_lora"))
        if train_text_encoder:
            text_encoder_lora.save_pretrained(os.path.join(final_dir, "text_encoder_lora"))
        
        # Salvar configuração
        config = {
            "model_base": model_base,
            "resolution": f"{size}x{size}",
            "train_text_encoder": train_text_encoder,
            "network_dim": network_dim_int,
            "network_alpha": network_alpha_int,
            "total_steps": global_step,
            "epochs": epoch + 1,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        with open(os.path.join(final_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        
        progress_text += f"\nTreinamento concluído! Modelo salvo em {final_dir}"
        yield progress_text
        
    except Exception as e:
        error_msg = f"Erro ao iniciar o treinamento: {str(e)}\n\nDetalhes: {traceback.format_exc()}"
        logger.error(error_msg)
        yield error_msg

# Interface Gradio
def create_ui():
    with gr.Blocks(title="LoRA Trainer") as demo:
        gr.Markdown("# 🚀 LoRA Trainer")
        gr.Markdown("Treine modelos LoRA para Stable Diffusion com facilidade!")
        
        with gr.Tab("Dataset"):
            with gr.Row():
                with gr.Column():
                    zip_file = gr.File(label="Upload do Dataset (ZIP)", file_types=[".zip"])
                    upload_button = gr.Button("Processar Dataset")
                with gr.Column():
                    images_output = gr.Textbox(label="Imagens Encontradas", interactive=False)
                    captions_output = gr.Textbox(label="Arquivos de Texto Encontrados", interactive=False)
        
        with gr.Tab("Treinamento"):
            with gr.Row():
                with gr.Column():
                    model_base = gr.Dropdown(
                        label="Modelo Base",
                        choices=sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(('.safetensors', '.ckpt'))]),
                        value=None
                    )
                    resolution = gr.Dropdown(
                        label="Resolução",
                        choices=["512x512", "768x768", "1024x1024"],
                        value="512x512"
                    )
                    batch_size = gr.Dropdown(
                        label="Batch Size",
                        choices=["1", "2", "4", "8"],
                        value="1"
                    )
                    learning_rate = gr.Textbox(
                        label="Learning Rate",
                        value="1e-4"
                    )
                    epochs = gr.Textbox(
                        label="Épocas",
                        value="5"
                    )
                    train_text_encoder = gr.Checkbox(
                        label="Treinar Text Encoder",
                        value=True
                    )
                    
                with gr.Column():
                    lr_scheduler = gr.Dropdown(
                        label="LR Scheduler",
                        choices=["constant", "linear", "cosine", "polynomial"],
                        value="constant"
                    )
                    precision = gr.Dropdown(
                        label="Precisão",
                        choices=["fp16", "bf16", "fp32"],
                        value="fp16"
                    )
                    use_vae = gr.Checkbox(
                        label="Usar VAE",
                        value=False
                    )
                    gradient_checkpoint = gr.Checkbox(
                        label="Gradient Checkpointing",
                        value=False
                    )
                    max_train_steps = gr.Textbox(
                        label="Passos Máximos (vazio = auto)",
                        value=""
                    )
                    save_every_n_steps = gr.Textbox(
                        label="Salvar a cada N passos (0 = apenas no final)",
                        value="0"
                    )
            
            with gr.Accordion("Configurações Avançadas", open=False):
                with gr.Row():
                    with gr.Column():
                        repeats = gr.Textbox(
                            label="Repetições do Dataset",
                            value="10"
                        )
                        clip_skip = gr.Dropdown(
                            label="CLIP Skip",
                            choices=["1", "2"],
                            value="1"
                        )
                        lr_text = gr.Textbox(
                            label="Learning Rate (Text Encoder)",
                            value="1e-5"
                        )
                        lr_unet = gr.Textbox(
                            label="Learning Rate (UNet)",
                            value="1e-4"
                        )
                    with gr.Column():
                        lr_scheduler_cycles = gr.Textbox(
                            label="Ciclos do Scheduler",
                            value="1"
                        )
                        warmup_steps = gr.Textbox(
                            label="Warmup Steps",
                            value="0"
                        )
                        optimizer = gr.Dropdown(
                            label="Otimizador",
                            choices=["AdamW", "Lion", "AdamW8bit"],
                            value="AdamW"
                        )
                        network_dim = gr.Dropdown(
                            label="Dimensão da Rede (Rank)",
                            choices=["4", "8", "16", "32", "64", "128"],
                            value="32"
                        )
                        network_alpha = gr.Dropdown(
                            label="Alpha da Rede",
                            choices=["1", "4", "8", "16", "32", "64"],
                            value="16"
                        )
            
            train_button = gr.Button("Iniciar Treinamento", variant="primary")
            output = gr.Textbox(label="Status do Treinamento", interactive=False)
        
        # Conectar funções aos elementos da UI
        upload_button.click(handle_zip_upload, inputs=[zip_file], outputs=[images_output, captions_output])
        train_button.click(
            start_training,
            inputs=[
                model_base, resolution, batch_size, learning_rate, epochs,
                train_text_encoder, lr_scheduler, precision, use_vae,
                gradient_checkpoint, max_train_steps, save_every_n_steps,
                repeats, clip_skip, lr_text, lr_unet, lr_scheduler_cycles,
                warmup_steps, optimizer, network_dim, network_alpha
            ],
            outputs=[output]
        )
    
    return demo

# Iniciar a aplicação
if __name__ == "__main__":
    import torch.nn.functional as F
    import traceback
    import logging
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("lora_training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("lora_trainer")
    
    demo = create_ui()
    demo.queue()
    demo.launch(share=True)
     
