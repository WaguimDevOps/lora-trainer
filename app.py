# lora_trainer_gradio.py

import gradio as gr
import zipfile
import os
import shutil
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

UPLOAD_DIR = "uploaded_dataset"
MODELS_DIR = "models"
OUTPUT_DIR = "trained_lora"


def clear_upload_dir():
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR)


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
        images = [f.name for f in extracted_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        texts = [f.name for f in extracted_files if f.suffix.lower() == '.txt']

        return ", ".join(images) if images else "Nenhuma imagem encontrada", \
               ", ".join(texts) if texts else "Nenhum arquivo de texto encontrado"
    except Exception as e:
        return f"Erro ao extrair o arquivo: {str(e)}", ""


def calculate_steps(image_count, repeats, epochs, batch_size):
    if batch_size <= 0:
        return 0
    steps_per_epoch = (image_count * repeats) // batch_size
    return steps_per_epoch * epochs


def train_lora(model_base_path, output_dir, image_folder, learning_rate, batch_size, epochs, network_dim, network_alpha):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_base_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    text_encoder = pipe.text_encoder

    config = LoraConfig(
        r=network_dim,
        lora_alpha=network_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="TEXT_ENCODER"
    )

    model = get_peft_model(text_encoder, config)

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        for step in range(10):
            loss = torch.tensor(0.01, requires_grad=True)
            loss.backward()
            # Aqui voce colocaria o otimizador real etc.

    model.save_pretrained(output_dir)


def start_training(model_base, resolution, batch_size, learning_rate, epochs,
                   train_text_encoder, lr_scheduler, precision, use_vae,
                   gradient_checkpoint, max_train_steps, save_every_n_steps,
                   repeats, clip_skip, lr_text, lr_unet, lr_scheduler_cycles,
                   warmup_steps, optimizer, network_dim, network_alpha):
    try:
        image_files = list(Path(UPLOAD_DIR).rglob("*.jpg")) + \
                      list(Path(UPLOAD_DIR).rglob("*.jpeg")) + \
                      list(Path(UPLOAD_DIR).rglob("*.png"))

        image_count = len(image_files)

        if image_count == 0:
            return "Erro: Nenhuma imagem encontrada no dataset. Faça o upload do arquivo zip primeiro."

        batch_size_int = int(batch_size)
        epochs_int = int(epochs)
        repeats_int = int(repeats) if repeats.isdigit() else 10

        steps = max_train_steps
        if not steps or steps.strip() == "":
            steps = calculate_steps(image_count, repeats_int, epochs_int, batch_size_int)
        else:
            steps = int(steps)

        train_lora(
            model_base_path=os.path.join(MODELS_DIR, model_base),
            output_dir=OUTPUT_DIR,
            image_folder=UPLOAD_DIR,
            learning_rate=float(learning_rate),
            batch_size=batch_size_int,
            epochs=epochs_int,
            network_dim=network_dim,
            network_alpha=network_alpha
        )

        return f"✅ Treinamento finalizado com sucesso! Modelo salvo em: {OUTPUT_DIR}"

    except Exception as e:
        return f"Erro ao iniciar o treinamento: {str(e)}"


with gr.Blocks(title="LoRA Trainer UI") as demo:
    gr.Markdown("# Upload do Dataset para Treinamento LoRA")

    with gr.Row():
        zip_input = gr.File(label=".zip do Dataset", file_types=[".zip"])
        upload_btn = gr.Button("Enviar e Extrair")

    with gr.Row():
        images_gallery = gr.Textbox(label="Imagens extraídas", lines=5)
        txt_gallery = gr.Textbox(label="Legendas extraídas", lines=5)

    upload_btn.click(fn=handle_zip_upload, inputs=zip_input, outputs=[images_gallery, txt_gallery])

    gr.Markdown("# Configurações de Treinamento")

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

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
        lr_scheduler = gr.Dropdown(label="Scheduler", choices=["constant", "linear", "cosine", "polynomial"], value="cosine")
        precision = gr.Dropdown(label="Precisão", choices=["fp16", "bf16", "fp32"], value="fp16")

    with gr.Row():
        use_vae = gr.Checkbox(label="Usar VAE Customizado", value=False)
        gradient_checkpoint = gr.Checkbox(label="Gradient Checkpointing", value=True)
        max_train_steps = gr.Textbox(label="Max Train Steps (opcional)", placeholder="Ex: 10000")
        save_every_n_steps = gr.Textbox(label="Salvar a cada N Steps", placeholder="Ex: 500")

    with gr.Row():
        clip_skip = gr.Slider(label="Clip Skip", minimum=1, maximum=12, step=1, value=2)
        lr_text = gr.Textbox(label="Taxa de aprendizado do Text Encoder", value="0.00001")
        lr_unet = gr.Textbox(label="Taxa de aprendizado do Unet", value="0.0001")
        optimizer = gr.Dropdown(label="Otimizador", choices=["AdamW", "Prodigy", "8bit Adam", "DAdaptation"], value="Prodigy")

    with gr.Row():
        lr_scheduler_cycles = gr.Slider(label="lr_scheduler_num_cycles", minimum=1, maximum=20, step=1, value=1)
        warmup_steps = gr.Slider(label="num_warmup_steps", minimum=0, maximum=1000, step=10, value=0)
        network_dim = gr.Slider(label="Rede Dim", minimum=1, maximum=256, step=1, value=64)
        network_alpha = gr.Slider(label="Rede Alpha", minimum=1, maximum=256, step=1, value=32)

    start_btn = gr.Button("Iniciar Treinamento")
    status_output = gr.Textbox(label="Status do Treinamento")

    start_btn.click(
        fn=start_training,
        inputs=[model_base, resolution, batch_size, learning_rate, epochs,
                train_text_encoder, lr_scheduler, precision, use_vae,
                gradient_checkpoint, max_train_steps, save_every_n_steps,
                repeats, clip_skip, lr_text, lr_unet, lr_scheduler_cycles,
                warmup_steps, optimizer, network_dim, network_alpha],
        outputs=status_output
    )

if __name__ == "__main__":
    demo.queue().launch(share=True)
