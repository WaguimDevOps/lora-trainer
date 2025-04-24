import gradio as gr
import zipfile
import os
import shutil
from pathlib import Path

UPLOAD_DIR = "uploaded_dataset"
MODELS_DIR = "models"

# Garantir que os diretórios necessários existam
def ensure_directories():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

# Limpa o diretório de upload anterior, se existir
def clear_upload_dir():
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR)

# Extrai o zip para a pasta de upload
def handle_zip_upload(zip_file):
    if zip_file is None:
        return "Nenhum arquivo enviado", ""
        
    if not zip_file.name.endswith(".zip"):
        return "Erro: o arquivo precisa ser um .zip", ""

    clear_upload_dir()
    
    # Extrair o arquivo zip
    try:
        with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
            zip_ref.extractall(UPLOAD_DIR)
        
        # Listar os arquivos extraídos
        extracted_files = list(Path(UPLOAD_DIR).rglob("*"))
        images = [f for f in extracted_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        texts = [f for f in extracted_files if f.suffix.lower() == '.txt']
        
        return f"Encontradas {len(images)} imagens e {len(texts)} arquivos de texto", \
               f"Imagens: {', '.join([img.name for img in images[:10]])}... \nTextos: {', '.join([txt.name for txt in texts[:10]])}"
    
    except Exception as e:
        return f"Erro ao extrair o arquivo: {str(e)}", ""

# Calcula o número total de steps baseado em imagens, repetições e épocas
def calculate_steps(image_count, repeats, epochs, batch_size):
    if batch_size <= 0:
        batch_size = 1
    steps_per_epoch = max(1, (image_count * repeats) // batch_size)
    return steps_per_epoch * epochs

# Função de treinamento
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
            return "Erro: Nenhuma imagem encontrada no dataset carregado."
            
        if not max_train_steps or max_train_steps.strip() == "":
            calculated_steps = calculate_steps(image_count, int(repeats), int(epochs), int(batch_size))
            max_train_steps = str(calculated_steps)
        
        # Aqui você implementaria a lógica real de treinamento do LoRA
        # Por enquanto, apenas retornamos informações sobre a configuração
        
        config_info = f"""
        Treinamento iniciado com as seguintes configurações:
        - Modelo base: {model_base}
        - Resolução: {resolution}
        - Batch size: {batch_size}
        - Learning rate: {learning_rate}
        - Épocas: {epochs}
        - Imagens encontradas: {image_count}
        - Steps de treinamento: {max_train_steps}
        """
        
        return config_info
        
    except Exception as e:
        return f"Erro ao iniciar o treinamento: {str(e)}"

# Interface Gradio
def create_interface():
    with gr.Blocks(title="LoRA Trainer UI") as demo:
        gr.Markdown("# Upload do Dataset para Treinamento LoRA")
        gr.Markdown("Faça upload de um `.zip` contendo as imagens e arquivos `.txt` de legenda.")

        with gr.Row():
            zip_input = gr.File(label=".zip do Dataset", file_types=[".zip"])
            upload_btn = gr.Button("Enviar e Extrair")

        with gr.Row():
            images_gallery = gr.Textbox(label="Imagens extraídas", lines=5)
            txt_gallery = gr.Textbox(label="Legendas extraídas", lines=5)

        upload_btn.click(fn=handle_zip_upload, inputs=zip_input, outputs=[images_gallery, txt_gallery])

        gr.Markdown("# Configurações de Treinamento")

        # Modelos base disponíveis na pasta 'models'
        model_files = [f.name for f in Path(MODELS_DIR).glob("*.safetensors")] or ["modelo_exemplo.safetensors"]
        
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
            max_train_steps = gr.Textbox(label="Max Train Steps (opcional)", placeholder="Automático")
            save_every_n_steps = gr.Textbox(label="Salvar a cada N Steps", value="500")

        gr.Markdown("## Configurações Avançadas")

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
        status_output = gr.Textbox(label="Status do Treinamento", lines=10)

        start_btn.click(
            fn=start_training,
            inputs=[model_base, resolution, batch_size, learning_rate, epochs,
                    train_text_encoder, lr_scheduler, precision, use_vae,
                    gradient_checkpoint, max_train_steps, save_every_n_steps,
                    repeats, clip_skip, lr_text, lr_unet, lr_scheduler_cycles,
                    warmup_steps, optimizer, network_dim, network_alpha],
            outputs=status_output
        )
        
    return demo

if __name__ == "__main__":
    ensure_directories()
    demo = create_interface()
    demo.launch(share=True)
