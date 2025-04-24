import gradio as gr
import zipfile
import os
import shutil
from pathlib import Path
import tempfile

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

# Tratamento do arquivo zip (versão simplificada, para evitar problemas do Gradio 4.20.0)
def handle_zip_upload(file_obj):
    if file_obj is None:
        return "Nenhum arquivo enviado", ""
    
    try:
        # Garante que o diretório de upload exista e esteja vazio
        clear_upload_dir()
        
        # Extrai o arquivo zip
        temp_dir = tempfile.mkdtemp()
        temp_zip = os.path.join(temp_dir, "uploaded.zip")
        
        # Salva o arquivo temporariamente
        with open(temp_zip, "wb") as f:
            f.write(file_obj)
            
        # Extrai o conteúdo
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(UPLOAD_DIR)
            
        # Remove os arquivos temporários
        shutil.rmtree(temp_dir)
        
        # Lista os arquivos extraídos
        image_files = []
        text_files = []
        
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(Path(UPLOAD_DIR).rglob(f"*{ext}")))
            image_files.extend(list(Path(UPLOAD_DIR).rglob(f"*{ext.upper()}")))
        
        text_files = list(Path(UPLOAD_DIR).rglob("*.txt"))
        
        img_report = f"Encontradas {len(image_files)} imagens"
        txt_report = f"Encontrados {len(text_files)} arquivos de texto"
        
        if image_files:
            img_report += f": {', '.join([img.name for img in image_files[:5]])}"
            if len(image_files) > 5:
                img_report += "..."
                
        if text_files:
            txt_report += f": {', '.join([txt.name for txt in text_files[:5]])}"
            if len(text_files) > 5:
                txt_report += "..."
                
        return img_report, txt_report
        
    except Exception as e:
        return f"Erro ao processar o arquivo zip: {str(e)}", ""

# Calcula o número total de steps baseado em imagens, repetições e épocas
def calculate_steps(image_count, repeats, epochs, batch_size):
    try:
        repeats_val = int(repeats)
        epochs_val = int(epochs)
        batch_size_val = max(1, int(batch_size))  # Evita divisão por zero
        
        if image_count <= 0:
            return 0
            
        steps_per_epoch = max(1, (image_count * repeats_val) // batch_size_val)
        return steps_per_epoch * epochs_val
    except ValueError:
        return 0

# Função de treinamento
def start_training(model_base, resolution, batch_size, learning_rate, epochs,
                   train_text_encoder, lr_scheduler, precision, use_vae,
                   gradient_checkpoint, max_train_steps, save_every_n_steps,
                   repeats, clip_skip, lr_text, lr_unet, lr_scheduler_cycles,
                   warmup_steps, optimizer, network_dim, network_alpha):
    try:
        # Conta os arquivos de imagem
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(Path(UPLOAD_DIR).rglob(f"*{ext}")))
            image_files.extend(list(Path(UPLOAD_DIR).rglob(f"*{ext.upper()}")))
        
        image_count = len(image_files)
        
        if image_count == 0:
            return "Erro: Nenhuma imagem encontrada no dataset carregado. Por favor, faça upload de um dataset válido."
        
        # Calcula os steps se não forem especificados
        try:
            max_train_steps_val = int(max_train_steps) if max_train_steps and max_train_steps.strip() else 0
        except ValueError:
            max_train_steps_val = 0
            
        if max_train_steps_val <= 0:
            max_train_steps_val = calculate_steps(image_count, repeats, epochs, batch_size)
        
        # Aqui seria implementada a lógica real de treinamento
        # Por enquanto, apenas retornamos informações sobre a configuração
        
        config_info = f"""
Treinamento iniciado com as seguintes configurações:

- Dataset: {image_count} imagens encontradas
- Modelo Base: {model_base}
- Resolução: {resolution}
- Batch Size: {batch_size}
- Learning Rate: {learning_rate}
- Épocas: {epochs}
- Steps de Treinamento: {max_train_steps_val}
- Text Encoder: {"Ativado" if train_text_encoder else "Desativado"}
- Precision: {precision}
- Otimizador: {optimizer}
- Network Dim: {network_dim}
- Network Alpha: {network_alpha}

O treinamento está sendo simulado. Em uma implementação real, aqui seria iniciado o processo de treinamento do LoRA com estes parâmetros.
"""
        return config_info
        
    except Exception as e:
        return f"Erro ao iniciar o treinamento: {str(e)}"

# Interface Gradio
def create_interface():
    with gr.Blocks(title="LoRA Trainer UI") as demo:
        gr.Markdown("# LoRA Trainer UI para Stable Diffusion")
        gr.Markdown("## Upload do Dataset")
        gr.Markdown("Faça upload de um arquivo .zip contendo as imagens e arquivos .txt de legenda.")

        with gr.Group():
            with gr.Row():
                zip_input = gr.File(label="Dataset ZIP", file_types=[".zip"], type="binary")
            
            with gr.Row():
                upload_btn = gr.Button("Processar Dataset")
            
            with gr.Row():
                images_gallery = gr.Textbox(label="Imagens", lines=3)
                txt_gallery = gr.Textbox(label="Arquivos de texto", lines=3)

        gr.Markdown("## Configurações de Treinamento")

        # Verifica se existem modelos na pasta e adiciona um modelo padrão se necessário
        ensure_directories()
        model_files = [f.name for f in Path(MODELS_DIR).glob("*.safetensors")]
        if not model_files:
            model_files = ["modelo_exemplo.safetensors"]
        
        with gr.Group():
            with gr.Row():
                model_base = gr.Dropdown(label="Modelo Base", choices=model_files, value=model_files[0])
                resolution = gr.Textbox(label="Resolução", value="512x512", placeholder="512x512")
                batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=32, step=1, value=4)
                learning_rate = gr.Textbox(label="Learning Rate", value="1e-4")

            with gr.Row():
                epochs = gr.Slider(label="Épocas", minimum=1, maximum=100, step=1, value=10)
                repeats = gr.Textbox(label="Repetições por imagem", value="10")
                train_text_encoder = gr.Checkbox(label="Treinar Text Encoder", value=True)
                lr_scheduler = gr.Dropdown(label="Scheduler", 
                                        choices=["constant", "linear", "cosine", "polynomial"], 
                                        value="cosine")

            with gr.Row():
                precision = gr.Dropdown(label="Precisão", 
                                      choices=["fp16", "bf16", "fp32"], 
                                      value="fp16")
                use_vae = gr.Checkbox(label="Usar VAE Customizado", value=False)
                gradient_checkpoint = gr.Checkbox(label="Gradient Checkpointing", value=True)
                max_train_steps = gr.Textbox(label="Max Train Steps (opcional)", value="")

            with gr.Row():
                save_every_n_steps = gr.Textbox(label="Salvar a cada N Steps", value="500")
                clip_skip = gr.Slider(label="Clip Skip", minimum=1, maximum=12, step=1, value=2)
                lr_text = gr.Textbox(label="Learning Rate Text Encoder", value="0.00001")
                lr_unet = gr.Textbox(label="Learning Rate Unet", value="0.0001")

            with gr.Row():
                optimizer = gr.Dropdown(label="Otimizador", 
                                      choices=["AdamW", "Prodigy", "8bit Adam", "DAdaptation"], 
                                      value="Prodigy")
                lr_scheduler_cycles = gr.Slider(label="Scheduler Cycles", minimum=1, maximum=20, step=1, value=1)
                warmup_steps = gr.Slider(label="Warmup Steps", minimum=0, maximum=1000, step=10, value=0)

            with gr.Row():
                network_dim = gr.Slider(label="Network Dim", minimum=1, maximum=256, step=1, value=64)
                network_alpha = gr.Slider(label="Network Alpha", minimum=1, maximum=256, step=1, value=32)

        start_btn = gr.Button("Iniciar Treinamento", variant="primary")
        status_output = gr.Textbox(label="Status do Treinamento", lines=10)

        # Configurando os eventos
        upload_btn.click(
            fn=handle_zip_upload,
            inputs=[zip_input],
            outputs=[images_gallery, txt_gallery]
        )
        
        start_btn.click(
            fn=start_training,
            inputs=[
                model_base, resolution, batch_size, learning_rate, epochs,
                train_text_encoder, lr_scheduler, precision, use_vae,
                gradient_checkpoint, max_train_steps, save_every_n_steps,
                repeats, clip_skip, lr_text, lr_unet, lr_scheduler_cycles,
                warmup_steps, optimizer, network_dim, network_alpha
            ],
            outputs=status_output
        )
        
    return demo

if __name__ == "__main__":
    ensure_directories()
    demo = create_interface()
    demo.launch(share=True)
