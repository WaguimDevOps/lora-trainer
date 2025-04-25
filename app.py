import gradio as gr
import zipfile
import os
import shutil
from pathlib import Path

UPLOAD_DIR = "uploaded_dataset"
MODELS_DIR = "models"

# Limpa o diretório de upload anterior, se existir
def clear_upload_dir():
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR)

# Extrai o zip para a pasta de upload
def handle_zip_upload(zip_file):
    if zip_file is None:
        return "Nenhum arquivo enviado", ""
        
    # Verificação para o tipo de arquivo
    if not str(zip_file.name).endswith(".zip"):
        return "Erro: o arquivo precisa ser um .zip", ""

    clear_upload_dir()
    
    try:
        with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
            zip_ref.extractall(UPLOAD_DIR)
        
        # Lista os arquivos extraídos
        extracted_files = list(Path(UPLOAD_DIR).rglob("*"))
        images = [f.name for f in extracted_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        texts = [f.name for f in extracted_files if f.suffix.lower() == '.txt']
        
        return ", ".join(images) if images else "Nenhuma imagem encontrada", ", ".join(texts) if texts else "Nenhum arquivo de texto encontrado"
    except Exception as e:
        return f"Erro ao extrair o arquivo: {str(e)}", ""

# Calcula o número total de steps baseado em imagens, repetições e épocas
def calculate_steps(image_count, repeats, epochs, batch_size):
    if batch_size <= 0:
        return 0
    steps_per_epoch = (image_count * repeats) // batch_size
    return steps_per_epoch * epochs

# Função de treinamento
def start_training(model_base, resolution, batch_size, learning_rate, epochs,
                   train_text_encoder, lr_scheduler, precision, use_vae,
                   gradient_checkpoint, max_train_steps, save_every_n_steps,
                   repeats, clip_skip, lr_text, lr_unet, lr_scheduler_cycles,
                   warmup_steps, optimizer, network_dim, network_alpha):
    try:
        # Verifica se há imagens no diretório de upload
        image_files = list(Path(UPLOAD_DIR).rglob("*.jpg")) + \
                      list(Path(UPLOAD_DIR).rglob("*.jpeg")) + \
                      list(Path(UPLOAD_DIR).rglob("*.png"))
        
        image_count = len(image_files)
        
        if image_count == 0:
            return "Erro: Nenhuma imagem encontrada no dataset. Faça o upload do arquivo zip primeiro."
        
        # Converte valores para o tipo adequado
        batch_size_int = int(batch_size)
        epochs_int = int(epochs)
        repeats_int = int(repeats) if repeats.isdigit() else 10
        
        # Calcula steps se não foi fornecido
        steps = max_train_steps
        if not steps or steps.strip() == "":
            steps = calculate_steps(image_count, repeats_int, epochs_int, batch_size_int)
        else:
            steps = int(steps)
        
        # Aqui você adicionaria o código real para iniciar o treinamento
        # Simulação do progresso de treinamento
        import time
        
        progress_text = f"Iniciando treinamento com {steps} steps...\n"
        
        for i in range(steps):
            time.sleep(0.1)  # Simula passo de treino
            progress = int((i + 1) / steps * 100)
            progress_text = f"Treinando... {progress}% ({i + 1}/{steps}) steps concluídos"
            yield progress_text
        
        yield f"✅ Treinamento finalizado com sucesso! Dataset: {image_count} imagens, {repeats_int} repetições, {epochs_int} épocas."
    
    except Exception as e:
        yield f"Erro ao iniciar o treinamento: {str(e)}"

# Interface Gradio
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

    # Verificar se a pasta models existe
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    # Modelos base disponíveis na pasta 'models'
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
