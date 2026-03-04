import argparse, glob, re
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=None)
parser.add_argument("--epoch", default=None, type=int)
parser.add_argument("--out", default="./vits_colombian/inferencia")
parser.add_argument("--base", default="./vits_colombian")
args = parser.parse_args()

BASE_DIR = Path(args.base)
OUTPUT_DIR = BASE_DIR / "output"
OUTPUTS = Path(args.out)
OUTPUTS.mkdir(parents=True, exist_ok=True)

if args.ckpt:
    best_ckpt = args.ckpt
else:
    candidates = sorted(glob.glob(str(OUTPUT_DIR / "**" / "best_model_*.pth"), recursive=True))
    if not candidates:
        candidates = sorted(glob.glob(str(OUTPUT_DIR / "**" / "checkpoint_*.pth"), recursive=True))
    assert candidates, "No hay checkpoints."

    if args.epoch is not None:
        def step_num(p):
            m = re.search(r'(\d+)\.pth$', p)
            return int(m.group(1)) if m else 0
        best_ckpt = min(candidates, key=lambda p: abs(step_num(p) - args.epoch * 111))
    else:
        best_ckpt = candidates[-1]

print(f"Checkpoint: {best_ckpt}")
config_path_ft = None
for parent in [Path(best_ckpt).parent, Path(best_ckpt).parent.parent, Path(best_ckpt).parent.parent.parent]:
    if (parent / "config.json").exists():
        config_path_ft = str(parent / "config.json")
        break

assert config_path_ft, "No se encontro config.json"
print(f"Config: {config_path_ft}")

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.layers.vits.networks import TextEncoder

inf_config = VitsConfig()
inf_config.load_json(config_path_ft)

inf_model = Vits.init_from_config(inf_config)
ma = inf_config.model_args
inf_model.text_encoder = TextEncoder(
    n_vocab=ma.num_chars,
    out_channels=ma.hidden_channels,
    hidden_channels=192,
    hidden_channels_ffn=768,
    num_heads=ma.num_heads_text_encoder,
    num_layers=6,
    kernel_size=3,
    dropout_p=0.1,
    language_emb_dim=4,
)

inf_model.load_checkpoint(inf_config, best_ckpt, eval=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
inf_model = inf_model.to(DEVICE).eval()
print(f"Modelo en {DEVICE}\n")

def extraer_wav(outputs):
    if isinstance(outputs, dict):
        print(f"  [claves disponibles]: {list(outputs.keys())}")
        for key in ["wav", "waveform", "audio", "model_outputs", "wav_seg"]:
            if key in outputs:
                return outputs[key]

        for v in outputs.values():
            if torch.is_tensor(v) and v.ndim >= 1:
                return v
        raise KeyError(f"Sin tensor de audio en: {list(outputs.keys())}")
    elif isinstance(outputs, (list, tuple)):
        return outputs[0]
    elif torch.is_tensor(outputs):
        return outputs
    raise TypeError(f"Formato desconocido: {type(outputs)}")

frases = [
    "Hola parce que dia tan bueno hace hoy, me voy a tomar un tinto y después salgo a dar una vuelta por el parque",
    "Hola buenos dias mano, en que puedo ayudarte hoy?"
]

sr_out = inf_config.audio.sample_rate
step = m.group(1) if m else "unk"
m = re.search(r'(\d+)\.pth$', best_ckpt)

for i, texto in enumerate(frases):
    try:
        tokens = inf_model.tokenizer.text_to_ids(texto)
        x = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)
        x_lengths = torch.LongTensor([x.shape[1]]).to(DEVICE)
        lang_ids = torch.LongTensor([0]).to(DEVICE)

        with torch.no_grad():
            outputs = inf_model.inference(
                x,
                aux_input={"x_lengths": x_lengths, "speaker_ids": None, "language_ids": lang_ids}
            )

        wav = extraer_wav(outputs)
        audio_np = wav.squeeze().cpu().numpy() if torch.is_tensor(wav) else np.asarray(wav).squeeze()
        peak = np.abs(audio_np).max()
        if peak > 0.001:
            audio_np = audio_np / peak * 0.92

        out_path = OUTPUTS / f"step{step}_frase{i+1}.wav"
        sf.write(str(out_path), audio_np, sr_out)
        print(f"{i+1} {texto}")

    except Exception as e:
        import traceback
        print(f"ERROR [{i+1}] {texto}")
        traceback.print_exc()