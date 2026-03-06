import numpy as np
import soundfile as sf
import torch
import traceback
import torchaudio
from pathlib import Path
import argparse
from resemble_enhance.enhancer.inference import denoise, enhance as re_enhance
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.layers.vits.networks import TextEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--text", required=True, help="Frase a sintetizar")
args = parser.parse_args()

FRASES = [args.text]
CKPT = "./vits_colombian/output/train5/best_model.pth"
OUTPUTS = Path()
OUTPUTS.mkdir(parents=True, exist_ok=True)
NOISE_SCALE = 0.8
NOISE_SCALE_W = 0.6
LENGTH_SCALE = 1.5


best_ckpt = Path(CKPT)
config_path_ft = None

for parent in [
    best_ckpt.parent,
    best_ckpt.parent.parent,
    best_ckpt.parent.parent.parent,
]:
    candidate = parent / "config.json"
    if candidate.exists():
        config_path_ft = candidate
        break

if config_path_ft is None:
    raise FileNotFoundError("No se encontró config.json para el modelo")

inf_config = VitsConfig()
inf_config.load_json(str(config_path_ft))
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

inf_model.load_checkpoint(inf_config, str(best_ckpt), eval=True)
DEVICE = "cpu"
inf_model = inf_model.to(DEVICE).eval()

def extraer_wav(outputs):

    if isinstance(outputs, dict):

        for key in ["wav", "waveform", "audio", "model_outputs", "wav_seg"]:
            if key in outputs:
                return outputs[key]

        for v in outputs.values():
            if torch.is_tensor(v):
                return v

        raise RuntimeError("No se encontró tensor de audio")

    if isinstance(outputs, (list, tuple)):
        return outputs[0]

    if torch.is_tensor(outputs):
        return outputs

    raise TypeError(f"Formato desconocido: {type(outputs)}")


def main():

    sr_out = inf_config.audio.sample_rate
    for i, texto in enumerate(FRASES):

        try:

            tokens = inf_model.tokenizer.text_to_ids(texto)
            x = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)
            x_lengths = torch.LongTensor([x.shape[1]]).to(DEVICE)
            lang_ids = torch.LongTensor([0]).to(DEVICE)

            mejor_wav = None
            mejor_energia = -1

            for seed in [0, 42, 99, 123, 7]:

                torch.manual_seed(seed)
                with torch.no_grad():

                    outputs = inf_model.inference(
                        x,
                        aux_input={
                            "x_lengths": x_lengths,
                            "speaker_ids": None,
                            "language_ids": lang_ids,
                            "noise_scale": NOISE_SCALE,
                            "noise_scale_w": NOISE_SCALE_W,
                            "length_scale": LENGTH_SCALE,
                        },
                    )

                wav = extraer_wav(outputs)
                audio_tmp = wav.squeeze().cpu().numpy()
                energia = np.abs(audio_tmp).mean()

                if energia > mejor_energia:
                    mejor_energia = energia
                    mejor_wav = audio_tmp

            audio_np = mejor_wav
            peak = np.abs(audio_np).max()

            if peak > 0:
                audio_np = audio_np / peak * 0.92

            umbral = 0.01
            margen = int(sr_out * 0.15)
            muestras = np.where(np.abs(audio_np) > umbral)[0]

            if len(muestras) > 0:
                ultimo = muestras[-1] + margen
                audio_np = audio_np[: min(ultimo, len(audio_np))]

            temp_path = OUTPUTS / f"temp_{i+1}.wav"
            sf.write(temp_path, audio_np, sr_out)

            dwav, sr = torchaudio.load(temp_path)
            dwav = dwav.mean(0)

            with torch.no_grad():

                denoised, sr2 = denoise(dwav, sr, device=DEVICE)
                enhanced, sr3 = re_enhance(
                    denoised,
                    sr2,
                    device=DEVICE,
                    nfe=16,
                    solver="midpoint",
                    lambd=0.5,
                    tau=0.5,
                )

            enhanced_np = enhanced.cpu().numpy()
            muestras = np.where(np.abs(enhanced_np) > umbral)[0]

            if len(muestras) > 0:
                ultimo = muestras[-1] + int(sr3 * 0.15)
                enhanced_np = enhanced_np[: min(ultimo, len(enhanced_np))]

            enhanced = torch.tensor(enhanced_np, dtype=torch.float32)
            out_path = OUTPUTS / f"resultado_{i+1}.wav"
            torchaudio.save(out_path, enhanced.unsqueeze(0), sr3)
            temp_path.unlink()
            
            print(f"RESULT_PATH={out_path.resolve()}")

        except Exception:

            print(f"ERROR frase {i+1}: {texto}")
            traceback.print_exc()


if __name__ == "__main__":
    main()