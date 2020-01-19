import json
import numpy as np
import requests

# Model Constants
TF_MODEL_SERVER_URL = "http://localhost:9000/v1/models/multisensory:predict"
INFERENCE_OUTPUT_SAMPLES_FG = 'samples_pred_fg'
INFERENCE_OUTPUT_SAMPLES_BG = 'samples_pred_bg'
INFERENCE_OUTPUT_SPECTROGRAM_FG = 'spec_pred_fg'
INFERENCE_OUTPUT_SPECTROGRAM_BG = 'spec_pred_bg'

class InferenceResult: 
    def __init__(self, samples_pred_fg, samples_pred_bg, spec_pred_fg, spec_pred_bg, samples_mix, spec_mix):
        self.samples_pred_fg = samples_pred_fg
        self.samples_pred_bg = samples_pred_bg
        self.spec_pred_fg = spec_pred_fg
        self.spec_pred_bg = spec_pred_bg
        self.samples_mix = samples_mix
        self.spec_mix = spec_mix

class InferenceClient:
    @staticmethod
    def predict(video_frames, mixed_audio_spectrogram):
        data = {"inputs" : { 
            "mixed_audio_spectrogram": mixed_audio_spectrogram.tolist(),
            "video_frames": video_frames.tolist()} 
        }

        r = requests.post(url=TF_MODEL_SERVER_URL, data=json.dumps(data))
        outputs = r.json()['outputs']

        return InferenceResult(
            samples_pred_fg = np.array(outputs[INFERENCE_OUTPUT_SAMPLES_FG]),
            samples_pred_bg = np.array(outputs[INFERENCE_OUTPUT_SAMPLES_BG]),
            spec_pred_fg = np.array(outputs[INFERENCE_OUTPUT_SPECTROGRAM_FG]),
            spec_pred_bg = np.array(outputs[INFERENCE_OUTPUT_SPECTROGRAM_BG]),
            samples_mix = mixed_audio_spectrogram,
            spec_mix = np.array(outputs['specgram_op'])
        )
