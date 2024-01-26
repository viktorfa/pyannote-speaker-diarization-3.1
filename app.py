import urllib.request
from pyannote.audio import Pipeline
import torch
from pathlib import Path
import json


class InferlessPythonModel:
    def download_file(self,url):
        file_path = Path(__file__).parent / url.split("/")[-1]
        
        try:
            urllib.request.urlretrieve(url, str(file_path))
            print(f"File {str(file_path)} downloaded successfully.")
        except Exception as e:
            print(f"Failed to download the file. Error: {e}")
        
        return str(file_path)
    
        
    def initialize(self):
        self.pipeline = Pipeline.from_pretrained(
          "pyannote/speaker-diarization-3.1",
          use_auth_token="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")
        
        self.pipeline.to(torch.device("cuda"))

        
    def infer(self, inputs):
        audio_url = inputs["audio_url"]
        num_speakers = inputs.get("num_speakers")
        min_speakers = inputs.get("min_speakers")
        max_speakers = inputs.get("max_speakers")
        
        file_path = self.download_file(audio_url)
        diarization = self.pipeline(str(file_path), num_speakers=num_speakers,  min_speakers=min_speakers, max_speakers=max_speakers)

        segments = []

        speakers = {}
       
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms = int(turn.start * 1000)  
            end_ms = int(turn.end * 1000)

            if speaker not in speakers.keys():
                speakers[speaker] = {"label": speaker, "utterances": 1}
            else:
                speakers[speaker]["utterances"] += 1
            
            segments.append({
                "speaker": speaker,
                "start": start_ms,
                "end": end_ms
            })
            
        return json.dumps({"generated_data":{"result": {"segments": segments, "speakers": speakers, "n_speakers": len(speakers)}}})


    def finalize(self):
        pass