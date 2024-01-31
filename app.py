import urllib.request
from pyannote.audio import Pipeline
import torch
from pathlib import Path
import json
import tempfile
import urllib.parse
import requests


class InferlessPythonModel:
    def download_file(self, url):
        file_path = Path(tempfile.mkdtemp()) / url.split("/")[-1]

        try:
            urllib.request.urlretrieve(url, str(file_path))
            print(f"File {str(file_path)} downloaded successfully.")
        except Exception as e:
            print(f"Failed to download the file. Error: {e}")
            raise e

        return file_path

    def initialize(self):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="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",
        )

        self.pipeline.to(torch.device("cuda"))

    def infer(self, inputs):
        audio_url: str = inputs["audio_url"]
        num_speakers = (
            int(inputs.get("num_speakers")) if inputs.get("num_speakers") else None
        )
        min_speakers = (
            int(inputs.get("min_speakers")) if inputs.get("min_speakers") else None
        )
        max_speakers = (
            int(inputs.get("max_speakers")) if inputs.get("max_speakers") else None
        )
        webhook_url = inputs.get("webhook_url") or None

        if webhook_url:
            parsed_url = urllib.parse.urlparse(webhook_url)
            if not parsed_url.scheme:
                raise ValueError(f"Invalid webhook URL {webhook_url}")

        print("num_speakers")
        print(num_speakers)

        try:
            file_path = self.download_file(audio_url)
            diarization = self.pipeline(
                str(file_path),
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except Exception as e:
            print(f"Failed to process the file. Error: {e}")
            if webhook_url:
                requests.post(
                    webhook_url,
                    json={
                        "error": {"message": f"Failed to process the file. Error: {e}"},
                        "status": "FAILED",
                    },
                )
            raise e

        segments = []

        speakers = {}

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)

            if speaker not in speakers.keys():
                speakers[speaker] = {"label": speaker, "utterances": 1}
            else:
                speakers[speaker]["utterances"] += 1

            segments.append({"speaker": speaker, "start": start_ms, "end": end_ms})

        result = {
            "status": "COMPLETED",
            "output": {
                "segments": segments,
                "speakers": speakers,
                "n_speakers": len(speakers),
            },
            "input": {
                "audio_url": audio_url,
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            },
        }

        if webhook_url:
            requests.post(webhook_url, json=result)

        return {"result": json.dumps(result)}

    def finalize(self):
        pass
