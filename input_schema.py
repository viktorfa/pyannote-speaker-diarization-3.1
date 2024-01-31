INPUT_SCHEMA = {
    "audio_url": {
        "datatype": "STRING",
        "required": True,
        "shape": [1],
        "example": [
            "https://transcription-python-sls-dev-resultbucket-lfgp8y5ths1i.s3.eu-central-1.amazonaws.com/transcription_jobs/8990a604-4943-4951-a6b0-c38b39b79703/audio.wav"
        ],
    },
    "num_speakers": {
        "datatype": "UINT8",
        "required": False,
        "example": [2],
        "shape": [1],
    },
    "min_speakers": {
        "datatype": "UINT8",
        "required": False,
        "example": [1],
        "shape": [1],
    },
    "max_speakers": {
        "datatype": "UINT8",
        "required": False,
        "example": [3],
        "shape": [1],
    },
    "webhook_url": {
        "datatype": "STRING",
        "required": False,
        "example": ["https://api.webhook.site"],
        "shape": [1],
    },
}
