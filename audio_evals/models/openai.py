import base64
import json
import os
from typing import Dict, Any
import openai

from audio_evals.models.model import APIModel
from audio_evals.base import PromptStruct


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com")

logger = logging.getLogger(__name__)


class GPT(APIModel):
    def __init__(self, model_name: str, sample_params: Dict[str, Any] = None):
        super().__init__(True, sample_params)
        self.model_name = model_name
        assert "OPENAI_API_KEY" in os.environ, ValueError(
            "not found OPENAI_API_KEY in your ENV"
        )

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        messages = []
        for item in prompt:
            messages.append(
                {"role": item["role"], "content": item["contents"][0]["value"]}
            )

        response = openai.ChatCompletion.create(
            model=self.model_name, messages=messages, **kwargs
        )

        return response.choices[0].message["content"]


class GPT4oAudio(APIModel):
    def __init__(self, model_name: str, sample_params: Dict[str, Any] = None):
        super().__init__(True, sample_params)
        self.model_name = model_name
        self.url = "{}"
        assert "OPENAI_API_KEY" in os.environ, ValueError(
            "not found OPENAI_API_KEY in your ENV"
        )

    async def access(self, audio_path, text):
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }

        async def send_event(websocket, event):
            await websocket.send(json.dumps(event))

        async with websockets.connect(self.url, extra_headers=headers) as websocket:
            print("Connected to OpenAI Realtime API")

            # Set up the session
            session_update_event = {
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "instructions": 'listen the audio, output the audio content with format {"content": ""}',
                    "tools": [],
                    "voice": "alloy",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 200,
                    },
                },
            }
            await send_event(websocket, session_update_event)

            audio_files = [audio_path]
            await stream_audio_files(websocket, audio_files)

            audio_buffer = bytearray()
            try:
                while True:
                    message = await websocket.recv()
                    event = json.loads(message)
                    logger.debug("Received event:", event["type"])

                    if event["type"] == "response.text.done":
                        logger.debug("Assistant:", event["text"])
                        return event["text"]
                    elif event["type"] == "error":
                        print("Error:", event["error"])
                        logger.debug(event["error"], "error_log.json")
                    elif event["type"] == "input_audio_buffer.speech_started":
                        print("Speech started")
                    elif event["type"] == "input_audio_buffer.speech_stopped":
                        print("Speech stopped")
                    elif event["type"] == "input_audio_buffer.committed":
                        print("Audio buffer committed")
                    elif event["type"] == "response.audio.delta":
                        audio_content = base64.b64decode(event["delta"])
                        audio_buffer.extend(audio_content)
                        print(
                            f"🔵 Received {len(audio_content)} bytes, total buffer size: {len(audio_buffer)}"
                        )
                    elif event["type"] == "response.audio.done":
                        # 创建WAV文件
                        with open("assistant_response.wav", "wb") as wav_file:
                            audio_array = np.frombuffer(audio_buffer, dtype=np.int16)
                            sf.write(wav_file, audio_array, samplerate=24000)
                        audio_buffer.clear()
                        print("🔵 AI finished speaking.")

            except websockets.exceptions.ConnectionClosed as e:
                print(f"{e} Disconnected from OpenAI Realtime API")

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio, text = None, None
        for line in prompt[0]["contents"]:
            if line["type"] == "audio":
                audio = line["value"]
            if line["type"] == "text":
                text = line["value"]
