from pathlib import Path

from locust import HttpUser, between, task


class MyUser(HttpUser):
    """
    Minimal Locust user for repeatedly hitting `/v1/audio/transcriptions`.
    The goal is to measure server-side performance, so we avoid any extra work
    (file writes, random generation, manual timing, custom event hooks, etc.)
    that could inflate client-side latency.
    """

    wait_time = between(0.5, 1)
    host = "http://0.0.0.0:8090"

    def on_start(self):
        self.api_key = "sk-1234"
        self.model_name = "fake-openai-transcription"
        self.audio_file_path = Path(__file__).resolve().parent / "speech.mp3"
        self.audio_file_bytes = self.audio_file_path.read_bytes()
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        self.prompt_counter = 0

    @task
    def audio_speech_request(self):
        self.prompt_counter += 1
        # Ensure prompts differ slightly so the backend can't reuse cached audio.
        prompt = (
            "Generate a short spoken status update mentioning counter "
            f"{self.prompt_counter}."
        )

        files = {
            "file": (self.audio_file_path.name, self.audio_file_bytes, "audio/mpeg"),
        }
        data = {
            "model": self.model_name,
            "prompt": prompt,
        }

        response = self.client.post(
            "v1/audio/transcriptions",
            data=data,
            files=files,
            headers=self.headers,
            name="audio_transcriptions",
        )

        if response.status_code != 200:
            # log the errors in error.txt
            with open("error.txt", "a") as error_log:
                error_log.write(response.text + "\n")