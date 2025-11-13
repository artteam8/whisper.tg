import numpy as np
import librosa
from pydub import AudioSegment
import whisper
import logging
from io import BytesIO
from aiogram import Bot, Dispatcher, Router
from aiogram import F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, Message
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

session = AiohttpSession(
    api=TelegramAPIServer.from_base('http://localhost:8081')
)

API_TOKEN = os.getenv("API_TOKEN")
bot = Bot(token=API_TOKEN, session=session)
dp = Dispatcher()
router = Router()
dp.include_router(router)

model = whisper.load_model("turbo")

@router.message(Command("start"))
async def send_welcome(message: Message):
    await message.answer(
        "🤖 Whisper is ready!\n\n"
        "Send me an audio file or a voice message.\n"
        "You can cut the audio file by adding a caption in the file message, like that: 0:13 4:07."
    )

async def file_to_np(file_id: str, bot: Bot, ftype: str) -> np.array:
    file = await bot.get_file(file_id)
    file_path = file.file_path
    print(file.file_path)

    #downloaded_file_buffer = BytesIO()
    #await bot.download_file(file_path, downloaded_file_buffer)
    #downloaded_file_buffer.seek(0)
    with open(file_path, "rb") as file:
        file_bytes = file.read()

    downloaded_file_buffer = BytesIO(file_bytes)

    # converting any audio format to librosa compatible (wav)
    audio = AudioSegment.from_file(downloaded_file_buffer, format=ftype)
    wav_buffer = BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)

    # converting audio to numpy array
    audio_array, sr = librosa.load(wav_buffer, sr=16000, mono=True)

    # closing buffers
    downloaded_file_buffer.close()
    wav_buffer.close()
    return audio_array, sr

def transcribe(audio_array: np.array, sr: int, start: float, end: float, ftype: str) -> str:
    # cutting
    if start is not None and end is not None:
        audio_array = audio_array[int(start * sr):int(end * sr)]
    text = model.transcribe(audio_array, fp16=False)
    return text['text']

def norm_int(num: str) -> str:
    while num[0] == '0':
        num = num[1:]
    return num

@router.message(F.content_type.in_({"audio"}))
async def handle_audio(message: Message):
    try:
        audio = message.audio

        if message.caption:
            start, end = message.caption.split()
            start_minutes, start_seconds = start.split(":")
            end_minutes, end_seconds = end.split(":")

            start_minutes = norm_int(start_minutes)
            end_minutes = norm_int(end_minutes)
            start_seconds = norm_int(start_seconds)
            end_seconds = norm_int(end_seconds)

            start = float(start_minutes) * 60 + float(start_seconds)
            end = float(end_minutes) * 60 + float(end_seconds)
        else:
            start = None
            end = None
        
        await message.answer("🎵 Audio file received! Processing...")

        filename = message.audio.file_name.lower()
        ftype = filename.split(".")[-1]

        audio_array, sr = await file_to_np(audio.file_id, bot, ftype)
        text = transcribe(audio_array, sr, start, end, ftype)

        stream = BytesIO()
        stream.write(text.encode('utf-8'))
        stream.seek(0)
        file_bytes = stream.getvalue()
        filename = filename.replace(f".{ftype}", ".txt")
        doc = BufferedInputFile(file_bytes, filename=filename)
        stream.close()
        await message.answer_document(doc)

    except Exception as e:
        logging.error(f"Error processing audio: {e}")
        await message.answer(f"❌ Sorry, an error occurred while processing your audio file:\n {e}")

@router.message(F.content_type.in_({"voice"}))
async def handle_audio(message: Message):
    try:
        await message.answer("🎵 Voice message received! Processing...")
        audio = message.voice
        
        start = None
        end = None

        try:
            audio_array, sr = await file_to_np(audio.file_id, bot, "ogg")
            text = transcribe(audio_array, sr, start, end, "ogg")
        except Exception as e:
            audio_array, sr = await file_to_np(audio.file_id, bot, "m4a")
            text = transcribe(audio_array, sr, start, end, "m4a")

        stream = BytesIO()
        stream.write(text.encode('utf-8'))
        stream.seek(0)
        file_bytes = stream.getvalue()
        doc = BufferedInputFile(file_bytes, filename = "transcribation.txt")
        stream.close()
        await message.answer_document(doc)
    except Exception as e:
        logging.error(f"Error processing audio: {e}")
        await message.answer(f"❌ Sorry, an error occurred while processing your audio file:\n {e}")

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
