import os
import asyncio
import tempfile
import math
from pathlib import Path
from groq import Groq
from pydub import AudioSegment
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)

TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

# Groq Whisper лимит — 25 МБ на файл
MAX_CHUNK_MB = 24
MAX_CHUNK_BYTES = MAX_CHUNK_MB * 1024 * 1024


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для перевода лекций в текст.\n\n"
        "Отправь голосовое сообщение или аудиофайл (MP3, OGG, M4A и др.)\n"
        "Лекции до 1.5 часов обрабатываются полностью.\n\n"
        "Использую Whisper AI — высокое качество распознавания!"
    )


def split_audio(audio: AudioSegment) -> list[AudioSegment]:
    """Разбивает аудио на куски по ~24 МБ (в формате mp3 ~128kbps)."""
    # 128 kbps = 16000 байт/сек
    bytes_per_sec = 16000
    max_duration_ms = int((MAX_CHUNK_BYTES / bytes_per_sec) * 1000)

    chunks = []
    total_ms = len(audio)
    start = 0
    while start < total_ms:
        end = min(start + max_duration_ms, total_ms)
        chunks.append(audio[start:end])
        start = end
    return chunks


def transcribe_chunk(chunk: AudioSegment, chunk_index: int) -> str:
    """Отправляет один кусок аудио в Groq Whisper."""
    with tempfile.NamedTemporaryFile(
        suffix=".mp3", dir=str(TEMP_DIR), delete=False
    ) as tmp:
        tmp_path = tmp.name

    try:
        chunk.export(tmp_path, format="mp3", bitrate="128k")
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(f"chunk_{chunk_index}.mp3", f.read()),
                model="whisper-large-v3",
                language="ru",
                response_format="text",
            )
        return result if isinstance(result, str) else result.text
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def send_long_text(message, text: str):
    """Отправляет длинный текст частями."""
    MAX_LEN = 4000
    parts = []
    current = ""

    for sentence in text.split(". "):
        chunk = sentence + ". "
        if len(current) + len(chunk) > MAX_LEN:
            if current:
                parts.append(current.strip())
            current = chunk
        else:
            current += chunk

    if current.strip():
        parts.append(current.strip())

    if not parts:
        parts = [text[i:i+MAX_LEN] for i in range(0, len(text), MAX_LEN)]

    total = len(parts)
    for i, part in enumerate(parts, 1):
        prefix = f"Часть {i}/{total}:\n\n" if total > 1 else ""
        await message.reply_text(prefix + part)
        await asyncio.sleep(0.3)


async def process_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    input_path = None

    try:
        # Определяем файл
        if message.voice:
            audio_file = await message.voice.get_file()
            suffix = ".ogg"
        elif message.audio:
            audio_file = await message.audio.get_file()
            name = message.audio.file_name or "audio.mp3"
            suffix = os.path.splitext(name)[1] or ".mp3"
        else:
            await message.reply_text("Отправь голосовое или аудиофайл.")
            return

        status_msg = await message.reply_text("Скачиваю аудиофайл...")

        # Скачиваем
        tmp = tempfile.NamedTemporaryFile(
            suffix=suffix, dir=str(TEMP_DIR), delete=False
        )
        input_path = tmp.name
        tmp.close()
        await audio_file.download_to_drive(input_path)

        await status_msg.edit_text("Конвертирую аудио...")

        # Конвертируем
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None, lambda: AudioSegment.from_file(input_path)
        )

        duration_min = len(audio) / 1000 / 60
        chunks = split_audio(audio)
        total_chunks = len(chunks)

        await status_msg.edit_text(
            f"Длительность: {duration_min:.1f} мин\n"
            f"Частей для обработки: {total_chunks}\n"
            f"Начинаю распознавание через Whisper AI..."
        )

        # Распознаём по кускам
        all_text = []
        for i, chunk in enumerate(chunks, 1):
            await status_msg.edit_text(
                f"Распознаю часть {i}/{total_chunks}...\n"
                f"{'█' * i}{'░' * (total_chunks - i)} {int(i/total_chunks*100)}%"
            )
            text = await loop.run_in_executor(
                None, transcribe_chunk, chunk, i
            )
            if text:
                all_text.append(text.strip())

        full_text = " ".join(all_text).strip()

        if full_text:
            await status_msg.edit_text(
                f"Готово! Распознано {len(full_text)} символов."
            )
            await send_long_text(message, full_text)
        else:
            await status_msg.edit_text(
                "Не удалось распознать речь.\n"
                "Проверь качество записи."
            )

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        try:
            await message.reply_text(f"Ошибка: {str(e)}")
        except Exception:
            pass
    finally:
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)


def main():
    print("Запуск бота...")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, process_audio))
    print("Бот запущен!")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
