import os
import asyncio
import tempfile
import subprocess
from pathlib import Path
from groq import Groq
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)

TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

# Groq Whisper лимит — 25 МБ, ~15 минут MP3 128kbps
MAX_SEGMENT_SEC = 800  # ~13 минут с запасом


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для перевода лекций в текст.\n\n"
        "Отправь голосовое сообщение или аудиофайл (MP3, OGG, M4A).\n"
        "Лекции до 1.5 часов обрабатываются полностью!\n\n"
        "Использую Whisper AI — высокое качество распознавания."
    )


def get_duration(path: str) -> float:
    """Получает длительность аудио через ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def convert_to_mp3(input_path: str, output_path: str):
    """Конвертирует любой аудиофайл в MP3 128kbps через ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-ar", "16000", "-ac", "1", "-b:a", "128k", output_path],
        capture_output=True, check=True
    )


def split_audio(input_path: str, duration: float) -> list[str]:
    """Разбивает MP3 на куски по MAX_SEGMENT_SEC секунд."""
    chunks = []
    start = 0
    idx = 0
    while start < duration:
        out_path = input_path + f"_chunk{idx}.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-ss", str(start), "-t", str(MAX_SEGMENT_SEC),
             "-c", "copy", out_path],
            capture_output=True, check=True
        )
        chunks.append(out_path)
        start += MAX_SEGMENT_SEC
        idx += 1
    return chunks


def transcribe_chunk(chunk_path: str, idx: int) -> str:
    """Отправляет кусок аудио в Groq Whisper."""
    with open(chunk_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(f"chunk_{idx}.mp3", f.read()),
            model="whisper-large-v3",
            language="ru",
            response_format="text",
        )
    return result if isinstance(result, str) else result.text


async def send_long_text(message, text: str):
    """Отправляет длинный текст частями по 4000 символов."""
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
    mp3_path = None
    chunks = []

    try:
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

        tmp = tempfile.NamedTemporaryFile(
            suffix=suffix, dir=str(TEMP_DIR), delete=False
        )
        input_path = tmp.name
        tmp.close()
        await audio_file.download_to_drive(input_path)

        await status_msg.edit_text("Конвертирую в MP3...")

        loop = asyncio.get_event_loop()
        mp3_path = input_path + ".mp3"
        await loop.run_in_executor(None, convert_to_mp3, input_path, mp3_path)

        duration = await loop.run_in_executor(None, get_duration, mp3_path)
        duration_min = duration / 60

        # Разбиваем если длиннее одного сегмента
        if duration > MAX_SEGMENT_SEC:
            chunks = await loop.run_in_executor(None, split_audio, mp3_path, duration)
        else:
            chunks = [mp3_path]

        total_chunks = len(chunks)
        await status_msg.edit_text(
            f"Длительность: {duration_min:.1f} мин\n"
            f"Частей: {total_chunks}\n"
            f"Распознаю через Whisper AI..."
        )

        all_text = []
        for i, chunk_path in enumerate(chunks, 1):
            await status_msg.edit_text(
                f"Распознаю часть {i}/{total_chunks}...\n"
                f"{'█' * i}{'░' * (total_chunks - i)} {int(i/total_chunks*100)}%"
            )
            text = await loop.run_in_executor(None, transcribe_chunk, chunk_path, i)
            if text:
                all_text.append(text.strip())

        full_text = " ".join(all_text).strip()

        if full_text:
            await status_msg.edit_text(f"Готово! Распознано {len(full_text)} символов.")
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
        for path in ([input_path, mp3_path] + chunks):
            if path and os.path.exists(path) and path != mp3_path or (path == mp3_path and chunks != [mp3_path]):
                try:
                    os.unlink(path)
                except Exception:
                    pass
        if mp3_path and os.path.exists(mp3_path):
            try:
                os.unlink(mp3_path)
            except Exception:
                pass


def main():
    print("Запуск бота...")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, process_audio))
    print("Бот запущен!")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
