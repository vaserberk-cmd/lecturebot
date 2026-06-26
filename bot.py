import os
import asyncio
import tempfile
import subprocess
import shutil
from pathlib import Path
from groq import Groq
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Устанавливаем ffmpeg если не найден
if not shutil.which("ffmpeg"):
    print("ffmpeg не найден, устанавливаю...")
    subprocess.run(["apt-get", "update", "-y"], check=False)
    subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=False)
    print("ffmpeg установлен!")

BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)

TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

MAX_SEGMENT_SEC = 800  # ~13 минут с запасом


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для перевода лекций в текст.\n\n"
        "Отправь голосовое сообщение или аудиофайл (MP3, OGG, M4A).\n"
        "Лекции до 1.5 часов обрабатываются полностью!\n\n"
        "Использую Whisper AI — высокое качество распознавания."
    )


def get_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def convert_to_mp3(input_path: str, output_path: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-ar", "16000", "-ac", "1", "-b:a", "128k", output_path],
        capture_output=True, check=True
    )


def split_audio(input_path: str, duration: float) -> list:
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
        chunks.append((out_path, start))  # путь + смещение времени
        start += MAX_SEGMENT_SEC
        idx += 1
    return chunks


def seconds_to_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe_chunk(chunk_path: str, idx: int, time_offset: float = 0.0) -> str:
    """Отправляет кусок аудио в Groq Whisper с временными метками."""
    with open(chunk_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(f"chunk_{idx}.mp3", f.read()),
            model="whisper-large-v3",
            language="ru",
            response_format="verbose_json",
        )

    lines = []
    if hasattr(result, "segments") and result.segments:
        for seg in result.segments:
            start = seg.start + time_offset
            text = seg.text.strip()
            if text:
                ts = seconds_to_timestamp(start)
                text = text[0].upper() + text[1:]
                lines.append(f"{ts} - {text}")
    else:
        # fallback если нет сегментов
        text = result.text.strip() if hasattr(result, "text") else str(result)
        if text:
            ts = seconds_to_timestamp(time_offset)
            lines.append(f"{ts} - {text}")

    return "\n\n".join(lines)


async def send_long_text(message, text: str):
    """Отправляет длинный текст частями, не разрывая блоки с метками."""
    MAX_LEN = 4000
    blocks = text.split("\n\n")
    parts = []
    current = ""

    for block in blocks:
        if len(current) + len(block) + 2 > MAX_LEN and current:
            parts.append(current.strip())
            current = block + "\n\n"
        else:
            current += block + "\n\n"

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
    chunk_paths = []

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

        if duration > MAX_SEGMENT_SEC:
            chunks = await loop.run_in_executor(None, split_audio, mp3_path, duration)
        else:
            chunks = [(mp3_path, 0.0)]

        chunk_paths = [c[0] for c in chunks if c[0] != mp3_path]
        total_chunks = len(chunks)

        await status_msg.edit_text(
            f"Длительность: {duration_min:.1f} мин\n"
            f"Частей: {total_chunks}\n"
            f"Распознаю через Whisper AI..."
        )

        all_text = []
        for i, (chunk_path, time_offset) in enumerate(chunks, 1):
            await status_msg.edit_text(
                f"Распознаю часть {i}/{total_chunks}...\n"
                f"{'█' * i}{'░' * (total_chunks - i)} {int(i/total_chunks*100)}%"
            )
            text = await loop.run_in_executor(
                None, transcribe_chunk, chunk_path, i, time_offset
            )
            if text:
                all_text.append(text.strip())

        full_text = "\n\n".join(all_text).strip()

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
        for path in ([input_path] + chunk_paths):
            if path and os.path.exists(path):
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
