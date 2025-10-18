import yt_dlp
import os
import shutil
def download_audio(youtube_url, config):
    """Download audio from YouTube and convert to compatible format"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        filename = ydl.prepare_filename(info)
        base, ext = os.path.splitext(filename)
        wav_file = base + '.wav'
        save_path=os.path.join(config["data"]["main_dir"],"audio")
        os.makedirs(save_path, exist_ok=True)
        print(f"Как назвать .wav файл для обучения? (Например, {config['name']}_pope)")
        name=str(input()).strip()
        savefilepath=os.path.join(save_path, name if name.endswith(".wav") else name+'.wav')
        shutil.move(wav_file, savefilepath)
        print(f"Файл '{savefilepath}' готов для начала обучения!")
        return savefilepath