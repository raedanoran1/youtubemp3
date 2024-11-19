from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from moviepy.editor import AudioFileClip
import os
import shutil
import traceback
from datetime import datetime, timedelta
import logging
from pydub import AudioSegment
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Yeni import
import yt_dlp

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Temizleme işlemi için son indirme zamanını takip etmek için global değişken
last_download_time = None

def parallel_cleanup_file(file_info):
    """Tek bir dosyayı temizle"""
    try:
        file_path, current_time = file_info
        file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
        if current_time - file_modified > timedelta(hours=24):
            os.remove(file_path)
            logger.info(f"Dosya silindi: {file_path}")
    except Exception as e:
        logger.error(f"Dosya silme hatası: {str(e)}")

def cleanup_downloads():
    """24 saatten eski dosyaları paralel olarak temizle"""
    download_path = 'static/downloads'
    if os.path.exists(download_path):
        current_time = datetime.now()
        # Tüm dosya yollarını ve şu anki zamanı içeren tuple listesi oluştur
        file_list = [(os.path.join(download_path, filename), current_time) 
                    for filename in os.listdir(download_path)]
        
        # İşlemci sayısının yarısını kullan
        num_processes = max(1, multiprocessing.cpu_count() // 2)
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            executor.map(parallel_cleanup_file, file_list)

def validate_youtube_url(url):
    """YouTube URL'sini doğrula"""
    if not url:
        logger.error("URL boş")
        return False
    
    valid_hosts = ['youtube.com', 'www.youtube.com', 'youtu.be', 'm.youtube.com']
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        logger.info(f"Parsing URL: {url}")
        logger.info(f"Parsed netloc: {parsed.netloc}")
        
        # URL'nin bir şema (http/https) içerdiğinden emin olun
        if not parsed.scheme:
            url = 'https://' + url
            parsed = urlparse(url)
        
        is_valid = any(host in parsed.netloc for host in valid_hosts)
        if not is_valid:
            logger.error(f"Geçersiz host: {parsed.netloc}")
        return is_valid
    except Exception as e:
        logger.error(f"URL doğrulama hatası: {str(e)}")
        return False

def convert_to_432hz(input_path, output_path):
    """MP3 dosyasını 432 Hz'e dönüştür"""
    try:
        # MP3 dosyasını yükle
        audio = AudioSegment.from_mp3(input_path)
        
        # Mevcut frekansı 440 Hz kabul edip, 432 Hz'e dönüştürmek için gerekli oranı hesapla
        ratio = 432.0 / 440.0
        
        # Yeni sample rate hesapla
        new_sample_rate = int(audio.frame_rate * ratio)
        
        # Sample rate'i değiştir
        converted_audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": new_sample_rate
        })
        
        # Orijinal sample rate'e geri dönüştür
        converted_audio = converted_audio.set_frame_rate(audio.frame_rate)
        
        # Yeni dosyayı kaydet
        converted_audio.export(output_path, format="mp3", bitrate="192k")
        
        return True
    except Exception as e:
        logger.error(f"432 Hz dönüşüm hatası: {str(e)}")
        return False

def download_audio(youtube_url, downloads_dir, ydl_opts):
    """Ses indirme işlemi"""
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        return info_dict

def process_audio(info_dict, downloads_dir):
    """İndirilen sesi işle ve 432 Hz'e dönüştür"""
    video_title = info_dict['title']
    video_id = info_dict['id']
    
    # Güvenli dosya adı oluştur
    safe_title = "".join(x for x in video_title if x.isalnum() or x == ' ').strip()
    safe_title = safe_title[:50]
    final_filename = f"{safe_title}_{video_id}.mp3"
    final_path = os.path.join(downloads_dir, final_filename)
    
    # İndirilen dosyayı bul
    for file in os.listdir(downloads_dir):
        if file.endswith('.mp3') and video_id in file:
            old_path = os.path.join(downloads_dir, file)
            if old_path != final_path:
                try:
                    os.rename(old_path, final_path)
                except OSError:
                    final_path = old_path
            break
    
    # 432 Hz'e dönüştür
    hz432_filename = f"{safe_title}_{video_id}_432hz.mp3"
    hz432_path = os.path.join(downloads_dir, hz432_filename)
    
    if convert_to_432hz(final_path, hz432_path):
        logger.info(f"432 Hz dönüşümü başarılı: {hz432_path}")
        os.remove(final_path)
        return hz432_path
    return final_path

def download_with_ytdlp(youtube_url, output_path):
    """yt-dlp ile video indirme ve işleme - paralel versiyon"""
    try:
        downloads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'downloads')
        os.makedirs(downloads_dir, exist_ok=True)
        
        safe_template = '%(title).50s_%(id)s.%(ext)s'
        filename_template = os.path.join(downloads_dir, safe_template)
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': filename_template,
            'prefer_ffmpeg': True,
            'keepvideo': False,
            'quiet': False,
            'no_warnings': False
        }

        # ThreadPoolExecutor kullanarak indirme ve işleme işlemlerini paralel yap
        with ThreadPoolExecutor(max_workers=2) as executor:
            # İndirme işlemini başlat
            future_download = executor.submit(download_audio, youtube_url, downloads_dir, ydl_opts)
            info_dict = future_download.result()
            
            # Ses işleme işlemini başlat
            future_process = executor.submit(process_audio, info_dict, downloads_dir)
            final_path = future_process.result()

            if not os.path.exists(final_path):
                raise Exception("Dosya indirme işlemi başarısız oldu")
            
            logger.info(f"Dosya başarıyla indirildi ve işlendi: {final_path}")
            return final_path
            
    except Exception as e:
        logger.error(f"İndirme hatası: {str(e)}")
        traceback.print_exc()
        raise e

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    try:
        data = request.get_json()
        if not data:
            logger.error("JSON verisi alınamadı")
            return jsonify({'error': 'JSON verisi gerekli'}), 400

        youtube_url = data.get('youtube_url')
        logger.info(f"Gelen YouTube URL: {youtube_url}")

        if not youtube_url:
            logger.error("YouTube URL eksik")
            return jsonify({'error': 'YouTube URL gerekli'}), 400

        if not validate_youtube_url(youtube_url):
            logger.error(f"Geçersiz YouTube URL: {youtube_url}")
            return jsonify({'error': 'Geçersiz YouTube URL formatı'}), 400

        # Temizleme işlemini kontrol et
        cleanup_downloads()

        try:
            # Download with yt-dlp and get the file path
            logger.info(f"Video indirme başlıyor: {youtube_url}")
            downloaded_file = download_with_ytdlp(youtube_url, 'static/downloads')
            
            if not os.path.exists(downloaded_file):
                logger.error(f"İndirilen dosya bulunamadı: {downloaded_file}")
                return jsonify({'error': 'Dosya indirme işlemi başarısız oldu'}), 500

            # Get just the filename from the path
            filename = os.path.basename(downloaded_file)
            logger.info(f"Dönüştürme başarılı. Dosya: {filename}")
            
            return jsonify({
                'success': True,
                'message': 'Dönüştürme başarılı',
                'filename': filename
            })

        except Exception as e:
            error_msg = f"Dönüştürme hatası: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return jsonify({'error': error_msg}), 500

    except Exception as e:
        error_msg = f"Genel hata: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/download/<filename>')
def download(filename):
    try:
        # Güvenlik kontrolü: filename'den zararlı karakterleri temizle
        filename = os.path.basename(filename)
        file_path = os.path.join('static', 'downloads', filename)

        if not os.path.exists(file_path):
            logger.error(f"Dosya bulunamadı: {file_path}")
            return jsonify({
                'error': 'Dosya bulunamadı. Lütfen önce dönüştürme işlemini yapın.',
                'file_path': file_path,
                'exists': False
            }), 404

        try:
            return send_file(
                file_path,
                as_attachment=True,
                download_name=filename,
                mimetype='audio/mpeg'
            )
        except Exception as e:
            logger.error(f"Dosya gönderme hatası: {str(e)}")
            return jsonify({'error': 'Dosya gönderme hatası'}), 500

    except Exception as e:
        logger.error(f"Download hatası: {str(e)}")
        return jsonify({'error': f'Dosya indirme hatası: {str(e)}'}), 500

if __name__ == '__main__':
    os.makedirs('static/downloads', exist_ok=True)
    cleanup_downloads()  # Başlangıçta eski dosyaları temizle
    app.run(debug=True)
