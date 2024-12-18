import os
import re
import logging
import traceback
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yt_dlp
from urllib.parse import urlparse, parse_qs

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def extract_video_id(url):
    """YouTube URL'sinden video ID'sini çıkarır"""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.path.startswith(('/embed/', '/v/')):
            return parsed_url.path.split('/')[2]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    return None

def download_with_ytdlp(youtube_url, output_path):
    """yt-dlp ile video indirme"""
    try:
        # Video ID'sini kontrol et
        video_id = extract_video_id(youtube_url)
        if not video_id:
            raise Exception("Geçersiz YouTube URL'si")

        downloads_dir = output_path
        os.makedirs(downloads_dir, exist_ok=True)

        # Önce video bilgilerini al
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'force_generic_extractor': False,
            'nocheckcertificate': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.youtube.com/',
                'Origin': 'https://www.youtube.com',
            }
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(youtube_url, download=False)
                if not info:
                    raise Exception("Video bilgileri alınamadı")
                
                video_title = info.get('title', 'video')
                logger.info(f"Video bilgileri alındı: {video_title}")
            except Exception as e:
                logger.error(f"Video bilgileri alınamadı: {str(e)}")
                raise Exception("Video bilgileri alınamadı")

        # Güvenli dosya adı oluştur
        safe_title = re.sub(r'[^\w\s-]', '', video_title)
        safe_filename = f"{safe_title}_{video_id}"
        filename_template = os.path.join(downloads_dir, safe_filename + '.%(ext)s')

        # İndirme seçeneklerini ayarla
        download_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': filename_template,
            'quiet': True,
            'no_warnings': True,
            'nocheckcertificate': True,
            'geo_bypass': True,
            'extractor_retries': 3,
            'socket_timeout': 30,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.youtube.com/',
                'Origin': 'https://www.youtube.com',
            },
            'youtube_include_dash_manifest': False,
            'prefer_insecure': True,
            'no_check_certificates': True
        }

        with yt_dlp.YoutubeDL(download_opts) as ydl:
            logger.info(f"Video indirme başlıyor: {youtube_url}")
            ydl.download([youtube_url])
            
            # İndirilen dosyayı bul
            mp3_filename = os.path.join(downloads_dir, safe_filename + '.mp3')
            if not os.path.exists(mp3_filename):
                raise Exception("Dosya indirme işlemi başarısız oldu")
            
            logger.info(f"Dosya başarıyla indirildi: {mp3_filename}")
            return {'filename': os.path.basename(mp3_filename)}

    except Exception as e:
        logger.error(f"İndirme hatası: {str(e)}")
        logger.error(f"Tam hata: {traceback.format_exc()}")
        raise Exception(f"Video indirme hatası: {str(e)}")

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    try:
        data = request.get_json()
        youtube_url = data.get('url')
        
        if not youtube_url:
            return jsonify({'error': 'YouTube URL\'si gerekli'}), 400
            
        # Geçici dizin oluştur
        temp_dir = os.path.join('/tmp', 'youtube_downloads')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Video indir
        result = download_with_ytdlp(youtube_url, temp_dir)
        
        if not result or 'filename' not in result:
            return jsonify({'error': 'Video indirilemedi'}), 500
            
        # Dosya yolunu oluştur
        file_path = os.path.join(temp_dir, result['filename'])
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'MP3 dosyası oluşturulamadı'}), 500
            
        return send_file(
            file_path,
            as_attachment=True,
            download_name=result['filename'],
            mimetype='audio/mpeg'
        )
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Dönüştürme hatası: {error_message}")
        return jsonify({'error': f"Video işleme hatası: {error_message}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
