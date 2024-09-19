import os
import sys
import boto3
from yt_dlp import YoutubeDL

def download_youtube_video(video_id, output_path):
    ydl_opts = {
        'format': 'best',
        'writesubtitles': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        'writeinfojson': True,
        'skip_download': False,
        'outtmpl': os.path.join(output_path, f'{video_id}.%(ext)s'),
    }
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_id, download=True)
        
        # Get the correct subtitle file path from the info_dict
        subtitle_file_path = None
        subtitles = info_dict.get('subtitles')
        if subtitles and 'en' in subtitles:
            subtitle_data = subtitles['en'][0]  # Get the first English subtitle entry
            subtitle_file_path = ydl.prepare_filename(info_dict).replace('.mp4', '.en.vtt')
        
        return info_dict, subtitle_file_path

def upload_to_s3(local_file_path, s3_bucket, s3_key):
    s3_client = boto3.client('s3')
    s3_client.upload_file(local_file_path, s3_bucket, s3_key)

def log_failure(video_id, error_message, s3_bucket, s3_path):
    error_file_path = f"/tmp/{video_id}.txt"
    with open(error_file_path, 'w') as f:
        f.write(error_message)
    
    # Upload the error file to S3 in the failed/ subfolder
    s3_client = boto3.client('s3')
    s3_client.upload_file(error_file_path, s3_bucket, f"failed/{video_id}.txt")

def process_video(video_id, s3_bucket, s3_path):
    try:
        # Create a temporary directory to store downloaded files
        download_path = '/tmp/youtube_downloads'
        os.makedirs(download_path, exist_ok=True)

        # Download the video, subtitles (if available), and metadata
        info_dict, subtitle_file_path = download_youtube_video(video_id, download_path)

        # Define file paths
        video_file = os.path.join(download_path, f'{video_id}.mp4')
        metadata_file = os.path.join(download_path, f'{video_id}.info.json')

        # Upload each file to the specified S3 path if it exists
        if os.path.exists(video_file):
            upload_to_s3(video_file, s3_bucket, os.path.join(s3_path, f'{video_id}.mp4'))
        if os.path.exists(metadata_file):
            upload_to_s3(metadata_file, s3_bucket, os.path.join(s3_path, f'{video_id}.json'))
        if subtitle_file_path and os.path.exists(subtitle_file_path):
            upload_to_s3(subtitle_file_path, s3_bucket, os.path.join(s3_path, f'{video_id}.en.vtt'))

        # Cleanup
        for file_name in os.listdir(download_path):
            os.remove(os.path.join(download_path, file_name))

    except Exception as e:
        error_message = str(e)
        log_failure(video_id, error_message, s3_bucket, s3_path)

def main(video_ids, s3_bucket, s3_path):
    for video_id in video_ids:
        process_video(video_id, s3_bucket, s3_path)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python download_and_upload.py <s3_bucket_name> <s3_path> <youtube_video_id_1> [<youtube_video_id_2> ...]")
        sys.exit(1)

    s3_bucket = sys.argv[1]
    s3_path = sys.argv[2]
    video_ids = sys.argv[3:]

    main(video_ids, s3_bucket, s3_path)
