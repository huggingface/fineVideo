import boto3
import subprocess
import os
import math

# Initialize the S3 client
s3 = boto3.client('s3')

def download_video_from_s3(bucket, path, video_id):
    """Download a video from S3 given a video ID."""
    # Safely handle filenames with hyphens and special characters
    video_file = f"./{video_id}.mp4"
    s3_key = f"{path}/{video_id}.mp4"
    try:
        s3.download_file(bucket, s3_key, video_file)
        print(f"Downloaded {video_file} from s3://{bucket}/{s3_key}")
        return video_file
    except Exception as e:
        print(f"Error downloading {video_file}: {e}")
        return None

def check_static_video(video_file, segment_duration=60, freeze_n=0.05, freeze_d=50, threshold=0.4):
    """Use ffmpeg freezedetect to check if a video has significant static content."""
    
    # Get video duration using ffprobe
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_file],
            capture_output=True, text=True
        )
        video_duration = float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting video duration for {video_file}: {e}")
        return None

    # Calculate the number of segments to analyze
    num_segments = math.ceil(video_duration / segment_duration)
    freeze_count = 0

    # Analyze video in segments
    for start_time in range(0, int(video_duration), segment_duration):
        try:
            command = [
                "ffmpeg", "-hide_banner", "-ss", str(start_time), "-i", video_file, 
                "-t", str(segment_duration), "-vf", f"freezedetect=n={freeze_n}:d={freeze_d}", "-an", "-f", "null", "-"
            ]
            result = subprocess.run(command, capture_output=True, text=True)

            # Check the stderr output for freeze detection
            if "freezedetect" in result.stderr:
                print(f"Static content detected in segment starting at {start_time} of {video_file}.")
                freeze_count += 1
        except Exception as e:
            print(f"Error processing segment starting at {start_time} of {video_file}: {e}")
            return None

    # Calculate the percentage of segments with freezes
    freeze_percentage = freeze_count / num_segments

    print(f"Freeze percentage for {video_file}: {freeze_percentage:.2%}")

    # Determine if the video is considered static based on threshold
    return freeze_percentage >= threshold

def upload_result_to_s3(bucket, video_id, is_static):
    """Upload the result to S3 based on whether the video is static or dynamic."""
    s3_key = f"{'static' if is_static else 'dynamic'}/{video_id}.txt"
    try:
        s3.put_object(Bucket=bucket, Key=s3_key, Body="")
        print(f"Uploaded result to s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"Error uploading result for {video_id}: {e}")

def main():
    # Environment variables set in AWS Batch
    bucket = os.environ.get("VIDEO_BUCKET")
    video_ids = os.environ.get("VIDEO_IDS").split(",")
    video_path = os.environ.get("BUCKET_VIDEO_FOLDER_PATH")
    
    for video_id in video_ids:
        # Download video from S3
        video_file = download_video_from_s3(bucket, video_path, video_id)
        if not video_file:
            continue

        # Check if the video is static
        is_static = check_static_video(video_file)
        if is_static is None:
            continue

        # Upload result to S3
        upload_result_to_s3(bucket, video_id, is_static)
        
        # Clean up downloaded video file
        os.remove(video_file)

if __name__ == "__main__":
    main()
