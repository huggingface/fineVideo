import json
import os
from typing import List
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
import argparse
import re
import json
import boto3
from datetime import datetime

#
# Given an input list of videos, this script downloads them from S3 and aligns the metadata from those videos generated with video2annotation.py with the videos itself.
#
# The code is prepared to run as a standalone application:
# The first parameter is size_chunk: it basically divide the list of videos in sublists of length size_chunk
# The worker_number decides in which sublist of size size_chunk the current execution will be working on
# --video-list is to specify the json file that contains a list of videoids as a JSON list. If that is not provided, it defaults to video_alignment_to_process.json
#


### CONFIG ###
bucket_name = '<bucket_name>'
video_folder_path = 'videos_minioracle/'
json_folder_path = 'videos_minioracle_results/'
output_folder_path = 'results_minioracle_aligned/'
###

# AWS S3 Configuration - specify your personal profile
session = boto3.Session() 
s3_client = session.client('s3')

# Function to download video from S3
def download_video_from_s3(video_key, local_path):
    try:
        s3_client.download_file(bucket_name, video_key, local_path)
        print(f"Downloaded {video_key} to {local_path}")
        return True
    except Exception as e:
        print(f"Failed to download {video_key} from S3: {e}")
        return False


def handle_error(video_id: str, error_message: str, output_folder_path: str, worker_number: str):
    """Handle errors by creating an error file and updating the status report."""
    error_data = {
        "error": error_message,
        "video_id": video_id,
        "worker_number": worker_number
    }
    error_file_path = os.path.join(output_folder_path, f"errors_{video_id}.json")
    with open(error_file_path, "w") as f:
        json.dump(error_data, f, indent=4)

    # Update status report for failure
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    status_report = f"{timestamp} - {video_id} - failed - {error_message}\n"
    print(status_report)
    with open(f"status/status_alignment_{worker_number}.txt", "a") as f:
        f.write(status_report)


def time_to_frametimecode(time_str: str, fps: float, scene_end_time: FrameTimecode = None, filename: str = "unknown_file", worker_number: str = None) -> str:
    """Convert mm:ss or ss time format to FrameTimecode, or handle special cases like 'end'."""
    # Define special cases
    if time_str == "end":
        if scene_end_time is not None:
            return scene_end_time.get_timecode()
        else:
            raise ValueError("time_str is end and no replacement for scene_end_time provided")

    special_cases = ["", "n/a", "varies", "throughout scene", "throughout the scene", 
                     "end", "throughout", "not present", "not applicable"]
    if time_str.lower() in special_cases or re.match(r"scene\s\d+", time_str.lower()):
        return None

    match = re.match(r"(\d+)s$", time_str.lower())
    if match:
        time_str = match.group(1)
    if 'around ' in time_str:
        time_str = time_str.split('around ')[0]
    if '~' in time_str:
        time_str = time_str.split('~')[0]
    if '+' in time_str:
        time_str = time_str.split('+')[0]
    if '-' in time_str:
        time_str = time_str.split("-")[0]
    if ' ' in time_str and ":" in time_str:
        time_str = time_str.split(" ")[0]
    if ":" in time_str:
        parts = time_str.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = parts
        elif len(parts) == 1:
            hours = 0
            minutes = 0
            seconds = parts[0]
        else:
            raise ValueError(f"Invalid timestamp format: {time_str}")

        if '.' in seconds:
            seconds = seconds.split(".")[0]

        match = re.match(r"^\d+", seconds)
        if match:
            seconds = int(match.group())
        else:
            raise ValueError(f"Invalid timestamp format: {time_str}")


        total_seconds = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    else:
        try:
            total_seconds = float(time_str)
        except ValueError:
            raise ValueError(f"Invalid timestamp format: {time_str}")
    return FrameTimecode(timecode=total_seconds, fps=fps).get_timecode()


def adjust_scene_boundaries(video_path, initial_scenes, video_id, worker_number):
    """Adjust scene boundaries based on scene detection."""
    # Initialize video manager and scene manager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=15.0))  # Adjust threshold for sensitivity

    # Start the video manager and obtain FPS
    video_manager.start()
    fps = video_manager.get_framerate()  # Get FPS from VideoManager
    # print(f"Detected FPS: {fps}")
    
    # Get total frames using duration in seconds and fps
    duration_seconds = video_manager.get_duration()[0].get_seconds()
    total_frames = int(duration_seconds * fps)
    last_frame_timecode = FrameTimecode(timecode=total_frames, fps=fps).get_timecode().split(".")[0].split(":")
    last_frame_timecode = last_frame_timecode[1] + ":" + last_frame_timecode[2]

    adjusted_scenes = []

    for idx, initial_scene in enumerate(initial_scenes):

        if idx == len(initial_scenes) - 1:
            #Hack to avoid issues with answers that signal the last timestamp as 'end'
            initial_scene['timestamps']['end_timestamp'] = last_frame_timecode
            # print(last_frame_timecode)
        
        start_timecode = time_to_frametimecode(initial_scene['timestamps']['start_timestamp'], fps, filename=video_id, worker_number = worker_number)
        end_timecode = time_to_frametimecode(initial_scene['timestamps']['end_timestamp'], fps, filename=video_id, worker_number = worker_number)

        # Ensure all FrameTimecode objects use the same fps
        start_frame_number = int(max(0, FrameTimecode(timecode=start_timecode, fps=fps).get_frames() - 2 * fps))
        end_frame_number = int(min(total_frames, FrameTimecode(timecode=end_timecode, fps=fps).get_frames() + 2 * fps))

        search_start = FrameTimecode(timecode=start_frame_number, fps=fps)
        search_end = FrameTimecode(timecode=end_frame_number, fps=fps)

        # Seek to the start frame for detection using FrameTimecode
        video_manager.seek(search_start)
        scene_manager.detect_scenes(frame_source=video_manager, end_time=search_end.get_seconds())

        detected_scenes = scene_manager.get_scene_list()

        # Find closest detected boundaries, default to original timecodes if no match found
        adjusted_start_timecode = start_timecode
        adjusted_end_timecode = end_timecode

        if detected_scenes:
            closest_start = min(detected_scenes, key=lambda x: abs(x[0].get_frames() - FrameTimecode(timecode=start_timecode, fps=fps).get_frames()), default=None)
            closest_end = min(detected_scenes, key=lambda x: abs(x[1].get_frames() - FrameTimecode(timecode=end_timecode, fps=fps).get_frames()), default=None)

            if closest_start and abs(closest_start[0].get_frames() - FrameTimecode(timecode=start_timecode, fps=fps).get_frames()) < 2 * fps:
                adjusted_start_timecode = closest_start[0].get_timecode()
                distance = abs(closest_start[0].get_seconds() - FrameTimecode(timecode=start_timecode, fps=fps).get_seconds())
                if distance > 2:
                    print(f"\t adjusting start timestamp by {distance:.2f} seconds")
                    print(f"\t\tFrom: {start_timecode} to {adjusted_start_timecode}" )
                    if distance >=5:
                        raise ValueError(f"Large start timestamp adjustment ({distance:.2f} seconds) required for scene {idx+1}")

            if closest_end and abs(closest_end[1].get_frames() - FrameTimecode(timecode=end_timecode, fps=fps).get_frames()) < 2 * fps:
                distance = abs(closest_end[1].get_seconds() - FrameTimecode(timecode=end_timecode, fps=fps).get_seconds())
                adjusted_end_timecode = closest_end[1].get_timecode()
                if distance > 2:
                    print(f"\t adjusting end timestamp by {distance:.2f} seconds")
                    print(f"\t\tFrom: {end_timecode} to {adjusted_end_timecode}" )
                    if distance >=5:
                        raise ValueError(f"Large start timestamp adjustment ({distance:.2f} seconds) required for scene {idx+1}")

        # Update the JSON with FrameTimecode formatted as HH:MM:SS:FF
        initial_scene['timestamps']['start_timestamp'] = adjusted_start_timecode
        initial_scene['timestamps']['end_timestamp'] = adjusted_end_timecode

        adjusted_scenes.append(initial_scene)

        # Ensure continuity between scenes
        if idx > 0:
            previous_scene_end = FrameTimecode(timecode=adjusted_scenes[idx - 1]['timestamps']['end_timestamp'], fps=fps)
            current_scene_start = FrameTimecode(timecode=adjusted_start_timecode, fps=fps)
            
            # if current_scene_start.get_frames() <= previous_scene_end.get_frames():
                # Set start of current scene to be exactly the frame after the end of the previous scene
            new_start_timecode = previous_scene_end.get_frames() + 1
            adjusted_scenes[idx]['timestamps']['start_timestamp'] = FrameTimecode(timecode=new_start_timecode, fps=fps).get_timecode()

            frame_adjustment = abs(current_scene_start.get_frames() - new_start_timecode)
            if frame_adjustment > 25:
                print(f"\t\tWARNING: adjusting a scene start by {frame_adjustment} frames")
                if frame_adjustment > 125:
                    raise ValueError(f"Large frame adjustment ({frame_adjustment} frames) required for scene {idx+1}")


    video_manager.release()
    return fps, adjusted_scenes

def update_timestamps_in_json(data: dict, fps: float, video_id: str, worker_number: str) -> dict:
    """Update all timestamp fields in the JSON data to FrameTimecode format and ensure they stay within scene boundaries."""
    # Update timestamps in scenes
    for scene in data.get('scenes', []):
        scene_start = FrameTimecode(timecode=scene['timestamps']['start_timestamp'], fps=fps)
        scene_end = FrameTimecode(timecode=scene['timestamps']['end_timestamp'], fps=fps)
        
        def enforce_within_boundaries(timestamp, start, end):
            if timestamp is None:
                return None
            frame_timecode = FrameTimecode(timecode=timestamp, fps=fps)
            if frame_timecode.get_frames() < start.get_frames():
                return start.get_timecode()
            elif frame_timecode.get_frames() > end.get_frames():
                return end.get_timecode()
            else:
                return timestamp

        # Update activities timestamps
        for activity in scene.get('activities', []):
            if 'timestamp' in activity:
                if 'start_timestamp' in activity['timestamp']:
                    activity['timestamp']['start_timestamp'] = enforce_within_boundaries(
                        time_to_frametimecode(activity['timestamp']['start_timestamp'], fps, filename=video_id, scene_end_time=scene_end, worker_number = worker_number), scene_start, scene_end
                    )
                if 'end_timestamp' in activity['timestamp']:
                    activity['timestamp']['end_timestamp'] = enforce_within_boundaries(
                        time_to_frametimecode(activity['timestamp']['end_timestamp'], fps, filename=video_id, scene_end_time=scene_end,worker_number = worker_number), scene_start, scene_end
                    )

        # Update props timestamps
        for prop in scene.get('props', []):
            if 'timestamp' in prop:
                if 'start_timestamp' in prop['timestamp']:
                    prop['timestamp']['start_timestamp'] = enforce_within_boundaries(
                        time_to_frametimecode(prop['timestamp']['start_timestamp'], fps, filename=video_id, scene_end_time=scene_end,worker_number = worker_number), scene_start, scene_end
                    )
                if 'end_timestamp' in prop['timestamp']:
                    prop['timestamp']['end_timestamp'] = enforce_within_boundaries(
                        time_to_frametimecode(prop['timestamp']['end_timestamp'], fps, filename=video_id, scene_end_time=scene_end,worker_number = worker_number), scene_start, scene_end
                    )

        # Update video editing details timestamps
        for video_editing in scene.get('videoEditingDetails', []):
            if 'timestamps' in video_editing:
                if 'start_timestamp' in video_editing['timestamps']:
                    video_editing['timestamps']['start_timestamp'] = enforce_within_boundaries(
                        time_to_frametimecode(video_editing['timestamps']['start_timestamp'], fps, filename=video_id, scene_end_time=scene_end,worker_number = worker_number), scene_start, scene_end
                    )
                if 'end_timestamp' in video_editing['timestamps']:
                    video_editing['timestamps']['end_timestamp'] = enforce_within_boundaries(
                        time_to_frametimecode(video_editing['timestamps']['end_timestamp'], fps, filename=video_id, scene_end_time=scene_end,worker_number = worker_number), scene_start, scene_end
                    )

        # Update mood key moments timestamps
        for key_moment in scene.get('mood', {}).get('keyMoments', []):
            if 'timestamp' in key_moment:
                key_moment['timestamp'] = enforce_within_boundaries(
                    time_to_frametimecode(key_moment['timestamp'], fps, filename=video_id, scene_end_time=scene_end,worker_number = worker_number), scene_start, scene_end
                )

        # Update narrative progression timestamps
        for narrative in scene.get('narrativeProgression', []):
            if 'timestamp' in narrative:
                narrative['timestamp'] = enforce_within_boundaries(
                    time_to_frametimecode(narrative['timestamp'], fps, filename=video_id, scene_end_time=scene_end,worker_number = worker_number), scene_start, scene_end
                )

    # Update storylines climax timestamps
    if 'storylines' in data and 'climax' in data['storylines'] and 'timestamp' in data['storylines']['climax']:
        data['storylines']['climax']['timestamp'] = time_to_frametimecode(data['storylines']['climax']['timestamp'], fps, filename=video_id, scene_end_time=scene_end,worker_number = worker_number)

    # Update trimming suggestions timestamps
    for trimming in data.get('trimmingSuggestions', []):
        if 'timestamps' in trimming:
            if 'start_timestamp' in trimming['timestamps']:
                trimming['timestamps']['start_timestamp'] = enforce_within_boundaries(
                    time_to_frametimecode(trimming['timestamps']['start_timestamp'], fps, filename=video_id, scene_end_time=scene_end,worker_number = worker_number), scene_start, scene_end
                )
            if 'end_timestamp' in trimming['timestamps']:
                trimming['timestamps']['end_timestamp'] = enforce_within_boundaries(
                    time_to_frametimecode(trimming['timestamps']['end_timestamp'], fps, filename=video_id, scene_end_time=scene_end,worker_number = worker_number), scene_start, scene_end
                )

    return data

def result_exists(video_filename,output_directory):
    video_id = os.path.splitext(video_filename)[0]
    result_file = os.path.join(output_directory, f"{video_id}.json")
    error_file = os.path.join(output_directory, f"errors_{video_id}.json")
    return os.path.exists(result_file) or os.path.exists(error_file)

def process_single_video(video_id, worker_number):
    s3_folder_videos = 'videos/'
    video_key = f'{s3_folder_videos}/{video_id}.mp4'
    video_filename = f'{video_id}.mp4'
    video_local_path = os.path.join(video_folder_path, video_filename)
    if result_exists(video_filename,output_folder_path):
        print(f"Skipping {video_filename}, result already exists.")
        return

    # Download video from S3
    if not download_video_from_s3(video_key, video_local_path):
        # Handle download failure
        error_data = {"error": "File not found in S3"}
        error_file_path = os.path.join(output_folder_path, f"errors_{video_id}.json")
        with open(error_file_path, "w") as f:
            json.dump(error_data, f, indent=4)
        
        # Update status report for download failure
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        status_report = f"{timestamp} - {video_id} - failed - File not found in S3\n"
        print(status_report)
        with open(f"status/status_alignment_{worker_number}.txt", "a") as f:
            f.write(status_report)
        
        return

    # Construct paths
    json_path = os.path.join(json_folder_path, f"{video_id}.json")
    json_result_path = os.path.join(output_folder_path, f"{video_id}.json")

    # Load JSON file
    with open(json_path, 'r') as json_file:
        video_data = json.load(json_file)

    try:
        # Adjust scene boundaries using PySceneDetect to determine FPS
        fps, adjusted_scenes = adjust_scene_boundaries(video_local_path, video_data['scenes'], video_id, str(worker_number))

        # Update scenes in the original data
        video_data['scenes'] = adjusted_scenes
        video_data['fps'] = fps

        # Update all timestamps to FrameTimecode format
        video_data = update_timestamps_in_json(video_data, fps, video_id, str(worker_number))

        # Write updated JSON back to file
        with open(json_result_path, 'w') as json_file:
            json.dump(video_data, json_file, indent=4)

        print(f"Processed video {video_id}.")

        # Prepare the status report
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        status_report = f"{timestamp} - {video_id} - complete\n"
        print(status_report)
        
        # Append the status report to status.txt
        if worker_number is None:
            with open("status_alignment.txt", "a") as f:
                f.write(status_report)
        else:    
            with open(f"status/status_alignment_{worker_number}.txt", "a") as f:
                f.write(status_report)

    except Exception as e:
        # Handle any errors in adjusting scenes or updating timestamps
        error_data = {
            "error": str(e),
            "video_id": video_id,
            "worker_number": worker_number
        }
        error_file_path = os.path.join(output_folder_path, f"errors_{video_id}.json")
        with open(error_file_path, "w") as f:
            json.dump(error_data, f, indent=4)

        # Update status report for failure
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        status_report = f"{timestamp} - {video_id} - failed - Error during processing: {str(e)}\n"
        print(status_report)
        with open(f"status/status_alignment_{worker_number}.txt", "a") as f:
            f.write(status_report)

    finally:
        # Remove the video file after processing, even if an error occurred
        if os.path.exists(video_local_path):
            os.remove(video_local_path)
            print(f"Deleted local file {video_local_path} after processing.")


def process_chunk(videos_to_process, size_chunk, worker_number):
    # Calculate start and end indices for this worker's chunk
    start_index = worker_number * size_chunk
    end_index = min(start_index + size_chunk, len(videos_to_process))

    # Process videos in this worker's chunk
    for video_id in videos_to_process[start_index:end_index]:
        process_single_video(video_id, worker_number)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process videos in chunks.')
    parser.add_argument('size_chunk', type=int, help='Size of each chunk to process')
    parser.add_argument('worker_number', type=int, help='Worker number (zero-indexed)')
    parser.add_argument('--video_list', type=str, help='Optional video list file in JSON format')
    args = parser.parse_args()

    # Load the list of videos
    if args.video_list:
        with open(args.video_list, 'r') as f:
            videos_to_process = json.load(f)
        print(f"Using provided video list: {args.video_list}")
    else:
        with open('video_alignment_to_process.json', 'r') as f:
            videos_to_process = json.load(f)
        print("Using default video list: video_alignment_to_process.json")

    # Process the assigned chunk
    process_chunk(videos_to_process, args.size_chunk, args.worker_number)




