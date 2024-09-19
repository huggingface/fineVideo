import boto3  
import google.generativeai as genai
from openai import OpenAI
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict
import instructor
import os
import google.api_core.exceptions 
import argparse

#
# Given an input list of videos, this script downloads them from S3 and annotates the videos with Gemini 
# and structures the data with instructor using GPT4o underneath.
#
# The code is prepared to run as a standalone application:
# The first parameter is size_chunk: it basically divide the list of videos in sublists of length size_chunk
# The worker_number decides in which sublist of size size_chunk the current execution will be working on
# --video-list is to specify the json file that contains a list of videoids as a JSON list. If that is not provided, it defaults to oracle_videos_server.json
#


### CONFIG ###

# Directories to download input videos and output annotation results
input_directory = 'videos_minioracle/'
output_directory = 'videos_minioracle_results/'
bucket_name = '<bucket_name>'

GEMINI_PATH="/path/to/your/key/file"
OPENAI_PATH="/path/to/your/key/file"
###


### Data Schema ###

class Character(TypedDict):
    characterId: str
    name: str
    description: str

class Timestamps(TypedDict):
    start_timestamp: str
    end_timestamp: str

class Activity(TypedDict):
    description: str
    timestamp: Timestamps

class Prop(TypedDict):
    name: str
    timestamp: Timestamps

class VideoEditingDetail(TypedDict):
    description: str
    timestamps: Timestamps

class KeyMoment(TypedDict):
    timestamp: str
    changeDescription: str

class Mood(TypedDict):
    description: str
    keyMoments: List[KeyMoment]

class NarrativeProgression(TypedDict):
    description: str
    timestamp: str

class CharacterInteraction(TypedDict):
    characters: List[str]
    description: str

class Scene(TypedDict):
    sceneId: int
    title: str
    timestamps: Timestamps
    cast: List[str]
    activities: List[Activity]
    props: List[Prop]
    videoEditingDetails: List[VideoEditingDetail]
    mood: Mood
    narrativeProgression: List[NarrativeProgression]
    characterInteraction: List[CharacterInteraction]
    thematicElements: str
    contextualRelevance: str
    dynamismScore: float
    audioVisualCorrelation: float

class Climax(TypedDict):
    description: str
    timestamp: str

class Storyline(TypedDict):
    description: str
    scenes: List[int]
    climax: Climax

class QAndA(TypedDict):
    question: str
    answer: str

class TrimmingSuggestion(TypedDict, total=False):
    timestamps: Timestamps
    description: str

class Schema(TypedDict):
    title: str
    description: str
    characterList: List[Character]
    scenes: List[Scene]
    storylines: Storyline
    qAndA: List[QAndA]
    trimmingSuggestions: List[TrimmingSuggestion]

###

class VideoProcessor:
    def __init__(self, gemini_api_key_path: str, openai_api_key_path: str):
        # Initialize API keys and clients
        self.gemini_apikey = self._read_api_key(gemini_api_key_path)
        self.openai_apikey = self._read_api_key(openai_api_key_path)
        genai.configure(api_key=self.gemini_apikey)
        self.clientOpenAI = OpenAI(api_key=self.openai_apikey)

    def _read_api_key(self, path: str) -> str:
        with open(path, "r") as file:
            return file.read().strip()


    def upload_video(self, file_path: str) -> Dict[str, Any]:
        print(f"Uploading file... {file_path}")
        try:
            video_file = genai.upload_file(path=file_path)

            while video_file.state.name == "PROCESSING":
                time.sleep(10)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                return {"error": "Upload failed", "video_file": None}
            return {"video_file": video_file}
        
        except Exception as e:
            return {"error": str(e), "video_file": None}



    def process_video(self, video_file: Any, addition_to_prompt=None) -> Dict[str, Optional[str]]:
        if "error" in video_file:
            return {"error": "video_file error: " + video_file["error"], "gemini_text": None}

        max_retries = 5
        attempt = 0
        delay = 2  # in seconds

        while attempt < max_retries:
            try:
                print(f"Processing {video_file['video_file'].display_name} (Attempt {attempt + 1})")
                prompt = open("gemini_prompt.txt", "r").read()
                if addition_to_prompt:
                    print(f"\t adding addition to prompt: {addition_to_prompt}")
                    prompt = prompt + addition_to_prompt
                model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
                response = model.generate_content(
                    [video_file['video_file'], prompt],
                    request_options={"timeout": 600},
                    safety_settings=[
                        {"category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE},
                        {"category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE},
                        {"category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE},
                        {"category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE}
                    ]
                )

                if not response.candidates:
                    return {"error": "No candidates returned. Feedback: " + str(response.prompt_feedback), "gemini_text": None}
                #Cleaning up the analyzed file
                genai.delete_file(video_file['video_file'].name)
                
                return {"gemini_text": response.text}

            except google.api_core.exceptions.InternalServerError as e:
                print(f"InternalServerError occurred: {e}. Retrying in {delay} seconds...")
                attempt += 1
                time.sleep(delay)
                delay *= 2  # Exponential backoff

            except Exception as e:
                if "The read operation timed out" in str(e) or "record layer failure" in str(e):
                    print(f"Gemini error: {str(e)}. Retrying in {delay} seconds...")
                    attempt +=1
                    time.sleep(delay)
                    delay *= 2
                else:
                    return {"error": str(e), "gemini_text": None}

        # If all retries fail
        return {"error": f"Failed after {max_retries} attempts due to InternalServerError / timeouts / SSL.", "gemini_text": None}



    def obtain_json(self, gemini_answer: Optional[str]) -> Dict[str, Optional[str]]:

        if gemini_answer is None or (isinstance(gemini_answer, dict) and "error" in gemini_answer):
            return {"error": gemini_answer.get("error") if gemini_answer else "No Gemini answer", "json_result": None}

        try:
            # Patch the OpenAI client
            client = instructor.from_openai(self.clientOpenAI)
            promptOpenAI = gemini_answer
            completion = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                response_model=Schema,
                messages=[
                    {"role": "user", "content": promptOpenAI},
                ]
            )

            return {"json_result": completion.json()}
        except Exception as e:
            return {"error": str(e), "json_result": None}
    def prep_return(self, final_answer=None, gemini_error = None, gemini_raw_result=None, 
                    instructor_error = None, instructor_raw_result=None):
        return {
                "final_answer": final_answer,
                "gemini": {
                    "error": gemini_error,
                    "raw_result": gemini_raw_result
                },
                "instructor": {
                    "error": instructor_error,
                    "raw_result": instructor_raw_result
                }
            }
    
    def process(self, file_path: str) -> Dict[str, Any]:

        # Upload video to Gemini
        gemini_result = self.upload_video(file_path)
        if gemini_result.get("error"):
            return self.prep_return(gemini_error=gemini_result['error'])

        # Process video with Gemini
        gemini_text = self.process_video(gemini_result)
        if gemini_text.get("error"):
            return self.prep_return(gemini_error=gemini_text['error'])

        gemini_out_text = gemini_text.get("gemini_text")
        if gemini_out_text is None or gemini_text == "":
            return self.prep_return(gemini_error="Empty gemini answer", 
                                    gemini_raw_result=gemini_out_text)


        # Obtain JSON from the instructor
        instructor_result = self.obtain_json(gemini_out_text)

        if "IncompleteOutputException" in instructor_result.get("error", "") or "ValidationError" in instructor_result.get("error", ""):
            print(f"\tRetrying full pipeline due to Instructor exception: {instructor_result['error']}")
            # Retry processing the video
            gemini_result = self.upload_video(file_path)
            gemini_text = self.process_video(gemini_result, addition_to_prompt=" be concise with your answer")
            print("\t retry completed")

            # Check if there was an error during reprocessing
            if gemini_text.get("error"):
                return self.prep_return(gemini_error=gemini_text['error'])
            gemini_out_text = gemini_text.get("gemini_text")
            if gemini_out_text is None or gemini_text == "":
                return self.prep_return(gemini_error="Empty gemini answer", 
                                        gemini_raw_result=gemini_out_text)

            # Retry obtaining JSON from the instructor with the new Gemini text
            instructor_result = self.obtain_json(gemini_text.get("gemini_text"))



        # Prepare the final response
        final_answer = json.loads(instructor_result["json_result"]) if instructor_result["json_result"] and not instructor_result.get("error") else None
        return self.prep_return(final_answer=final_answer,
                         gemini_raw_result=gemini_text.get("gemini_text"),
                         instructor_raw_result=instructor_result.get("json_result",None),
                         instructor_error=instructor_result.get("error",None))

# AWS S3 Configuration
session = boto3.Session() 
s3_client = session.client('s3')

# Ensure the input and output directories exist
os.makedirs(input_directory, exist_ok=True)
os.makedirs(output_directory, exist_ok=True)

# Function to check if a result or error file exists for a video
def result_exists(video_filename):
    video_id = os.path.splitext(video_filename)[0]
    result_file = os.path.join(output_directory, f"{video_id}.json")
    error_file = os.path.join(output_directory, f"errors_{video_id}.json")
    return os.path.exists(result_file) or os.path.exists(error_file)

# Function to download video from S3
def download_video_from_s3(video_key, local_path):
    bucket_name = '<bucket_name>'
    try:
        s3_client.download_file(bucket_name, video_key, local_path)
        print(f"Downloaded {video_key} to {local_path}")
        return True
    except Exception as e:
        print(f"Failed to download {video_key} from S3: {e}")
        return False

def process_single_video(video_id, worker_number):
    videos_path = 'path/'
    video_key = f'{videos_path}/{video_id}.mp4'
    video_filename = f'{video_id}.mp4'
    local_path = os.path.join(input_directory, video_filename)

    if result_exists(video_filename):
        print(f"Skipping {video_filename}, result already exists.")
        return

    # Download video from S3
    if not download_video_from_s3(video_key, local_path):
        # Handle download failure
        error_data = {"error": "File not found in S3"}
        error_file_path = os.path.join(output_directory, f"errors_{video_id}.json")
        with open(error_file_path, "w") as f:
            json.dump(error_data, f, indent=4)
        
        # Update status report for download failure
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        status_report = f"{timestamp} - {video_id} - failed - File not found in S3\n"
        print(status_report)
        with open("status.txt", "a") as f:
            f.write(status_report)
        
        return

    # Process the video using VideoProcessor class
    processor = VideoProcessor(gemini_api_key_path=GEMINI_PATH, openai_api_key_path=OPENAI_PATH)
    result = processor.process(local_path)
    video_id = os.path.splitext(os.path.basename(local_path))[0]

    # Save final answer to JSON if available
    if result.get("final_answer") is not None:
        with open(os.path.join(output_directory, f"{video_id}.json"), "w") as f:
            json.dump(result["final_answer"], f, indent=4)
        status = "successful"
    else:
        status = "failed"

    # Save errors to JSON if any errors exist
    errors = {}
    if (result.get("gemini", {}).get("error") is not None) or (result.get("instructor", {}).get("error") is not None):
        gemini_raw = result["gemini"].get("raw_result")
        errors = {
            "gemini_error": result["gemini"].get("error"),
            "instructor_error": result["instructor"].get("error"),
            "gemini_raw_result": gemini_raw
        }
        with open(os.path.join(output_directory, f"errors_{video_id}.json"), "w") as f:
            json.dump(errors, f, indent=4)
    
    # Prepare the status report
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    error_details = ', '.join(filter(None, [result.get("gemini", {}).get("error"), result.get("instructor", {}).get("error")]))
    status_report = f"{timestamp} - {video_id} - {status} - {error_details if error_details else 'None'}\n"
    print(status_report)
    
    # Append the status report to status.txt
    if worker_number is None:
        with open("status.txt", "a") as f:
            f.write(status_report)
    else:    
        with open(f"status/status_{worker_number}.txt", "a") as f:
            f.write(status_report)

    # Remove the video file after processing
    os.remove(local_path)
    print(f"Deleted local file {local_path} after processing.")

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
        with open('oracle_videos_server.json', 'r') as f:
            videos_to_process = json.load(f)
        print("Using default video list: oracle_videos_server.json")

    # Process the assigned chunk
    process_chunk(videos_to_process, args.size_chunk, args.worker_number)

