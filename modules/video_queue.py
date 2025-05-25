import threading
import time
import uuid
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
import queue as queue_module  # Renamed to avoid conflicts
import io
import base64
from PIL import Image
import numpy as np
import time
import sys

from diffusers_helper.thread_utils import AsyncStream


# Simple LIFO queue implementation to avoid dependency on queue.LifoQueue
class SimpleLifoQueue:
    def __init__(self):
        self._queue = []
        self._mutex = threading.Lock()
        self._not_empty = threading.Condition(self._mutex)
    
    def put(self, item):
        with self._mutex:
            self._queue.append(item)
            self._not_empty.notify()
    
    def get(self):
        with self._not_empty:
            while not self._queue:
                self._not_empty.wait()
            return self._queue.pop()
    
    def task_done(self):
        pass  # For compatibility with queue.Queue


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    params: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[str] = None
    progress_data: Optional[Dict] = None
    queue_position: Optional[int] = None
    stream: Optional[Any] = None
    input_image: Optional[np.ndarray] = None
    latent_type: Optional[str] = None
    thumbnail: Optional[str] = None
    generation_type: Optional[str] = None # Added generation_type

    def __post_init__(self):
        # Store generation type
        self.generation_type = self.params.get('model_type', 'Original') # Initialize generation_type

        # Store input image or latent type
        if 'input_image' in self.params and self.params['input_image'] is not None:
            self.input_image = self.params['input_image']
            # Create thumbnail
            if isinstance(self.input_image, np.ndarray):
                # Handle numpy array (image)
                img = Image.fromarray(self.input_image)
                img.thumbnail((100, 100))
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                self.thumbnail = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
            elif isinstance(self.input_image, str):
                # Handle string (video path)
                # Create a generic video thumbnail
                img = Image.new('RGB', (100, 100), (0, 0, 128))  # Blue for video
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                self.thumbnail = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
            else:
                # Handle other types
                self.thumbnail = None
        elif 'latent_type' in self.params:
            self.latent_type = self.params['latent_type']
            # Create a colored square based on latent type
            color_map = {
                "Black": (0, 0, 0),
                "White": (255, 255, 255),
                "Noise": (128, 128, 128),
                "Green Screen": (0, 177, 64)
            }
            color = color_map.get(self.latent_type, (0, 0, 0))
            img = Image.new('RGB', (100, 100), color)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            self.thumbnail = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"


class VideoJobQueue:
    def __init__(self):
        self.queue = queue_module.Queue()  # Using standard Queue instead of LifoQueue
        self.jobs = {}
        self.current_job = None
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.worker_function = None  # Will be set from outside
        self.is_processing = False  # Flag to track if we're currently processing a job
    
    def set_worker_function(self, worker_function):
        """Set the worker function to use for processing jobs"""
        self.worker_function = worker_function
    
    def serialize_job(self, job):
        """Serialize a job to a JSON-compatible format"""
        try:
            # Create a simplified representation of the job
            serialized = {
                "id": job.id,
                "status": job.status.value,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "error": job.error,
                "result": job.result,
                "queue_position": job.queue_position,
                "generation_type": job.generation_type,
            }
            
            # Add simplified params (excluding complex objects)
            serialized_params = {}
            for k, v in job.params.items():
                if k not in ["input_image", "end_frame_image", "stream"]:
                    # Try to include only JSON-serializable values
                    try:
                        # Test if value is JSON serializable
                        json.dumps({k: v})
                        serialized_params[k] = v
                    except (TypeError, OverflowError):
                        # Skip non-serializable values
                        pass
            
            # Handle LoRA information specifically
            # Only include selected LoRAs for the generation
            if "selected_loras" in job.params and job.params["selected_loras"]:
                selected_loras = job.params["selected_loras"]
                # Ensure it's a list
                if not isinstance(selected_loras, list):
                    selected_loras = [selected_loras] if selected_loras is not None else []
                
                # Get LoRA values if available
                lora_values = job.params.get("lora_values", [])
                if not isinstance(lora_values, list):
                    lora_values = [lora_values] if lora_values is not None else []
                
                # Get loaded LoRA names
                lora_loaded_names = job.params.get("lora_loaded_names", [])
                if not isinstance(lora_loaded_names, list):
                    lora_loaded_names = [lora_loaded_names] if lora_loaded_names is not None else []
                
                # Create LoRA data dictionary
                lora_data = {}
                for lora_name in selected_loras:
                    try:
                        # Find the index of the LoRA in loaded names
                        idx = lora_loaded_names.index(lora_name) if lora_loaded_names else -1
                        # Get the weight value
                        weight = lora_values[idx] if lora_values and idx >= 0 and idx < len(lora_values) else 1.0
                        # Handle weight as list
                        if isinstance(weight, list):
                            weight_value = weight[0] if weight and len(weight) > 0 else 1.0
                        else:
                            weight_value = weight
                        # Store as float
                        lora_data[lora_name] = float(weight_value)
                    except (ValueError, IndexError):
                        # Default weight if not found
                        lora_data[lora_name] = 1.0
                    except Exception as e:
                        print(f"Error processing LoRA {lora_name}: {e}")
                        lora_data[lora_name] = 1.0
                
                # Add to serialized params
                serialized_params["loras"] = lora_data
            
            serialized["params"] = serialized_params
            
            # Don't include the thumbnail as it can be very large and cause issues
            # if job.thumbnail:
            #     serialized["thumbnail"] = job.thumbnail
                
            return serialized
        except Exception as e:
            print(f"Error serializing job {job.id}: {e}")
            # Return minimal information that should always be serializable
            return {
                "id": job.id,
                "status": job.status.value,
                "error": f"Error serializing: {str(e)}"
            }
    
    def save_queue_to_json(self):
        """Save the current queue to queue.json"""
        try:
            # Make a copy of job IDs to avoid holding the lock while serializing
            with self.lock:
                job_ids = list(self.jobs.keys())
            
            # Serialize jobs outside the lock
            serialized_jobs = {}
            for job_id in job_ids:
                job = self.get_job(job_id)
                if job:
                    serialized_jobs[job_id] = self.serialize_job(job)
            
            # Save to file
            with open("queue.json", "w") as f:
                json.dump(serialized_jobs, f, indent=2)
                
            print(f"Saved {len(serialized_jobs)} jobs to queue.json")
        except Exception as e:
            print(f"Error saving queue to JSON: {e}")
    
    def add_job(self, params):
        """Add a job to the queue and return its ID"""
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            params=params,
            status=JobStatus.PENDING,
            created_at=time.time(),
            progress_data={},
            stream=AsyncStream()
        )
        
        with self.lock:
            print(f"Adding job {job_id} to queue, current job is {self.current_job.id if self.current_job else 'None'}")
            self.jobs[job_id] = job
            self.queue.put(job_id)
        
        # Save the queue to JSON after adding a new job (outside the lock)
        try:
            self.save_queue_to_json()
        except Exception as e:
            print(f"Error saving queue to JSON after adding job: {e}")
        
        return job_id
    
    def get_job(self, job_id):
        """Get job by ID"""
        with self.lock:
            return self.jobs.get(job_id)
    
    def get_all_jobs(self):
        """Get all jobs"""
        with self.lock:
            return list(self.jobs.values())
    
    def cancel_job(self, job_id):
        """Cancel a pending job"""
        with self.lock:
            job = self.jobs.get(job_id)
            if job and job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()  # Mark completion time
                return True
            elif job and job.status == JobStatus.RUNNING:
                # Send cancel signal to the job's stream
                job.stream.input_queue.push('end')
                # Mark job as cancelled (this will be confirmed when the worker processes the end signal)
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()  # Mark completion time
                return True
            return False
    
    def get_queue_position(self, job_id):
        """Get position in queue (0 = currently running)"""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
                
            if job.status == JobStatus.RUNNING:
                return 0
                
            if job.status != JobStatus.PENDING:
                return None
                
            # Count pending jobs ahead in queue
            position = 1  # Start at 1 because 0 means running
            for j in self.jobs.values():
                if (j.status == JobStatus.PENDING and 
                    j.created_at < job.created_at):
                    position += 1
            return position
    
    def update_job_progress(self, job_id, progress_data):
        """Update job progress data"""
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                job.progress_data = progress_data
    
    def _worker_loop(self):
        """Worker thread that processes jobs from the queue"""
        while True:
            try:
                #race condition avoidance
                time.sleep(0.01)
                # Get the next job ID from the queue
                try:
                    job_id = self.queue.get(block=True, timeout=1.0)  # Use timeout to allow periodic checks
                except queue_module.Empty:
                    # No jobs in queue, just continue the loop
                    continue
                
                with self.lock:
                    job = self.jobs.get(job_id)
                    if not job:
                        self.queue.task_done()
                        continue
                    
                    # Skip cancelled jobs
                    if job.status == JobStatus.CANCELLED:
                        self.queue.task_done()
                        continue
                    
                    # If we're already processing a job, wait for it to complete
                    if self.is_processing:
                        # Put the job back in the queue
                        self.queue.put(job_id)
                        self.queue.task_done()
                        time.sleep(0.1)  # Small delay to prevent busy waiting
                        continue
                    
                    print(f"Starting job {job_id}, current job was {self.current_job.id if self.current_job else 'None'}")
                    job.status = JobStatus.RUNNING
                    job.started_at = time.time()
                    self.current_job = job
                    self.is_processing = True
                
                job_completed = False
                
                try:
                    if self.worker_function is None:
                        raise ValueError("Worker function not set. Call set_worker_function() first.")
                    
                    # Start the worker function with the job parameters
                    from diffusers_helper.thread_utils import async_run
                    print(f"Starting thread (worker) for job: {job_id}..", end="")
                    sys.stdout.flush()
                    async_run(
                        self.worker_function,
                        **job.params,
                        job_stream=job.stream
                    )
                    print(f"done")
                    
                    # Process the results from the stream
                    output_filename = None
                    
                    # Track activity time for logging purposes
                    last_activity_time = time.time()
                    
                    while True:
                        #race condition avoidance
                        time.sleep(0.005)
                        # Check if job has been cancelled before processing next output
                        with self.lock:
                            if job.status == JobStatus.CANCELLED:
                                print(f"Job {job_id} was cancelled, breaking out of processing loop")
                                job_completed = True
                                break
                        
                        # Get current time for activity checks
                        current_time = time.time()
                        
                        # Check for inactivity (no output for a while)
                        if current_time - last_activity_time > 60:  # 1 minute of inactivity
                            print(f"Checking if job {job_id} is still active...")
                            # Just a periodic check, don't break yet
                        
                        try:
                            # Try to get data from the queue with a non-blocking approach
                            flag, data = job.stream.output_queue.next()
                            
                            # Update activity time since we got some data
                            last_activity_time = time.time()
                            
                            if flag == 'file':
                                output_filename = data
                                with self.lock:
                                    job.result = output_filename
                            
                            elif flag == 'progress':
                                preview, desc, html = data
                                with self.lock:
                                    job.progress_data = {
                                        'preview': preview,
                                        'desc': desc,
                                        'html': html
                                    }
                            
                            elif flag == 'end':
                                print(f"Received end signal for job {job_id}")
                                job_completed = True
                                break
                            
                        except IndexError:
                            # Queue is empty, wait a bit and try again
                            time.sleep(0.1)
                            continue
                        except Exception as e:
                            print(f"Error processing job output: {e}")
                            # Wait a bit before trying again
                            time.sleep(0.1)
                            continue
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error processing job {job_id}: {e}")
                    with self.lock:
                        job.status = JobStatus.FAILED
                        job.error = str(e)
                        job.completed_at = time.time()
                    job_completed = True
                
                finally:
                    with self.lock:
                        # Make sure we properly clean up the job state
                        if job.status == JobStatus.RUNNING:
                            if job_completed:
                                job.status = JobStatus.COMPLETED
                            else:
                                # Something went wrong but we didn't mark it as completed
                                job.status = JobStatus.FAILED
                                job.error = "Job processing was interrupted"
                            
                            job.completed_at = time.time()
                        
                        print(f"Finishing job {job_id} with status {job.status}")
                        self.is_processing = False
                        self.current_job = None
                        self.queue.task_done()
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error in worker loop: {e}")
                
                # Make sure we reset processing state if there was an error
                with self.lock:
                    self.is_processing = False
                    if self.current_job:
                        self.current_job.status = JobStatus.FAILED
                        self.current_job.error = f"Worker loop error: {str(e)}"
                        self.current_job.completed_at = time.time()
                        self.current_job = None
                
                time.sleep(0.5)  # Prevent tight loop on error
