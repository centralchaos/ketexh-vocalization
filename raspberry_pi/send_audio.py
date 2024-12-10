"""

by: Jose Carlo Sia / chaoserver

Client Application for Chicken Vocalization Analysis

This module provides the client-side functionality for sending audio files
to the analysis API and handling the responses.

Dependencies:
- requests: HTTP client for API communication
- pathlib: File path handling
- logging: Structured logging
- json: Response data handling

Features:
- Automatic directory monitoring
- File processing and upload
- Result storage and organization
- Status tracking and reporting
"""

import requests
import logging
import time
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, Dict, Any

# Configure logging with timestamp and severity level
logging.basicConfig(
    filename='audio_sender.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AudioSender:
    """
    Handles audio file processing and API communication.
    
    Directory Structure:
        audio_files/: Input directory for WAV files
        processed_audio/: Storage for processed files
        results/: JSON results from analysis
    
    Configuration:
        api_url: API endpoint (default: http://localhost:1200)
        
    Stats Tracking:
        - Files processed
        - High alerts detected
        - Last processing timestamp
    """
    
    def __init__(self, api_url: str = "http://localhost:1200"):
        self.api_url = api_url
        self.audio_dir = Path("audio_files")
        self.processed_dir = Path("processed_audio")
        self.results_dir = Path("results")
        self.stats = {
            'files_processed': 0,
            'high_alerts': 0,
            'last_processed': None
        }
        
        # Ensure required directories exist
        self.processed_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

    def send_file(self, audio_path: Path) -> Optional[Dict[str, Any]]:
        """Send audio file to API and get results"""
        file_handle = None
        try:
            # Prepare the file for sending
            file_handle = open(audio_path, 'rb')
            files = {'file': file_handle}
            
            # Send to API
            logging.info(f"Sending {audio_path.name} to API")
            print(f"\nProcessing: {audio_path.name}")
            response = requests.post(f"{self.api_url}/api/v1/analyze", files=files)
            response.raise_for_status()
            
            # Get job ID
            job_data = response.json()
            job_id = job_data['job_id']
            logging.info(f"Got job ID: {job_id}")
            print(f"Job ID: {job_id}")
            
            # Poll for results
            while True:
                status_response = requests.get(f"{self.api_url}/api/v1/status/{job_id}")
                status_data = status_response.json()
                
                if status_data['status'] == 'completed':
                    # Save results
                    result_path = self.results_dir / f"{audio_path.stem}_result.json"
                    with open(result_path, 'w') as f:
                        json.dump(status_data['result'], f, indent=2)
                    
                    # Move processed file
                    processed_path = self.processed_dir / audio_path.name
                    audio_path.rename(processed_path)
                    
                    # Update stats
                    self.stats['files_processed'] += 1
                    self.stats['last_processed'] = datetime.now().strftime("%H:%M:%S")
                    if status_data['result']['alert_level'] == 'HIGH':
                        self.stats['high_alerts'] += 1
                    
                    return status_data['result']
                    
                elif status_data['status'] == 'failed':
                    logging.error(f"Processing failed for {audio_path.name}")
                    print(f"Failed: {audio_path.name}")
                    return None
                    
                time.sleep(1)  # Wait before polling again
                print(".", end="", flush=True)  # Show progress
                
        except Exception as e:
            logging.error(f"Error processing {audio_path.name}: {str(e)}")
            print(f"\nError: {audio_path.name} - {str(e)}")
            return None
        finally:
            if file_handle:
                file_handle.close()

    def process_directory(self):
        """Process all WAV files in audio_files directory"""
        try:
            files = list(self.audio_dir.glob("*.wav"))
            if not files:
                print("\nNo files to process")
                return False

            print(f"\nMonitoring directory: {self.audio_dir}")
            print(f"Stats: Processed={self.stats['files_processed']}, "
                  f"High Alerts={self.stats['high_alerts']}, "
                  f"Last={self.stats['last_processed']}")

            for audio_path in files:
                logging.info(f"Found audio file: {audio_path.name}")
                result = self.send_file(audio_path)
                
                if result and result['alert_level'] == 'HIGH':
                    logging.warning(f"HIGH alert for {audio_path.name}: {result['recommendation']}")
                    print(f"\n⚠️  HIGH ALERT: {audio_path.name} - {result['recommendation']}")
            
            print("\nAll files processed")
            return True
                    
        except Exception as e:
            logging.error(f"Error in process_directory: {str(e)}")
            print(f"\nError in processing directory: {str(e)}")
            return False

def main():
    sender = AudioSender()
    try:
        print("\nAudio Sender Started")
        sender.process_directory()
        print("\nProcessing complete")
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    finally:
        logging.info("Shutting down audio sender")
        print(f"\nFinal Stats: Processed={sender.stats['files_processed']}, "
              f"High Alerts={sender.stats['high_alerts']}")

if __name__ == "__main__":
    main()