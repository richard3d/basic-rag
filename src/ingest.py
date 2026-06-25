import os
import asyncio
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from embed import embed_text_file
from query import query_rag

class FileObserver(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        print(f" New file detected: {event.src_path}")
        asyncio.run(self._handle(event.src_path))

    async def _handle(self, path):
        try:
            print(f"Processing content of {os.path.basename(path)}...")
            await embed_text_file(path)
            print(f"✅ Successfully processed {os.path.basename(path)}")
        except Exception as e:
            print(f"Error processing file: {e}")

def start_file_observer(watch_directory):
    print("file-observer running...")
    # Ensure the directory exists
    os.makedirs(watch_directory, exist_ok=True)
    
    event_handler = FileObserver()
    observer = Observer()
    observer.schedule(event_handler, path=watch_directory, recursive=False)
    
    print(f"🚀 Starting file observer on {watch_directory}...")
    observer.start()
    return observer