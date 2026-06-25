import time
import threading
import uvicorn
from config import FILE_OBSERVER_DIRECTORY
from ingest import start_file_observer

if __name__ == "__main__":
    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": "server:app", "host": "0.0.0.0", "port": 8000},
        daemon=True,
    )
    server_thread.start()
    
    observer = start_file_observer(watch_directory=FILE_OBSERVER_DIRECTORY)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()