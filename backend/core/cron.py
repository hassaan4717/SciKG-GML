import gc
import os
import shutil
import subprocess
import sys

from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from core.models import ChatSession

load_dotenv()

BASE_RAG_DIR = os.getenv("BASE_RAG_DIR")
INTERVAL = 30


def clean_trash_folder():
    """
    Scans the RAG storage directory and removes orphaned session folders 
    that no longer have corresponding database records.
    """
    if not BASE_RAG_DIR or not os.path.exists(BASE_RAG_DIR):
        return

    print(f"[Cron Janitor] Initiating scheduled {INTERVAL}-minute automated storage garbage collection sweep...")
    gc.collect()

    for folder_name in os.listdir(BASE_RAG_DIR):
        if not folder_name.startswith("session_"):
            continue

        folder_path = os.path.normpath(os.path.join(BASE_RAG_DIR, folder_name))
        if not os.path.isdir(folder_path):
            continue

        try:
            session_id = int(folder_name.split("_")[1])
        except (IndexError, ValueError):
            continue

        if ChatSession.objects.filter(id=session_id).exists():
            continue

        print(f"[Cron Janitor] Found orphaned directory: {folder_name}")
        _force_delete_folder(folder_path, folder_name)


def _force_delete_folder(folder_path: str, folder_name: str):
    """
    Attempts multiple strategies to delete a folder, from safest to most aggressive.
    """
    # Strategy 1: Standard Python deletion
    try:
        shutil.rmtree(folder_path)
        if not os.path.exists(folder_path):
            print(f"[Cron Janitor] Successfully removed with shutil: {folder_name}")
            return
    except Exception as e:
        print(f"[Cron Janitor] shutil failed: {e}")

    # Strategy 2: Platform-specific force deletion
    try:
        if sys.platform == 'win32':
            subprocess.run(
                ['cmd', '/c', 'rmdir', '/s', '/q', folder_path],
                capture_output=True,
                text=True,
                timeout=30
            )
        else:
            subprocess.run(
                ['rm', '-rf', folder_path],
                capture_output=True,
                text=True,
                timeout=30
            )

        if not os.path.exists(folder_path):
            print(f"[Cron Janitor] Force wiped with system command: {folder_name}")
        else:
            print(f"[Cron Janitor] System command reported success but folder still exists")
    except Exception as e:
        print(f"[Cron Janitor] System command failed: {e}")

    if os.path.exists(folder_path):
        print(f"[Cron Janitor] Could not delete {folder_name} — will retry next cycle")


def start_cleaner_scheduler():
    """
    Initializes and starts the background scheduler for periodic garbage collection.
    """
    scheduler = BackgroundScheduler()
    scheduler.add_job(clean_trash_folder, 'interval', minutes=INTERVAL)
    scheduler.start()
    print(f"[Scheduler Service] Asynchronous {INTERVAL}-Minute Garbage Collector thread is now active.")