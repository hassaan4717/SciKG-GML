import os
from django.apps import AppConfig

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        """
        Application startup lifecycle hook. 
        Guarantees background tasks only launch inside the master runtime worker process.
        """
        # Prevent double execution during auto-reload
        if os.environ.get('RUN_MAIN') == 'true':
            from core.cron import start_cleaner_scheduler
            start_cleaner_scheduler()