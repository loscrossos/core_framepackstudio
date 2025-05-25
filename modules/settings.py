import json
from pathlib import Path
from typing import Dict, Any, Optional
import os

class Settings:
    def __init__(self):
        # Get the project root directory (where settings.py is located)
        project_root = Path(__file__).parent.parent
        
        self.settings_file = project_root / ".framepack" / "settings.json"
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Set default paths relative to project root
        self.default_settings = {
            "save_metadata": True,
            "gpu_memory_preservation": 6,
            "output_dir": str(project_root / "outputs"),
            "metadata_dir": str(project_root / "outputs"),
            "lora_dir": str(project_root / "loras"),
            "gradio_temp_dir": str(project_root / "temp"),
            "input_files_dir": str(project_root / "input_files"),  # New setting for input files
            "auto_save_settings": True,
            "gradio_theme": "base",
            "mp4_crf": 16,
            "clean_up_videos": True,
            "cleanup_temp_folder": False
        }
        self.settings = self.load_settings()

    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file or return defaults"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                    # Merge with defaults to ensure all settings exist
                    settings = self.default_settings.copy()
                    settings.update(loaded_settings)
                    return settings
            except Exception as e:
                print(f"Error loading settings: {e}")
                return self.default_settings.copy()
        return self.default_settings.copy()

    def save_settings(self, **kwargs):
        """Save settings to file. Accepts keyword arguments for any settings to update."""
        # Update self.settings with any provided keyword arguments
        self.settings.update(kwargs)
        # Ensure all default fields are present
        for k, v in self.default_settings.items():
            self.settings.setdefault(k, v)

        # Ensure directories exist for relevant fields
        for dir_key in ["output_dir", "metadata_dir", "lora_dir", "gradio_temp_dir"]:
            dir_path = self.settings.get(dir_key)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

        # Save to file
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=4)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value"""
        return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a setting value"""
        self.settings[key] = value
        if self.settings.get("auto_save_settings", True):
            self.save_settings()

    def update(self, settings: Dict[str, Any]) -> None:
        """Update multiple settings at once"""
        self.settings.update(settings)
        if self.settings.get("auto_save_settings", True):
            self.save_settings()
