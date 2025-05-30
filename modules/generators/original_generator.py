
# Input data
in_dotenv_needed_models = {
    "hunyuanvideo-community/HunyuanVideo",
    "lllyasviel/flux_redux_bfl",
    "lllyasviel/FramePack_F1_I2V_HY_20250503",
    "lllyasviel/FramePackI2V_HY",
}

in_dotenv_needed_paths = {
    "HF_HOME": "./models/hf_download",
}

in_dotenv_needed_params = {
    "DEBUG_MODE": False,

}


#LCX1.0##################################################################
#FILELOADER##############################################################
#########################################################################
debug_mode=False
LCX_APP_NAME="CROSSOS_FILE_CHECK"
DEFAULT_MODEL_CONFIG_FILE = "configmodel.txt"
in_model_config_file=DEFAULT_MODEL_CONFIG_FILE
# --- Helper Functions ---
#dotenv prefixes
PREFIX_MODEL="PATH_MODEL_"
PREFIX_PATH="PATH_NEEDED_"
LOG_PREFIX="CROSSOS_LOG"
import re
import os 
from pathlib import Path
from typing import Dict, Set, Any, Union
def model_to_varname(model_path: str, prefix: str) -> str:
    """Converts a model path to a dotenv-compatible variable name"""
    model_name = model_path.split("/")[-1]
    varname = re.sub(r"[^a-zA-Z0-9]", "_", model_name.upper())
    return f"{prefix}{varname}"

def varname_to_model(varname: str, prefix: str) -> str:
    """Converts a variable name back to original model path format"""
    if varname.startswith("PATH_MODEL_"):
        model_part = varname[prefix.len():].lower().replace("_", "-")
        return f"Zyphra/{model_part}"
    return ""

def read_existing_config(file_path: str) -> Dict[str, str]:
    """Reads existing config file and returns key-value pairs"""
    existing = {}
    path = Path(file_path)
    if path.exists():
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        existing[parts[0].strip()] = parts[1].strip()
    return existing

def update_model_paths_file(
    models: Set[str],
    paths: Dict[str, str],
    params: Dict[str, Any],
    file_path: str 
) -> None:
    """Updates config file, adding only new variables"""
    existing = read_existing_config(file_path)
    new_lines = []
    
    # Process models
    for model in models:
        varname = model_to_varname(model, PREFIX_MODEL)
        if varname not in existing:
            print(f"{LOG_PREFIX}: Adding Model rquirement to config: {model}")
            new_lines.append(f"{varname} = ./models/{model.split('/')[-1]}")
    
    # Process paths - now handles any path keys
    for key, value in paths.items():
        varname = model_to_varname(key, PREFIX_PATH)
        if varname not in existing:
            print(f"{LOG_PREFIX}: Adding path rquirement to config: {key}")
            new_lines.append(f"{varname} = {value}")
    
    # Process params
    for key, value in params.items():
        if key not in existing:
            print(f"{LOG_PREFIX}: Adding Parameter rquirement to config: {key}")
            new_lines.append(f"{key} = {value}")
    
    # Append new lines if any
    if new_lines:
        with open(file_path, 'a') as f:
            f.write("\n" + "\n".join(new_lines) + "\n")

def parse_model_paths_file(file_path: str , dotenv_needed_models, dotenv_needed_paths ) -> tuple[
    Set[str], Dict[str, str], Dict[str, Union[bool, int, float, str]]
]:
    """Reads config file and returns loaded variables"""
    loaded_models = {}
    loaded_paths = {}
    loaded_params = {}
    loaded_models_values= {}
    existing = read_existing_config(file_path)
    
    for key, value in existing.items():
        # Handle model paths
        if key.startswith(PREFIX_MODEL):
            for mod in dotenv_needed_models:
                #we find out if the current key value belongs to one of our models
                if key == model_to_varname(mod,PREFIX_MODEL):
                    #if a path has been defined and it exists we use the local path
                    if value and os.path.isdir(value):
                        loaded_models[mod] = value
                    else:
                        #else we use the model id so its downloaded from github later
                        loaded_models[mod] = mod
                    #still we collect the values to show to the user so he knows what to fix in config file
                    loaded_models_values[mod] = value
        # Handle ALL paths (not just HF_HOME)
        elif key.startswith(PREFIX_PATH):
            for mod in dotenv_needed_paths:
                if key == model_to_varname(mod,PREFIX_PATH):
                    loaded_paths[mod] = value
        # Handle params with type conversion
        else:
            if value.lower() in {"true", "false"}:
                loaded_params[key] = value.lower() == "true"
            elif value.isdigit():
                loaded_params[key] = int(value)
            else:
                try:
                    loaded_params[key] = float(value)
                except ValueError:
                    loaded_params[key] = value
    
    return loaded_models, loaded_paths, loaded_params, loaded_models_values

def is_online_model(model: str,dotenv_needed_models, debug_mode: bool = False) -> bool:
    """Checks if a model is in the online models set."""
    is_onlinemodel = model in dotenv_needed_models
    if debug_mode:
        print(f"Model '{model}' is online: {is_onlinemodel}")
    return is_onlinemodel

import os
def count_existing_paths(paths):
    """
    Checks if each path in the list exists.
    Returns:
        - summary (str): Summary of found/missing count
        - all_found (bool): True if all paths were found
        - none_found (bool): True if no paths were found
        - details (list of str): List with "[found]" or "[not found]" per path
    """
    total = len(paths)
    if total == 0:
        return "No paths provided.", False, True, []
    found_count = 0
    details = []
    for path in paths:
        if os.path.exists(path):
            found_count += 1
            details.append(f"[!FOUND!]: {path}")
        else:
            details.append(f"[MISSING]: {path}")
    missing_count = total - found_count
    all_found = (missing_count == 0)
    none_found = (found_count == 0)
    summary = f"Found {found_count}, missing {missing_count}, out of {total} paths."
    return summary, all_found, none_found, details


def remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[:-len(suffix)]
    return text


def get_hf_model_cache_dirname(model_id: str) -> str:
    """
    Returns the HF cache directory name for a given model.
    """
    base = "models--"
    return base + model_id.replace('/', '--')


def lcx_checkmodels(dotenv_needed_models,dotenv_loaded_paths, dotenv_loaded_models, dotenv_loaded_models_values  ):
    #TODO: load dynamically from array
    test_models_hf = []
    test_models_dir=[]
    test_paths_dir=[]
    
    #add model paths as path and as hf cache path
    for currmodel in dotenv_needed_models:
        test_models_hf.append(f"{dotenv_loaded_paths["HF_HOME"]}{os.sep}hub{os.sep}{get_hf_model_cache_dirname(currmodel)}{os.sep}snapshots")
        test_models_dir.append(f"{dotenv_loaded_models[  currmodel]}")
    
    #add needed dirs as path
    for curr_path in dotenv_loaded_paths:
        test_paths_dir.append(f"{dotenv_loaded_paths[  curr_path]}")
    
    if debug_mode:
        print(f"test pathf hf: {test_models_hf}")
        print(f"test pathf dirs: {test_models_dir}")
        
    print(f"{LCX_APP_NAME}: checking model accessibility")
    summary_hf, all_exist_hf, none_exist_hf, path_details_hf = count_existing_paths(test_models_hf)

    print(f"\n-Searching Group1: Model HF_HOME----------------------------------------------")
    for line in path_details_hf:
        print_line= remove_suffix(line, "snapshots")
        print(print_line)

    summary_dir, all_exist_dir, none_exist_dir, path_details_dir = count_existing_paths(test_models_dir)
    print("-Searching Group2: Manual Model Directories-----------------------------------")
    for line in path_details_dir:
        print_line= remove_suffix(line, "model_index.json")
        print_line= remove_suffix(print_line, "config.json")
        print(print_line)

    summary_path, all_exist_path, none_exist_path, path_details_path = count_existing_paths(test_paths_dir)
    print("-Searching Group3: Needed Directories-----------------------------------------")
    for line in path_details_path:
        print(line)


    print("")
    #we show the dir values to the user
    if all_exist_dir==False:
        print("-Values in config (resolved to your OS)---------------------------------------")
        for key in dotenv_loaded_models_values:
            print(f"{key}: {os.path.abspath(dotenv_loaded_models_values[key])}")
    if all_exist_path==False:
        for key in dotenv_loaded_paths:
            print(f"{key}: {os.path.abspath(dotenv_loaded_paths[  key])}")

    print("")
    
    #Needed Dirs summary
    if all_exist_path == False:
        print("-Needed Paths---------------------------------------------------") 
        print("Not all paths were found. Check documentation if you need them")

    print("-Needed Models--------------------------------------------------")
    #some model directories were missing 
    if none_exist_dir == False and all_exist_dir == False: 
        print ("Some manually downloaded models were found. Some might need to be downloaded!")
    #some hf cache models were missing
    if  all_exist_hf == False and none_exist_hf==False:
        print ("Some HF_Download models were found. Some might need to be downloaded!")
    if none_exist_dir and none_exist_hf:
        print ("No models were found! Models will be downloaded at next app start")

    if all_exist_hf==True or all_exist_dir==True:
        print("RESULT: It seems all models were found. Nothing will be downloaded!") 
    sys.exit()

# Update the config file
update_model_paths_file(in_dotenv_needed_models, in_dotenv_needed_paths, in_dotenv_needed_params, in_model_config_file)

# Read back the values
out_dotenv_loaded_models, out_dotenv_loaded_paths, out_dotenv_loaded_params , out_dotenv_loaded_models_values= parse_model_paths_file(in_model_config_file, in_dotenv_needed_models,in_dotenv_needed_paths)

if debug_mode:
    print("Loaded models:", out_dotenv_loaded_models)
    print("Loaded models values:", out_dotenv_loaded_models_values)
    print("Loaded paths:", out_dotenv_loaded_paths)
    print("Loaded params:", out_dotenv_loaded_params)
    
    
os.environ['HF_HOME'] = out_dotenv_loaded_paths["HF_HOME"]
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
#os.environ["TOKENIZERS_PARALLELISM"] = "true"


#originalblock#################################
import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--output_dir", type=str, default='./outputs')
parser.add_argument("--checkmodels", action='store_true')

parser.add_argument("--lora", type=str, default=None, help="Lora path (comma separated for multiple)")
parser.add_argument("--offline", action='store_true', help="Run in offline mode")

args = parser.parse_args()
###################################
# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

#return out_dotenv_loaded_models, out_dotenv_loaded_paths, out_dotenv_loaded_params 

if args.checkmodels: 
    lcx_checkmodels(in_dotenv_needed_models,out_dotenv_loaded_paths, out_dotenv_loaded_models, out_dotenv_loaded_models_values )


if debug_mode:
    print("---current model paths---------")
    for id in out_dotenv_loaded_models:
        print (f"{id}: {out_dotenv_loaded_models[id]}")

####################################################################################################################
####################################################################################################################
####################################################################################################################
#prefix end#########################################################################################################

 


import torch
import os # for offline loading path
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.memory import DynamicSwapInstaller
from .base_generator import BaseModelGenerator

class OriginalModelGenerator(BaseModelGenerator):
    """
    Model generator for the Original HunyuanVideo model.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Original model generator.
        """
        super().__init__(**kwargs)
        self.model_name = "Original"
        self.model_path = out_dotenv_loaded_models["lllyasviel/FramePackI2V_HY"]
        self.model_repo_id_for_cache = "models--lllyasviel--FramePackI2V_HY"
    
    def get_model_name(self):
        """
        Get the name of the model.
        """
        return self.model_name

    def load_model(self):
        """
        Load the Original transformer model.
        If offline mode is True, attempts to load from a local snapshot.
        """
        print(f"Loading {self.model_name} Transformer...") 
        
        path_to_load = self.model_path # Initialize with the default path

        if self.offline:
            path_to_load = self._get_offline_load_path() # Calls the method in BaseModelGenerator
        
        # Create the transformer model
        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            path_to_load, 
            torch_dtype=torch.bfloat16
        ).cpu()
        
        # Configure the model
        self.transformer.eval()
        self.transformer.to(dtype=torch.bfloat16)
        self.transformer.requires_grad_(False)
        
        # Set up dynamic swap if not in high VRAM mode
        if not self.high_vram:
            DynamicSwapInstaller.install_model(self.transformer, device=self.gpu)
        
        print(f"{self.model_name} Transformer Loaded from {path_to_load}.")
        return self.transformer
    
    def prepare_history_latents(self, height, width):
        """
        Prepare the history latents tensor for the Original model.
        
        Args:
            height: The height of the image
            width: The width of the image
            
        Returns:
            The initialized history latents tensor
        """
        return torch.zeros(
            size=(1, 16, 1 + 2 + 16, height // 8, width // 8), 
            dtype=torch.float32
        ).cpu()
    
    def get_latent_paddings(self, total_latent_sections):
        """
        Get the latent paddings for the Original model.
        
        Args:
            total_latent_sections: The total number of latent sections
            
        Returns:
            A list of latent paddings
        """
        # Original model uses reversed latent paddings
        if total_latent_sections > 4:
            return [3] + [2] * (total_latent_sections - 3) + [1, 0]
        else:
            return list(reversed(range(total_latent_sections)))
    
    def prepare_indices(self, latent_padding_size, latent_window_size):
        """
        Prepare the indices for the Original model.
        
        Args:
            latent_padding_size: The size of the latent padding
            latent_window_size: The size of the latent window
            
        Returns:
            A tuple of (clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices)
        """
        indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
        clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
        
        return clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices
    
    def prepare_clean_latents(self, start_latent, history_latents):
        """
        Prepare the clean latents for the Original model.
        
        Args:
            start_latent: The start latent
            history_latents: The history latents
            
        Returns:
            A tuple of (clean_latents, clean_latents_2x, clean_latents_4x)
        """
        clean_latents_pre = start_latent.to(history_latents)
        clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
        
        return clean_latents, clean_latents_2x, clean_latents_4x
    
    def update_history_latents(self, history_latents, generated_latents):
        """
        Update the history latents with the generated latents for the Original model.
        
        Args:
            history_latents: The history latents
            generated_latents: The generated latents
            
        Returns:
            The updated history latents
        """
        # For Original model, we prepend the generated latents
        return torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
    
    def get_real_history_latents(self, history_latents, total_generated_latent_frames):
        """
        Get the real history latents for the Original model.
        
        Args:
            history_latents: The history latents
            total_generated_latent_frames: The total number of generated latent frames
            
        Returns:
            The real history latents
        """
        return history_latents[:, :, :total_generated_latent_frames, :, :]
    
    def update_history_pixels(self, history_pixels, current_pixels, overlapped_frames):
        """
        Update the history pixels with the current pixels for the Original model.
        
        Args:
            history_pixels: The history pixels
            current_pixels: The current pixels
            overlapped_frames: The number of overlapped frames
            
        Returns:
            The updated history pixels
        """
        from diffusers_helper.utils import soft_append_bcthw
        # For Original model, current_pixels is first, history_pixels is second
        return soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
    
    def get_section_latent_frames(self, latent_window_size, is_last_section):
        """
        Get the number of section latent frames for the Original model.
        
        Args:
            latent_window_size: The size of the latent window
            is_last_section: Whether this is the last section
            
        Returns:
            The number of section latent frames
        """
#        return latent_window_size * 2
        return (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
    
    def get_current_pixels(self, real_history_latents, section_latent_frames, vae):
        """
        Get the current pixels for the Original model.
        
        Args:
            real_history_latents: The real history latents
            section_latent_frames: The number of section latent frames
            vae: The VAE model
            
        Returns:
            The current pixels
        """
        from diffusers_helper.hunyuan import vae_decode
        return vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
       
    def format_position_description(self, total_generated_latent_frames, current_pos, original_pos, current_prompt):
        """
        Format the position description for the Original model.
        
        Args:
            total_generated_latent_frames: The total number of generated latent frames
            current_pos: The current position in seconds
            original_pos: The original position in seconds
            current_prompt: The current prompt
            
        Returns:
            The formatted position description
        """
        return (f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, '
                f'Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30). '
                f'Current position: {current_pos:.2f}s (original: {original_pos:.2f}s). '
                f'using prompt: {current_prompt[:256]}...')
