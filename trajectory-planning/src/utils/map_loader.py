import os
import json
from pathlib import Path

class MapLoader:
    """Utility for loading map configurations from files."""
    
    def __init__(self, maps_dir=None):
        """Initialize the map loader with the maps directory."""
        if maps_dir is None:
            # Default to the maps directory relative to the project root
            # This assumes the script is run from the project root
            self.maps_dir = Path(__file__).parent.parent.parent / 'maps' / 'scenarios'
        else:
            self.maps_dir = Path(maps_dir)
            
        if not self.maps_dir.exists():
            raise FileNotFoundError(f"Maps directory not found: {self.maps_dir}")
    
    def list_available_maps(self):
        """List all available map files in the maps directory."""
        map_files = list(self.maps_dir.glob('*.json'))
        maps = []
        
        for map_file in map_files:
            try:
                with open(map_file, 'r') as f:
                    map_data = json.load(f)
                    maps.append({
                        'filename': map_file.name,
                        'name': map_data.get('name', map_file.stem),
                        'description': map_data.get('description', '')
                    })
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {map_file} as JSON")
                
        return maps
    
    def load_map(self, map_name):
        """
        Load a map configuration from file.
        
        Args:
            map_name: Either the filename (with or without .json extension) or
                     the name property in the map file
        
        Returns:
            dict: Map configuration with keys: map_boundary, obstacles, start_point, end_point
        """
        # Check if map_name is a filename or a map name
        if not map_name.endswith('.json'):
            map_name += '.json'
            
        map_path = self.maps_dir / map_name
        
        # If not found directly, try to find by map name
        if not map_path.exists():
            # List all maps and find by name
            maps = self.list_available_maps()
            for map_info in maps:
                if map_info['name'].lower() == map_name.lower():
                    map_path = self.maps_dir / map_info['filename']
                    break
        
        if not map_path.exists():
            raise FileNotFoundError(f"Map file not found: {map_name}")
            
        with open(map_path, 'r') as f:
            map_data = json.load(f)
            
        return map_data