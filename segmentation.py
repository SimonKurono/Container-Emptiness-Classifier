import google.generativeai as genai
from google.generativeai import types
from PIL import Image, ImageDraw
import io
import base64
import json
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def parse_json(json_output: str):
  # Parsing out the markdown fencing
  lines = json_output.splitlines()
  json_start = -1
  json_end = -1
  
  # Find the start and end of JSON block
  for i, line in enumerate(lines):
    if line.strip() == "```json":
      json_start = i + 1
    elif line.strip() == "```" and json_start != -1:
      json_end = i
      break
  
  if json_start != -1 and json_end != -1:
    # Extract JSON content between the markers
    json_content = "\n".join(lines[json_start:json_end])
    print(f"Extracted JSON from markdown blocks: {len(json_content)} characters")
    return json_content.strip()
  
  # If no markdown blocks found, try to find JSON directly
  print("No markdown blocks found, searching for JSON directly")
  
  # Look for the first '[' and try to find matching ']'
  start_idx = json_output.find('[')
  if start_idx != -1:
    print(f"Found '[' at position {start_idx}")
    # Try to find the matching ']' by counting brackets
    bracket_count = 0
    end_idx = start_idx
    for i in range(start_idx, len(json_output)):
      if json_output[i] == '[':
        bracket_count += 1
      elif json_output[i] == ']':
        bracket_count -= 1
        if bracket_count == 0:
          end_idx = i
          break
    
    if bracket_count == 0:
      json_content = json_output[start_idx:end_idx + 1]
      print(f"Found complete JSON: {len(json_content)} characters")
      return json_content
    else:
      print(f"JSON appears incomplete, bracket count: {bracket_count}")
      # Return what we have and let the main function handle it
      return json_output[start_idx:]
  
  # If all else fails, return the original string
  print("No JSON found, returning original string")
  return json_output

def _create_simple_mask(width: int, height: int) -> Image.Image:
    """Create a simple rectangular mask as fallback"""
    mask = Image.new('L', (width, height), 255)  # White mask (fully opaque)
    return mask

def extract_segmentation_masks(image_path: str, output_dir: str = "segmentation_outputs"):
  # Load and resize image more aggressively
  im = Image.open(image_path)
  # Convert to RGB if needed (JPG to PNG conversion)
  if im.mode != 'RGB':
    im = im.convert('RGB')
  
  # Resize more aggressively to reduce API response size
  im.thumbnail([512, 512], Image.Resampling.LANCZOS)
  print(f"Image resized to: {im.size}")

  prompt = """
            You are an annotation engine. Look at the image and enumerate EVERY distinct, non-overlapping, real-world object fully or partially visible in the frame (e.g., bottles, tubes, jars, boxes). For each object, output EXACTLY one JSON object with this schema and key order:

[
  {
    "box_2d": [y0, x0, y1, x1],            // integers in [0,1000], normalized to image height/width; y before x; y0<x1 etc.
    "label": "<concise object name + short attribute/description>", // e.g., "shampoo bottle, white, pump"
    "percent_full": N,                      // integer 0–100 estimating how full the container is; 0 = empty, 100 = full; if not a container, use 100
    "is_low": true|false,                   // true iff percent_full < 25
    "confidence": C                         // float between 0.0 and 1.0 for the percent_full estimate
  },
  ...
]

STRICT RULES (follow exactly):
- OUTPUT FORMAT: Respond ONLY with a Markdown JSON code block. First line MUST be ```json and last line MUST be ```. No prose before/after. No explanations. No comments.
- JSON: Strict JSON; no trailing commas; double-quoted keys/strings; numbers only for percent_full/confidence; booleans are true/false (lowercase).
- SORTING: Sort objects by the top-left corner of the box (ascending x0, then ascending y0) for deterministic order.
- LABELING: Use generic names unless the brand text is clearly readable in the image; do NOT hallucinate brands or models.
- PERCENT FULL HEURISTICS (deterministic): 
  1) Transparent/Translucent containers: estimate fill line via visible liquid boundary within the mask’s vertical extent. 
  2) Opaque containers with sight window: use the visible window only. 
  3) Fully opaque with no cues: use 100 if unopened-sealed cues; otherwise 50 by default. 
  4) Squeezables (tubes): infer from creases/flattening; if uncertain, default 40. 
  Clip the final value to [0,100] and round to nearest integer.
- CONFIDENCE: 0.9 when a clear fill boundary/window exists; 0.6 when inferred from shape/creases; 0.4 when no visual cues.
- BOX_2D: Use [y0, x0, y1, x1] normalized to 0–1000 (integers). Ensure y0<y1 and x0<x1. The box must tightly enclose the object’s mask.

IF NOTHING IS DETECTED: Return an empty JSON array [] in the required code block.

            """

  model = genai.GenerativeModel("gemini-2.5-flash")
  response = model.generate_content([prompt, im])

  # Parse JSON response
  try:
    parsed_json = parse_json(response.text)
    print(f"Parsed JSON length: {len(parsed_json)} characters")
    
    # Try to fix common JSON issues
    if not parsed_json.strip().startswith('['):
      print("JSON doesn't start with '[' - trying to find array start")
      start_idx = parsed_json.find('[')
      if start_idx != -1:
        parsed_json = parsed_json[start_idx:]
    
    # Check if JSON is complete (ends with ']')
    if not parsed_json.strip().endswith(']'):
      print("JSON doesn't end with ']' - trying to complete it")
      # Find the last complete object
      last_brace = parsed_json.rfind('}')
      if last_brace != -1:
        # Find the position after the last complete object
        end_pos = last_brace + 1
        # Add closing bracket
        parsed_json = parsed_json[:end_pos] + ']'
        print(f"Attempted to complete JSON by adding ']'")
    
    print(f"First 200 chars: {parsed_json[:200]}")
    print(f"Last 200 chars: {parsed_json[-200:]}")
    
    items = json.loads(parsed_json)
    print(f"Successfully parsed {len(items)} items")
    
  except json.JSONDecodeError as e:
    print(f"JSON parsing error: {e}")
    print(f"Raw response length: {len(response.text)}")
    print(f"Raw response first 500 chars: {response.text[:500]}")
    print(f"Raw response last 500 chars: {response.text[-500:]}")
    print(f"Parsed JSON length: {len(parsed_json)}")
    print(f"Parsed JSON first 500 chars: {parsed_json[:500]}")
    print(f"Parsed JSON last 500 chars: {parsed_json[-500:]}")
    
    # Try to extract partial JSON
    try:
      # Find the last complete object
      last_brace = parsed_json.rfind('}')
      if last_brace != -1:
        partial_json = parsed_json[:last_brace + 1] + ']'
        items = json.loads(partial_json)
        print(f"Successfully parsed partial JSON with {len(items)} items")
      else:
        print("No complete objects found")
        items = []
    except:
      print("Failed to parse even partial JSON")
      items = []
      
  except Exception as e:
    print(f"Unexpected error: {e}")
    print(f"Raw response: {response.text[:500]}...")
    items = []

  # Create output directory
  os.makedirs(output_dir, exist_ok=True)

  # Process each item (simplified - no masks, just bounding boxes)
  for i, item in enumerate(items):
      print(f"\nProcessing item {i+1}: {item['label']}")
      
      # Get bounding box coordinates
      box = item["box_2d"]
      y0 = int(box[0] / 1000 * im.size[1])
      x0 = int(box[1] / 1000 * im.size[0])
      y1 = int(box[2] / 1000 * im.size[1])
      x1 = int(box[3] / 1000 * im.size[0])

      # Skip invalid boxes
      if y0 >= y1 or x0 >= x1:
          print(f"  Skipping invalid box: {box}")
          continue

      print(f"  Box coordinates: ({x0}, {y0}) to ({x1}, {y1})")
      print(f"  Percent full: {item['percent_full']}%")
      print(f"  Is low: {item['is_low']}")
      print(f"  Confidence: {item['confidence']}")

      # Create a simple bounding box overlay
      overlay = Image.new('RGBA', im.size, (0, 0, 0, 0))
      overlay_draw = ImageDraw.Draw(overlay)
      
      # Draw bounding box
      overlay_draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=3)
      
      # Add label text
      try:
          overlay_draw.text((x0, y0-20), f"{item['label']} ({item['percent_full']}%)", fill=(255, 0, 0, 255))
      except:
          pass  # Skip text if font not available

      # Save overlay
      overlay_filename = f"{item['label'].replace(' ', '_')}_{i}_overlay.png"
      composite = Image.alpha_composite(im.convert('RGBA'), overlay)
      composite.save(os.path.join(output_dir, overlay_filename))
      print(f"  Saved overlay: {overlay_filename}")

# Example usage
if __name__ == "__main__":
  extract_segmentation_masks("soap.jpg")
  extract_segmentation_masks("multiple_products.png")
  extract_segmentation_masks("water_bottle1.jpg")
