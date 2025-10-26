"""Product segmentation and fill level analysis using Gemini Vision."""

import json
import os
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


DETECTION_PROMPT = """
You are an annotation engine. Look at the image and enumerate EVERY distinct, non-overlapping, real-world object fully or partially visible in the frame (e.g., bottles, tubes, jars, boxes). For each object, output EXACTLY one JSON object with this schema and key order:

[
  {
    "box_2d": [y0, x0, y1, x1],
    "label": "<concise object name + short attribute/description>",
    "percent_full": N,
    "is_low": true|false,
    "confidence": C
  },
  ...
]

STRICT RULES (follow exactly):
- OUTPUT FORMAT: Respond ONLY with a Markdown JSON code block. First line MUST be ```json and last line MUST be ```. No prose before/after. No explanations. No comments.
- JSON: Strict JSON; no trailing commas; double-quoted keys/strings; numbers only for percent_full/confidence; booleans are true/false (lowercase).
- SORTING: Sort objects by the top-left corner of the box (ascending x0, then ascending y0) for deterministic order.
- LABELING: Use generic names unless the brand text is clearly readable in the image; do NOT hallucinate brands or models.
- PERCENT FULL HEURISTICS (deterministic): 
  1) Transparent/Translucent containers: estimate fill line via visible liquid boundary within the mask's vertical extent. 
  2) Opaque containers with sight window: use the visible window only. 
  3) Fully opaque with no cues: use 100 if unopened-sealed cues; otherwise 50 by default. 
  4) Squeezables (tubes): infer from creases/flattening; if uncertain, default 40. 
  Clip the final value to [0,100] and round to nearest integer.
- CONFIDENCE: 0.9 when a clear fill boundary/window exists; 0.6 when inferred from shape/creases; 0.4 when no visual cues.
- BOX_2D: Use [y0, x0, y1, x1] normalized to 0–1000 (integers). Ensure y0<y1 and x0<x1. The box must tightly enclose the object's mask.

IF NOTHING IS DETECTED: Return an empty JSON array [] in the required code block.
"""


class JSONParser:
    """Parses JSON from Gemini API responses."""
    
    @staticmethod
    def parse_json(json_output: str) -> str:
        """
        Extract JSON from markdown code blocks or find JSON directly.
        
        Args:
            json_output: Raw output from Gemini API
            
        Returns:
            Extracted JSON string
        """
        lines = json_output.splitlines()
        json_start = -1
        json_end = -1
        
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                json_start = i + 1
            elif line.strip() == "```" and json_start != -1:
                json_end = i
                break
        
        if json_start != -1 and json_end != -1:
            json_content = "\n".join(lines[json_start:json_end])
            print(f"Extracted JSON from markdown blocks: {len(json_content)} characters")
            return json_content.strip()
        
        print("No markdown blocks found, searching for JSON directly")
        
        start_idx = json_output.find('[')
        if start_idx != -1:
            print(f"Found '[' at position {start_idx}")
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
                return json_output[start_idx:]
        
        print("No JSON found, returning original string")
        return json_output


class ProductSegmenter:
    """Segments and analyzes products in images."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """Initialize the segmenter with model."""
        self.model = genai.GenerativeModel(model_name)
        self.parser = JSONParser()
    
    def load_and_preprocess_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image for API processing."""
        im = Image.open(image_path)
        
        if im.mode != 'RGB':
            im = im.convert('RGB')
        
        im.thumbnail([512, 512], Image.Resampling.LANCZOS)
        print(f"Image resized to: {im.size}")
        
        return im
    
    def detect_products(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect products in the image using Gemini Vision.
        
        Args:
            image: PIL Image object
            
        Returns:
            List of detected products with their properties
        """
        response = self.model.generate_content([DETECTION_PROMPT, image])
        
        try:
            parsed_json = self.parser.parse_json(response.text)
            print(f"Parsed JSON length: {len(parsed_json)} characters")
            
            if not parsed_json.strip().startswith('['):
                print("JSON doesn't start with '[' - trying to find array start")
                start_idx = parsed_json.find('[')
                if start_idx != -1:
                    parsed_json = parsed_json[start_idx:]
            
            if not parsed_json.strip().endswith(']'):
                print("JSON doesn't end with ']' - trying to complete it")
                last_brace = parsed_json.rfind('}')
                if last_brace != -1:
                    end_pos = last_brace + 1
                    parsed_json = parsed_json[:end_pos] + ']'
                    print(f"Attempted to complete JSON by adding ']'")
            
            print(f"First 200 chars: {parsed_json[:200]}")
            print(f"Last 200 chars: {parsed_json[-200:]}")
            
            items = json.loads(parsed_json)
            print(f"Successfully parsed {len(items)} items")
            return items
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response length: {len(response.text)}")
            print(f"Raw response first 500 chars: {response.text[:500]}")
            
            try:
                last_brace = parsed_json.rfind('}')
                if last_brace != -1:
                    partial_json = parsed_json[:last_brace + 1] + ']'
                    items = json.loads(partial_json)
                    print(f"Successfully parsed partial JSON with {len(items)} items")
                    return items
                else:
                    print("No complete objects found")
                    return []
            except:
                print("Failed to parse even partial JSON")
                return []
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            print(f"Raw response: {response.text[:500]}...")
            return []
    
    def create_overlay(
        self, 
        image: Image.Image, 
        items: List[Dict[str, Any]], 
        output_dir: str
    ) -> None:
        """
        Create annotated overlay images for detected products.
        
        Args:
            image: Original PIL Image
            items: List of detected products
            output_dir: Directory to save output images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, item in enumerate(items):
            print(f"\nProcessing item {i+1}: {item['label']}")
            
            box = item["box_2d"]
            y0 = int(box[0] / 1000 * image.size[1])
            x0 = int(box[1] / 1000 * image.size[0])
            y1 = int(box[2] / 1000 * image.size[1])
            x1 = int(box[3] / 1000 * image.size[0])
            
            if y0 >= y1 or x0 >= x1:
                print(f"  Skipping invalid box: {box}")
                continue
            
            print(f"  Box coordinates: ({x0}, {y0}) to ({x1}, {y1})")
            print(f"  Percent full: {item['percent_full']}%")
            print(f"  Is low: {item['is_low']}")
            print(f"  Confidence: {item['confidence']}")
            
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            overlay_draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=3)
            
            try:
                overlay_draw.text(
                    (x0, y0 - 20), 
                    f"{item['label']} ({item['percent_full']}%)", 
                    fill=(255, 0, 0, 255)
                )
            except:
                pass
            
            overlay_filename = f"{item['label'].replace(' ', '_')}_{i}_overlay.png"
            composite = Image.alpha_composite(image.convert('RGBA'), overlay)
            composite.save(os.path.join(output_dir, overlay_filename))
            print(f"  Saved overlay: {overlay_filename}")


def extract_segmentation_masks(
    image_path: str, 
    output_dir: str = "segmentation_outputs"
) -> None:
    """
    Extract and analyze products from an image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory for output files
    """
    segmenter = ProductSegmenter()
    
    image = segmenter.load_and_preprocess_image(image_path)
    items = segmenter.detect_products(image)
    segmenter.create_overlay(image, items, output_dir)


def main():
    """Main execution function."""
    test_images = ["soap.jpg", "multiple_products.png", "water_bottle1.jpg"]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n{'=' * 60}")
            print(f"Processing: {image_path}")
            print('=' * 60)
            extract_segmentation_masks(image_path)
        else:
            print(f"Warning: {image_path} not found, skipping")


if __name__ == "__main__":
    main()
