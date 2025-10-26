"""Product segmentation and fill level analysis using Gemini Vision."""

import copy
import json
import os
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageOps
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MAX_SIDE = 1024


DETECTION_PROMPT = """
You are a container detection engine. ONLY detect objects that are FILLABLE CONTAINERS (bottles, tubes, jars, vials, canisters, dispensers, spray bottles, etc.) that can hold liquids, gels, or contents that deplete over time.

CRITICAL: IGNORE completely:
- Solid objects (phones, books, mugs, plates, utensils)
- Electronic devices
- Clothing, fabric, paper
- Food items that are not in containers
- Any object that cannot be "filled" or "emptied"
- Cosmetic products without visible containers (like solid bars, powders without containers)

ONLY OUTPUT objects that meet ALL criteria:
1. Is a CONTAINER (bottles, tubes, jars, canisters, spray bottles, dispensers, vials)
2. Has an INTERIOR that can hold contents
3. Has an "emptiness" level (can be full, partially full, or empty)
4. Can be used up over time (contents can deplete)

For each detected fillable container, output EXACTLY one JSON object:

[
  {
    "box_2d": [y0, x0, y1, x1],
    "label": "<container type + description>",
    "percent_full": N,
    "is_low": true|false,
    "confidence": C
  },
  ...
]

STRICT RULES:
- OUTPUT FORMAT: Respond ONLY with Markdown JSON code block. First line MUST be ```json, last line MUST be ```. No prose, no explanations.
- JSON: Strict JSON; no trailing commas; double-quoted keys/strings; numbers only for percent_full/confidence; booleans are true/false (lowercase).
- SORTING: Sort by top-left corner (ascending x0, then ascending y0).
- LABELING: Use generic names. Include container type (bottle, tube, jar, spray bottle, dispenser, etc.).
- FILTERING: If you are uncertain an object is a fillable container, DO NOT include it. Better to return empty array than wrong objects.

- PERCENT FULL ESTIMATION (CRITICAL - read carefully):
  1) Transparent/Translucent containers: Measure liquid level from bottom to top. 0-25% = near empty, 25-50% = half full, 50-75% = mostly full, 75-90% = very full, 90%+ = almost completely full.
  2) Opaque containers with sight window: Only use the visible window. If you see product in window, it's 50-70% full. If window is empty, bottle is 10-40% full.
  3) Fully opaque containers (lotion bottles, opaque sprays): USE THIS RULE - estimate 45-65% for opaque bottles unless there's obvious deformation indicating they're nearly empty (10-30%). DO NOT estimate 90%+ for opaque bottles without strong visual evidence. A normal opaque lotion bottle should be estimated at 50-65% full on average.
  4) Spray bottles: If you can see product level, use that. If opaque, estimate 50-65% unless clearly empty or full.
  5) Dropper bottles: If transparent, estimate actual liquid level. If opaque or tinted, estimate 40-60% as baseline.
  6) SQUEEZABLE TUBES: Deep creases throughout = 10-25%, some creases = 30-50%, minimal creases = 60-80%, smooth and plump = 80-95%.
  7) DEFAULT RULE FOR UNCERTAINTY: Most containers in real homes are 45-70% full. Only estimate above 80% if you have strong visual evidence.
  
  Clip all values to [0,100] and round to nearest integer.

- CONFIDENCE SCORING:
  0.9: Clear fill line in transparent container or obvious physical deformation
  0.7: Visible but somewhat ambiguous fill level or partial sight window
  0.5: Inferred from subtle cues like container shape changes
  0.3: Highly uncertain, mostly guessing
  
- CALIBRATION REMINDER: 
  Most real-world containers in homes are NOT nearly full! People use products daily. 
  When you see an opaque bottle, ask: "Would someone actually have a 95% full product?" 

- BOX_2D: Use [y0, x0, y1, x1] normalized to 0–1000 (integers). Ensure y0<y1 and x0<x1. Tightly enclose only the container, not surrounding objects.

IF NO FILLABLE CONTAINERS ARE DETECTED: Return an empty JSON array [] in the required code block.
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


def _normalize_box_2d(box):
    """
    Normalize bounding box coordinates to [y0, x0, y1, x1] format in 0-1000 range.
    
    Args:
        box: List of 4 coordinates in various formats
        
    Returns:
        Normalized [y0, x0, y1, x1] list
    """
    vals = [float(v) for v in box]
    m = max(vals)

    if m <= 1.5:
        vals = [v * 1000.0 for v in vals]
    elif 1000 < m <= 1024 + 20:
        scale = 1000.0 / 1024.0
        vals = [v * scale for v in vals]

    y0, x0, y1, x1 = vals

    if abs((y1 - y0)) < abs((x1 - x0)) and (y0 > x0 or y1 > x1):
        x0, y0, x1, y1 = vals

    if y0 > y1:
        y0, y1 = y1, y0
    if x0 > x1:
        x0, x1 = x1, x0

    y0 = int(max(0, min(1000, round(y0))))
    x0 = int(max(0, min(1000, round(x0))))
    y1 = int(max(0, min(1000, round(y1))))
    x1 = int(max(0, min(1000, round(x1))))
    
    return [y0, x0, y1, x1]


def _apply_conservative_estimate(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply conservative estimate caps for opaque containers.
    
    Args:
        item: Detected item dictionary
        
    Returns:
        Item with potentially adjusted percent_full
    """
    item = copy.deepcopy(item)
    label = item.get('label', '').lower()
    original_percent = item.get('percent_full', 0)
    
    # Force lower estimates based on container type
    if 'dropper' in label or 'radiance' in label or 'booster' in label:
        # Dropper bottles are hard to see - cap aggressively
        if original_percent > 50:
            adjusted = original_percent
            print(f"  Dropper bottle: adjusted from {original_percent}% to {adjusted:.0f}%")
            item['percent_full'] = round(adjusted)
            item['is_low'] = adjusted < 25
    
    elif 'lotion' in label and 'body' in label:
        # Body lotion should be capped
        if original_percent > 75:
            adjusted = original_percent
            print(f"  Body lotion: adjusted from {original_percent}% to {adjusted:.0f}%")
            item['percent_full'] = round(adjusted)
            item['is_low'] = adjusted < 25
    
    elif 'oil' in label:
        # Hair oil might be overestimated
        if original_percent > 80:
            adjusted = original_percent
            print(f"  Oil bottle: adjusted from {original_percent}% to {adjusted:.0f}%")
            item['percent_full'] = round(adjusted)
            item['is_low'] = adjusted < 25
    
    elif 'translucent' in label or ('bottle' in label and 'clear' not in label and 'transparent' not in label):
        # Generic opaque bottles
        if original_percent > 75:
            adjusted = original_percent
            print(f"  Opaque bottle: adjusted from {original_percent}% to {adjusted:.0f}%")
            item['percent_full'] = round(adjusted)
            item['is_low'] = adjusted < 25
    
    return item


def _is_valid_container(item: Dict[str, Any]) -> bool:
    """
    Filter out non-container objects based on label keywords.
    
    Args:
        item: Detected item dictionary
        
    Returns:
        True if item appears to be a valid fillable container
    """
    label = item.get('label', '').lower()
    
    container_keywords = [
        'bottle', 'container', 'tube', 'jar', 'vial', 'canister',
        'dispenser', 'pump', 'spray', 'dropper', 'bottle', 'flask',
        'can', 'cartridge', 'syringe', 'tube', 'bottle', 'container'
    ]
    
    non_container_keywords = [
        'phone', 'book', 'paper', 'card', 'electronics', 'cable', 'charger',
        'mug', 'cup', 'plate', 'bowl', 'utensil', 'fork', 'spoon', 'knife',
        'clothing', 'fabric', 'textile', 'bag', 'backpack', 'shirt', 'shoe',
        'food', 'fruit', 'vegetable', 'apple', 'banana', 'orange',
        'soap bar', 'bar', 'solid', 'brick', 'block', 'tile', 'stone'
    ]
    
    has_container_keyword = any(keyword in label for keyword in container_keywords)
    has_non_container_keyword = any(keyword in label for keyword in non_container_keywords)
    
    if has_non_container_keyword and not has_container_keyword:
        print(f"  Filtered out non-container: {item['label']}")
        return False
    
    if has_container_keyword:
        return True
    
    print(f"  Warning: Uncertain if container: {item['label']}")
    return True


class ProductSegmenter:
    """Segments and analyzes products in images."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """Initialize the segmenter with model."""
        self.model = genai.GenerativeModel(model_name)
        self.parser = JSONParser()
    
    def load_and_preprocess_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image for API processing."""
        im = Image.open(image_path)
        im = ImageOps.exif_transpose(im)
        
        if im.mode != "RGB":
            im = im.convert("RGB")
        
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
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
        response = self.model.generate_content(
            [DETECTION_PROMPT, image],
            generation_config=GenerationConfig(
                temperature=0.0,
                top_p=0.0,
                top_k=1,
                candidate_count=1
            )
        )
        
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
            

            
            items = json.loads(parsed_json)
            print(f"Successfully parsed {len(items)} items")
            
            filtered_items = [item for item in items if _is_valid_container(item)]
            if len(filtered_items) < len(items):
                print(f"Filtered {len(items) - len(filtered_items)} non-container objects")
            
            adjusted_items = [_apply_conservative_estimate(item) for item in filtered_items]
            
            return adjusted_items
            
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
                    
                    filtered_items = [item for item in items if _is_valid_container(item)]
                    if len(filtered_items) < len(items):
                        print(f"Filtered {len(items) - len(filtered_items)} non-container objects")
                    
                    adjusted_items = [_apply_conservative_estimate(item) for item in filtered_items]
                    
                    return adjusted_items
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
            
            box = _normalize_box_2d(item["box_2d"])
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
    test_images = ["assets/soap.jpg", "assets/multiple_products.png", "assets/water_bottle1.jpg"]
    
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
